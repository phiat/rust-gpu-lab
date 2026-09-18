//! GPU side: the Margolus block step, paint application and rendering.
//!
//! A cellular automaton where cells *move* can't be a plain stencil: two
//! cells would try to fall into the same spot. The Margolus scheme fixes
//! that by partitioning the grid into 2x2 blocks, updating each block on
//! its own (a block's four cells only trade places among themselves), and
//! shifting the partition by one cell on alternate passes so blocks talk
//! to their neighbors. Every cell therefore reads its 3x3 neighborhood,
//! picks the three cells of its block out of it, runs the block rule, and
//! keeps the result for its own position. All four cells of a block compute
//! the same thing, so nothing has to be shared.

use std::sync::Arc;

use cuda_core::Stream;
use cutile::prelude::*;
use tilekit::{Eager, Pinned, Submit};

use crate::world::Layout;

pub const TILE: usize = 32;
pub const PARAMS: usize = 16;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    type I = Tile<i32, { [32, 32] }>;
    type Mask = Tile<bool, { [32, 32] }>;

    fn fill(v: i32) -> I {
        broadcast_scalar(v, const_shape![32, 32])
    }

    fn or(a: Mask, b: Mask) -> Mask {
        select(a, a, b)
    }

    fn and(a: Mask, b: Mask) -> Mask {
        select(a, b, a)
    }

    fn not(a: Mask) -> Mask {
        let f: Mask = lt_tile(fill(1i32), fill(0i32));
        let t: Mask = lt_tile(fill(0i32), fill(1i32));
        select(a, f, t)
    }

    fn is(v: I, k: i32) -> Mask {
        eq_tile(v, fill(k))
    }

    /// Buffer row and column of every cell in this tile.
    fn coords() -> (I, I) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let idx: Tile<i32, { [32] }> = iota(const_shape![32]);
        let row0: Tile<i32, { [32] }> = broadcast_scalar(pid.0 * 32i32, const_shape![32]);
        let col0: Tile<i32, { [32] }> = broadcast_scalar(pid.1 * 32i32, const_shape![32]);
        let rows: Tile<i32, { [32] }> = idx + row0;
        let cols: Tile<i32, { [32] }> = idx + col0;
        let rows: Tile<i32, { [32, 1] }> = rows.reshape(const_shape![32, 1]);
        let cols: Tile<i32, { [1, 32] }> = cols.reshape(const_shape![1, 32]);
        (
            rows.broadcast(const_shape![32, 32]),
            cols.broadcast(const_shape![32, 32]),
        )
    }

    fn param(t: &Tensor<i32, { [-1] }>, k: i32) -> I {
        let p: Tile<i32, { [16] }> = t.partition(const_shape![16]).load([0i32]);
        let idx: Tile<i32, { [] }> = scalar_to_tile(k);
        let v: Tile<i32, { [1] }> = extract(p, [idx]);
        let v: Tile<i32, { [1, 1] }> = v.reshape(const_shape![1, 1]);
        v.broadcast(const_shape![32, 32])
    }

    /// The pass number, broadcast from a `[4]` view of the pass table.
    fn pass_index(t: &Tensor<i32, { [-1] }>) -> I {
        let p: Tile<i32, { [4] }> = t.partition(const_shape![4]).load([0i32]);
        let idx: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let v: Tile<i32, { [1] }> = extract(p, [idx]);
        let v: Tile<i32, { [1, 1] }> = v.reshape(const_shape![1, 1]);
        v.broadcast(const_shape![32, 32])
    }

    // ---- Rules, mirroring cpu.rs -------------------------------------------

    fn kind(v: I) -> I {
        andi(v, fill(255i32))
    }

    fn cell(k: i32, shade: I) -> I {
        ori(fill(k), shade * fill(256i32))
    }

    /// Per-element tables as bit masks indexed by kind, so a test is a
    /// shift and a mask instead of a chain of compares. (Every op here is
    /// inlined at each call site, and this kernel has hundreds of them; the
    /// compare chains doubled its compile time.)
    fn has(k: I, table: i32) -> Mask {
        eq_tile(andi(shri(fill(table), k), fill(1i32)), fill(1i32))
    }

    fn lacks(k: I, table: i32) -> Mask {
        eq_tile(andi(shri(fill(table), k), fill(1i32)), fill(0i32))
    }

    /// Densities of kinds 0-7, 4 bits each: empty 2, sand 5, water 4,
    /// oil 3, wood 9, wall 9, fire 1, smoke 0. Ember (8) is 9.
    fn density(k: I) -> I {
        let d: I = andi(shri(fill(0x01993452i32), k * fill(4i32)), fill(15i32));
        select(is(k, 8i32), fill(9i32), d)
    }

    fn solid(k: I) -> Mask {
        has(k, 0x130i32) // wood, wall, ember
    }

    fn not_solid(k: I) -> Mask {
        lacks(k, 0x130i32)
    }

    fn liquid(k: I) -> Mask {
        has(k, 0x0ci32) // water, oil
    }

    fn falls(k: I) -> Mask {
        has(k, 0x0ei32) // sand, water, oil
    }

    fn rises(k: I) -> Mask {
        has(k, 0xc0i32) // fire, smoke
    }

    fn spreads(k: I) -> Mask {
        has(k, 0xcdi32) // empty, water, oil, fire, smoke
    }

    /// lowbias32. The `*` operator emits `muli` with no overflow flag, so
    /// it wraps like `wrapping_mul` (`muli(.., overflow::None)` fails to
    /// serialize, as `trunci` did in filters). `shri` is arithmetic on
    /// `i32`, like Rust's `>>`.
    fn hash(x: I) -> I {
        let x: I = xori(x, shri(x, fill(16i32)));
        let x: I = x * fill(0x7feb_352di32);
        let x: I = xori(x, shri(x, fill(15i32)));
        let x: I = x * fill(-2073254261i32); // 0x846ca68b
        xori(x, shri(x, fill(16i32)))
    }

    fn near(k1: I, k2: I, k3: I, k: i32) -> Mask {
        or(or(is(k1, k), is(k2, k)), is(k3, k))
    }

    fn react(x: I, o1: I, o2: I, o3: I, byte: I) -> I {
        let k: I = kind(x);
        let (k1, k2, k3) = (kind(o1), kind(o2), kind(o3));
        let near_fire: Mask = near(k1, k2, k3, 6i32);
        let near_ember: Mask = near(k1, k2, k3, 8i32);
        let near_water: Mask = near(k1, k2, k3, 2i32);
        let heat: Mask = or(near_fire, near_ember);
        let shade: I = shri(byte, fill(4i32));
        let fire: I = cell(6i32, shade);
        let smoke: I = cell(7i32, shade);
        let empty: I = fill(0i32);
        // Fire: out with water or by chance, to smoke or nothing.
        let fire_out: I = select(lt_tile(byte, fill(20i32)), empty, x);
        let fire_out: I = select(or(near_water, lt_tile(byte, fill(16i32))), smoke, fire_out);
        let v: I = select(is(k, 6i32), fire_out, x);
        // Embers throw flames into empty cells.
        let v: I = select(
            and(is(k, 0i32), and(near_ember, lt_tile(byte, fill(48i32)))),
            fire,
            v,
        );
        // Oil and wood catch.
        let catches: Mask = and(heat, lt_tile(byte, fill(128i32)));
        let v: I = select(and(is(k, 3i32), catches), fire, v);
        let catches: Mask = and(heat, lt_tile(byte, fill(40i32)));
        let v: I = select(and(is(k, 4i32), catches), cell(8i32, shade), v);
        // Embers: doused to wood, or burn out to smoke.
        let ember_out: I = select(lt_tile(byte, fill(6i32)), smoke, x);
        let ember_out: I = select(near_water, cell(4i32, shade), ember_out);
        let v: I = select(is(k, 8i32), ember_out, v);
        // Smoke thins out.
        select(and(is(k, 7i32), lt_tile(byte, fill(4i32))), empty, v)
    }

    fn sinks(top: I, bottom: I) -> Mask {
        let (kt, kb) = (kind(top), kind(bottom));
        and(
            and(not_solid(kt), not_solid(kb)),
            gt_tile(density(kt), density(kb)),
        )
    }

    fn slides_down(x: I, into: I) -> Mask {
        let (kx, ki) = (kind(x), kind(into));
        and(
            and(falls(kx), not_solid(ki)),
            lt_tile(density(ki), density(kx)),
        )
    }

    fn slides_up(x: I, into: I) -> Mask {
        let (kx, ki) = (kind(x), kind(into));
        and(
            and(rises(kx), not_solid(ki)),
            gt_tile(density(ki), density(kx)),
        )
    }

    /// Sideways movement for the pair `x` (left), `y` (right): liquids keep
    /// their `DIR` direction while they can and turn around when blocked;
    /// gases wander at random. Cells that just fell stay put.
    fn flow(x: I, y: I, x_fell: Mask, y_fell: Mask, bit: Mask) -> (I, I) {
        let (kx, ky) = (kind(x), kind(y));
        let dir: I = fill(4096i32); // DIR
        let open: Mask = and(and(spreads(kx), spreads(ky)), ne_tile(kx, ky));
        let x_right: Mask = and(
            and(liquid(kx), ne_tile(andi(x, dir), fill(0i32))),
            not(x_fell),
        );
        let y_left: Mask = and(
            and(liquid(ky), eq_tile(andi(y, dir), fill(0i32))),
            not(y_fell),
        );
        let gas_pair: Mask = and(not(liquid(kx)), not(liquid(ky)));
        let go: Mask = and(open, or(or(x_right, y_left), and(gas_pair, bit)));
        let x2: I = select(and(x_right, not(go)), xori(x, dir), x);
        let y2: I = select(and(y_left, not(go)), xori(y, dir), y);
        (select(go, y, x2), select(go, x, y2))
    }

    fn swap(m: Mask, x: I, y: I) -> (I, I) {
        (select(m, y, x), select(m, x, y))
    }

    fn bit(r: I, n: i32) -> Mask {
        eq_tile(andi(shri(r, fill(n)), fill(1i32)), fill(1i32))
    }

    /// One pass on a block `[a, b]` over `[c, d]`, in cpu.rs order.
    fn update(a: I, b: I, c: I, d: I, r: I) -> (I, I, I, I) {
        let mask: I = fill(255i32);
        let a: I = react(a, b, c, d, andi(r, mask));
        let b: I = react(b, a, c, d, andi(shri(r, fill(8i32)), mask));
        let c: I = react(c, a, b, d, andi(shri(r, fill(16i32)), mask));
        let d: I = react(d, a, b, c, andi(shri(r, fill(24i32)), mask));
        let r2: I = hash(r);

        // Gravity. The masks remember who moved vertically this pass.
        let m_ac: Mask = sinks(a, c);
        let (a, c) = swap(m_ac, a, c);
        let m_bd: Mask = sinks(b, d);
        let (b, d) = swap(m_bd, b, d);
        // Diagonals, in a random order so piles don't lean.
        let left_first: Mask = not(bit(r2, 0i32));
        let right_first: Mask = not(left_first);
        let m1: Mask = and(left_first, slides_down(a, d));
        let (a, d) = swap(m1, a, d);
        let m2: Mask = and(right_first, slides_down(b, c));
        let (b, c) = swap(m2, b, c);
        let m3: Mask = and(left_first, slides_down(b, c));
        let (b, c) = swap(m3, b, c);
        let m4: Mask = and(right_first, slides_down(a, d));
        let (a, d) = swap(m4, a, d);
        let m5: Mask = and(left_first, slides_up(c, b));
        let (c, b) = swap(m5, c, b);
        let m6: Mask = and(right_first, slides_up(d, a));
        let (d, a) = swap(m6, d, a);
        let m7: Mask = and(left_first, slides_up(d, a));
        let (d, a) = swap(m7, d, a);
        let m8: Mask = and(right_first, slides_up(c, b));
        let (c, b) = swap(m8, c, b);
        let m_ad: Mask = or(or(m1, m4), or(m6, m7));
        let m_bc: Mask = or(or(m2, m3), or(m5, m8));
        let fell_a: Mask = or(m_ac, m_ad);
        let fell_b: Mask = or(m_bd, m_bc);
        let fell_c: Mask = or(m_ac, m_bc);
        let fell_d: Mask = or(m_bd, m_ad);
        // Sideways.
        let (a, b) = flow(a, b, fell_a, fell_b, bit(r2, 1i32));
        let (c, d) = flow(c, d, fell_c, fell_d, bit(r2, 2i32));
        (a, b, c, d)
    }

    /// One Margolus pass. `pass_buf[0]` is the pass number: its low bit is
    /// the partition offset, and with the frame in `params[0]` it salts the
    /// randomness. It comes in a tensor rather than as an `i32` argument
    /// because integer arguments specialize the kernel by divisibility, and
    /// pass numbers 0, 1, 2, 3 would mean three compiles of this (large)
    /// kernel at every start.
    ///
    /// The nine views are the 3x3 neighborhood as in `life`: `mm` is the
    /// buffer itself, `u*`/`d*` are shifted a row, `*l`/`*r` a column. The
    /// ghost ring keeps whatever it holds.
    #[cutile::entry()]
    pub fn step(
        out: &mut Tensor<i32, { [32, 32] }>,
        ul: &Tensor<i32, { [-1, -1] }>,
        um: &Tensor<i32, { [-1, -1] }>,
        ur: &Tensor<i32, { [-1, -1] }>,
        ml: &Tensor<i32, { [-1, -1] }>,
        mm: &Tensor<i32, { [-1, -1] }>,
        mr: &Tensor<i32, { [-1, -1] }>,
        dl: &Tensor<i32, { [-1, -1] }>,
        dm: &Tensor<i32, { [-1, -1] }>,
        dr: &Tensor<i32, { [-1, -1] }>,
        params: &Tensor<i32, { [-1] }>,
        pass_buf: &Tensor<i32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let pass: I = pass_index(pass_buf);
        let (i, j) = (pid.0, pid.1);
        let shape = mm.shape();
        let last_i: i32 = shape[0] / 32i32 - 1i32;
        let last_j: i32 = shape[1] / 32i32 - 1i32;
        // Ghost tiles copy themselves. Their "up"/"left" block may not
        // exist, so clamp; the loaded values are never stored.
        let up: i32 = max(i - 1i32, 0i32);
        let left: i32 = max(j - 1i32, 0i32);

        let t_ul: I = ul.partition(const_shape![32, 32]).load([up, left]);
        let t_um: I = um.partition(const_shape![32, 32]).load([up, j]);
        let t_ur: I = ur.partition(const_shape![32, 32]).load([up, j]);
        let t_ml: I = ml.partition(const_shape![32, 32]).load([i, left]);
        let t_mm: I = mm.partition(const_shape![32, 32]).load([i, j]);
        let t_mr: I = mr.partition(const_shape![32, 32]).load([i, j]);
        let t_dl: I = dl.partition(const_shape![32, 32]).load([i, left]);
        let t_dm: I = dm.partition(const_shape![32, 32]).load([i, j]);
        let t_dr: I = dr.partition(const_shape![32, 32]).load([i, j]);

        // Which corner of its block is this cell?
        let (row, col) = coords();
        let parity: I = andi(pass, fill(1i32));
        let pr: I = andi(row + parity, fill(1i32));
        let pc: I = andi(col + parity, fill(1i32));
        let top: Mask = eq_tile(pr, fill(0i32));
        let lft: Mask = eq_tile(pc, fill(0i32));
        // Rows of the block: for a top cell they are (this row, below);
        // for a bottom cell (above, this row). Same for columns.
        let a_l: I = select(top, t_ml, t_ul);
        let a_m: I = select(top, t_mm, t_um);
        let a_r: I = select(top, t_mr, t_ur);
        let c_l: I = select(top, t_dl, t_ml);
        let c_m: I = select(top, t_dm, t_mm);
        let c_r: I = select(top, t_dr, t_mr);
        let a: I = select(lft, a_m, a_l);
        let b: I = select(lft, a_r, a_m);
        let c: I = select(lft, c_m, c_l);
        let d: I = select(lft, c_r, c_m);

        let bx: I = shri(col - pc, fill(1i32));
        let by: I = shri(row - pr, fill(1i32));
        let salt: I = param(params, 0i32) * fill(64i32) + pass;
        let r: I = hash(xori(bx, hash(xori(by, hash(salt)))));

        let (na, nb, nc, nd) = update(a, b, c, d, r);
        let mine: I = select(top, select(lft, na, nb), select(lft, nc, nd));
        let ghost: bool = i == 0i32 || j == 0i32 || i == last_i || j == last_j;
        out.store(if ghost { t_mm } else { mine });
    }

    /// `hash(row * 4096 + col)` per cell, to check the hash against the CPU.
    #[cutile::entry()]
    pub fn hash_grid(out: &mut Tensor<i32, { [32, 32] }>) {
        let (row, col) = coords();
        out.store(hash(row * fill(4096i32) + col));
    }

    /// Replace cells where `paint` has the `PAINT` flag.
    #[cutile::entry()]
    pub fn paint(
        out: &mut Tensor<i32, { [32, 32] }>,
        cells: &Tensor<i32, { [-1, -1] }>,
        paint: &Tensor<i32, { [-1, -1] }>,
    ) {
        let c: I = cells.load_like(out);
        let p: I = paint.load_like(out);
        let flag: I = fill(1048576i32); // PAINT
        let painted: Mask = ne_tile(andi(p, flag), fill(0i32));
        out.store(select(painted, andi(p, fill(0x1fffi32)), c));
    }

    fn color(k: I, rgbs: (I, I, I, I), kk: i32, r: i32, g: i32, b: i32, s: i32) -> (I, I, I, I) {
        let m: Mask = is(k, kk);
        (
            select(m, fill(r), rgbs.0),
            select(m, fill(g), rgbs.1),
            select(m, fill(b), rgbs.2),
            select(m, fill(s), rgbs.3),
        )
    }

    /// Colors. `params[0]` is the frame, for the fire flicker.
    #[cutile::entry()]
    pub fn render(
        out: &mut Tensor<i32, { [32, 32] }>,
        cells: &Tensor<i32, { [-1, -1] }>,
        params: &Tensor<i32, { [-1] }>,
    ) {
        let v: I = cells.load_like(out);
        let k: I = kind(v);
        let shade: I = andi(shri(v, fill(8i32)), fill(15i32)) - fill(8i32);
        let (row, col) = coords();
        let flicker: I = andi(
            hash(xori(
                row * fill(977i32) + col,
                param(params, 0i32) * fill(7i32),
            )),
            fill(63i32),
        );
        // Base color per element, and how much the shade bits tint it.
        let rgbs: (I, I, I, I) = (fill(14i32), fill(14i32), fill(20i32), fill(0i32));
        let rgbs = color(k, rgbs, 1i32, 194i32, 168i32, 98i32, 5i32); // sand
        let rgbs = color(k, rgbs, 2i32, 40i32, 90i32, 205i32, 2i32); // water
        let rgbs = color(k, rgbs, 3i32, 80i32, 55i32, 25i32, 2i32); // oil
        let rgbs = color(k, rgbs, 4i32, 112i32, 72i32, 40i32, 4i32); // wood
        let rgbs = color(k, rgbs, 5i32, 82i32, 82i32, 92i32, 2i32); // wall
        let rgbs = color(k, rgbs, 6i32, 255i32, 120i32, 30i32, 0i32); // fire
        let rgbs = color(k, rgbs, 7i32, 105i32, 105i32, 112i32, 3i32); // smoke
        let rgbs = color(k, rgbs, 8i32, 235i32, 80i32, 20i32, 0i32); // ember
        let (r, g, b, s) = rgbs;
        let r: I = r + shade * s;
        let g: I = g + shade * s;
        let b: I = b + shade * s;
        // Fire flickers between orange and yellow, embers glow.
        let g: I = select(is(k, 6i32), g + flicker * fill(2i32), g);
        let g: I = select(is(k, 8i32), g + flicker, g);
        let lo: I = fill(0i32);
        let hi: I = fill(255i32);
        let r: I = min_tile(max_tile(r, lo), hi);
        let g: I = min_tile(max_tile(g, lo), hi);
        let b: I = min_tile(max_tile(b, lo), hi);
        out.store(r * fill(65536i32) + g * fill(256i32) + b);
    }
}

/// The nine views of `src` for a 3x3 stencil, in `step` order. Offset -1
/// is "view shifted by 31, loaded at block - 1"; offset +1 is "view shifted
/// by 1, loaded at the same block" (see `life`).
fn views(src: &Tensor<i32>) -> Result<[TensorView<'_, i32>; 9], Error> {
    let (rows, cols) = (src.shape()[0] as usize, src.shape()[1] as usize);
    let shifts = [TILE - 1, 0, 1];
    let v = |r: usize, c: usize| src.slice(&[shifts[r]..rows, shifts[c]..cols]);
    Ok([
        v(0, 0)?,
        v(0, 1)?,
        v(0, 2)?,
        v(1, 0)?,
        v(1, 1)?,
        v(1, 2)?,
        v(2, 0)?,
        v(2, 1)?,
        v(2, 2)?,
    ])
}

/// Device buffers: a ping-pong pair of worlds, the paint layer, and the
/// rendered frame.
pub struct Pipeline {
    pub layout: Layout,
    /// Margolus passes per frame; even, so a frame's graph ends in the
    /// buffer it started from.
    pub passes: usize,
    stream: Arc<Stream>,
    cells: [Tensor<i32>; 2],
    paint: Tensor<i32>,
    params: Tensor<i32>,
    /// Pass numbers, one every 4 elements (16 bytes) so every pass's view
    /// has the same pointer alignment and shape, hence the same kernel.
    pass_table: Tensor<i32>,
    frame: Tensor<i32>,
    paint_host: Pinned<i32>,
    params_host: Pinned<i32>,
    frame_host: Pinned<i32>,
    pixels: Vec<u32>,
}

impl Pipeline {
    pub fn new(stream: &Arc<Stream>, layout: Layout, passes: usize) -> Result<Self, Error> {
        assert!(
            passes >= 2 && passes.is_multiple_of(2),
            "passes must be even"
        );
        let shape = [layout.buf_rows(), layout.buf_cols()];
        let n = shape[0] * shape[1];
        Ok(Pipeline {
            layout,
            passes,
            stream: stream.clone(),
            cells: [
                api::zeros::<i32>(&shape).sync_on(stream)?,
                api::zeros::<i32>(&shape).sync_on(stream)?,
            ],
            paint: api::zeros::<i32>(&shape).sync_on(stream)?,
            params: api::zeros::<i32>(&[PARAMS]).sync_on(stream)?,
            pass_table: {
                let table: Vec<i32> = (0..passes as i32).flat_map(|p| [p, 0, 0, 0]).collect();
                api::copy_host_vec_to_device(&Arc::new(table)).sync_on(stream)?
            },
            frame: api::zeros::<i32>(&shape).sync_on(stream)?,
            paint_host: Pinned::new(stream, n)?,
            params_host: Pinned::new(stream, PARAMS)?,
            frame_host: Pinned::new(stream, n)?,
            pixels: vec![0; layout.rows * layout.cols],
        })
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Replace the world in buffer `parity` (the one the next frame of
    /// that parity reads).
    pub fn upload_world(&mut self, parity: usize, world: &[i32]) -> Result<(), Error> {
        self.paint_host.as_mut_slice().copy_from_slice(world);
        self.paint_host
            .upload(&mut self.cells[parity], &self.stream)
    }

    /// The paint layer to upload next; a frame's brush strokes go here.
    pub fn paint_mut(&mut self) -> &mut [i32] {
        self.paint_host.as_mut_slice()
    }

    pub fn upload_paint(&mut self) -> Result<(), Error> {
        self.paint_host.upload(&mut self.paint, &self.stream)
    }

    pub fn set_frame(&mut self, frame: i32) -> Result<(), Error> {
        self.params_host.as_mut_slice()[0] = frame;
        self.params_host.upload(&mut self.params, &self.stream)
    }

    /// One frame: apply paint, run the passes, render. Frame parity p reads
    /// `cells[p]`; the paint step writes `cells[1 - p]`, and the even number
    /// of passes brings the result back to `cells[1 - p]`... which the next
    /// frame (parity 1 - p) reads. So the two parities alternate buffers.
    pub fn run(&mut self, sub: &impl Submit, parity: usize) -> Result<(), Error> {
        let [c0, c1] = &mut self.cells;
        let (mut src, mut dst) = if parity == 0 { (c0, c1) } else { (c1, c0) };
        sub.submit(kernels::paint(
            dst.partition([TILE, TILE]),
            &*src,
            &self.paint,
        ))?;
        std::mem::swap(&mut src, &mut dst);
        for pass in 0..self.passes {
            {
                let v = views(src)?;
                let range = 4 * pass..4 * pass + 4;
                let pass_view = self.pass_table.slice(std::slice::from_ref(&range))?;
                sub.submit(kernels::step(
                    dst.partition([TILE, TILE]),
                    &v[0],
                    &v[1],
                    &v[2],
                    &v[3],
                    &v[4],
                    &v[5],
                    &v[6],
                    &v[7],
                    &v[8],
                    &self.params,
                    &pass_view,
                ))?;
            }
            std::mem::swap(&mut src, &mut dst);
        }
        sub.submit(kernels::render(
            (&mut self.frame).partition([TILE, TILE]),
            &*src,
            &self.params,
        ))?;
        Ok(())
    }

    pub fn run_eager(&mut self, parity: usize) -> Result<(), Error> {
        let stream = self.stream.clone();
        self.run(&Eager(&stream), parity)
    }

    /// One graph per parity.
    pub fn capture(&mut self) -> Result<[CudaGraph<()>; 2], Error> {
        let stream = self.stream.clone();
        let even = CudaGraph::scope(&stream, |s| Ok(self.run(s, 0)?))?;
        let odd = CudaGraph::scope(&stream, |s| Ok(self.run(s, 1)?))?;
        Ok([even, odd])
    }

    /// The kernel's hash of `row * 4096 + col` for every buffer cell.
    pub fn hash_grid(&mut self) -> Result<&[i32], Error> {
        let stream = self.stream.clone();
        kernels::hash_grid((&mut self.frame).partition([TILE, TILE])).sync_on(&stream)?;
        self.frame_host.download(&self.frame, &self.stream)?;
        Ok(self.frame_host.as_slice())
    }

    /// The world after a frame of parity `parity`.
    pub fn download_world(&mut self, parity: usize) -> Result<&[i32], Error> {
        self.frame_host
            .download(&self.cells[1 - parity], &self.stream)?;
        Ok(self.frame_host.as_slice())
    }

    /// The rendered frame without the ghost ring, as 0x00RRGGBB.
    pub fn download_frame(&mut self) -> Result<&mut [u32], Error> {
        self.frame_host.download(&self.frame, &self.stream)?;
        let buf = tilekit::as_u32(self.frame_host.as_slice());
        let (bc, lay) = (self.layout.buf_cols(), self.layout);
        for (r, line) in self.pixels.chunks_exact_mut(lay.cols).enumerate() {
            let start = (TILE + r) * bc + TILE;
            line.copy_from_slice(&buf[start..start + lay.cols]);
        }
        Ok(&mut self.pixels)
    }
}
