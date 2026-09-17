//! GPU pipeline: jump flooding to a distance field, then 2D global
//! illumination by marching rays through it.

use std::sync::Arc;

use cuda_core::Stream;
use cutile::prelude::*;
use tilekit::{Eager, Pinned, Submit};

use crate::Layout;

pub const TILE: usize = 32;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    // Literal 32x32 shapes, as in raymarch: no generic, one type spelling.
    type I = Tile<i32, { [32, 32] }>;
    type Mask = Tile<bool, { [32, 32] }>;

    fn fill_i(v: i32) -> I {
        broadcast_scalar(v, const_shape![32, 32])
    }

    /// Buffer row and column of every pixel in this tile.
    fn pixel_coords() -> (I, I) {
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

    /// A seed is the position of the nearest occupied pixel, packed as
    /// `(row + 8192) * 32768 + (col + 8192)`.
    ///
    /// Zero decodes to (-8192, -8192), farther from every pixel than any
    /// real seed can be. So 0 means "no seed" without a special case, and
    /// the zero padding cuTile uses for partial tiles reads as "no seed" too.
    fn encode(row: I, col: I) -> I {
        let bias: I = fill_i(8192i32);
        (row + bias) * fill_i(32768i32) + col + bias
    }

    fn decode(seed: I) -> (I, I) {
        let bias: I = fill_i(8192i32);
        (
            shri(seed, fill_i(15i32)) - bias,
            andi(seed, fill_i(32767i32)) - bias,
        )
    }

    /// Squared distance from each pixel to its seed.
    fn dist2(seed: I, row: I, col: I) -> I {
        let (sr, sc) = decode(seed);
        let dr: I = sr - row;
        let dc: I = sc - col;
        dr * dr + dc * dc
    }

    /// Keep `candidate` where it is strictly closer than `best`.
    fn closer(best: I, best_d: I, candidate: I, row: I, col: I) -> (I, I) {
        let d: I = dist2(candidate, row, col);
        let wins: Mask = lt_tile(d, best_d);
        (select(wins, candidate, best), select(wins, d, best_d))
    }

    /// Every occupied scene pixel is its own seed.
    #[cutile::entry()]
    pub fn seed_init(out: &mut Tensor<i32, { [32, 32] }>, scene: &Tensor<i32, { [-1, -1] }>) {
        let s: I = scene.load_like(out);
        let (row, col) = pixel_coords();
        let occupied: Mask = ne_tile(s, fill_i(0i32));
        out.store(select(occupied, encode(row, col), fill_i(0i32)));
    }

    /// One jump flooding pass at offset k: every pixel looks at the seeds of
    /// the 9 pixels at (-k, 0, +k) in each axis and keeps the closest.
    ///
    /// Offsets use Life's split, d = q * 32 + s: a host view shifted by s,
    /// loaded at block index + q. For k < 32 that is (32 - k, q = -1) behind
    /// and (k, q = 0) ahead; for k >= 32 it is s = 0 and q = -k/32 or +k/32.
    /// `back` and `ahead` are those q values. Views are named by row
    /// (u/m/d) and column (l/m/r); `mm` is the unshifted buffer.
    ///
    /// Blocks outside the buffer read as "no seed". A leading ring of ghost
    /// tiles (row 0 and column 0) makes block i - 1 exist for the first real
    /// tile; ghost tiles always store "no seed".
    #[cutile::entry()]
    pub fn jfa_step(
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
        back: i32,
        ahead: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let shape = mm.shape();
        let tile_rows: i32 = shape[0] / 32i32;
        let tile_cols: i32 = shape[1] / 32i32;
        let zero: I = fill_i(0i32);

        let up: i32 = pid.0 + back;
        let down: i32 = pid.0 + ahead;
        let left: i32 = pid.1 + back;
        let right: i32 = pid.1 + ahead;
        let up_ok: bool = up >= 0i32;
        let down_ok: bool = down < tile_rows;
        let left_ok: bool = left >= 0i32;
        let right_ok: bool = right < tile_cols;
        // Clamped so every load is in range; the *_ok flags decide whether
        // the loaded tile is used.
        let up: i32 = max(up, 0i32);
        let down: i32 = min(down, tile_rows - 1i32);
        let left: i32 = max(left, 0i32);
        let right: i32 = min(right, tile_cols - 1i32);
        let (i, j) = (pid.0, pid.1);

        let t_ul: I = ul.partition(const_shape![32, 32]).load([up, left]);
        let t_um: I = um.partition(const_shape![32, 32]).load([up, j]);
        let t_ur: I = ur.partition(const_shape![32, 32]).load([up, right]);
        let t_ml: I = ml.partition(const_shape![32, 32]).load([i, left]);
        let t_mm: I = mm.partition(const_shape![32, 32]).load([i, j]);
        let t_mr: I = mr.partition(const_shape![32, 32]).load([i, right]);
        let t_dl: I = dl.partition(const_shape![32, 32]).load([down, left]);
        let t_dm: I = dm.partition(const_shape![32, 32]).load([down, j]);
        let t_dr: I = dr.partition(const_shape![32, 32]).load([down, right]);
        let t_ul: I = if up_ok && left_ok { t_ul } else { zero };
        let t_um: I = if up_ok { t_um } else { zero };
        let t_ur: I = if up_ok && right_ok { t_ur } else { zero };
        let t_ml: I = if left_ok { t_ml } else { zero };
        let t_mr: I = if right_ok { t_mr } else { zero };
        let t_dl: I = if down_ok && left_ok { t_dl } else { zero };
        let t_dm: I = if down_ok { t_dm } else { zero };
        let t_dr: I = if down_ok && right_ok { t_dr } else { zero };

        let (row, col) = pixel_coords();
        let best: I = t_mm;
        let best_d: I = dist2(best, row, col);
        let (best, best_d) = closer(best, best_d, t_ul, row, col);
        let (best, best_d) = closer(best, best_d, t_um, row, col);
        let (best, best_d) = closer(best, best_d, t_ur, row, col);
        let (best, best_d) = closer(best, best_d, t_ml, row, col);
        let (best, best_d) = closer(best, best_d, t_mr, row, col);
        let (best, best_d) = closer(best, best_d, t_dl, row, col);
        let (best, best_d) = closer(best, best_d, t_dm, row, col);
        let (best, _best_d) = closer(best, best_d, t_dr, row, col);
        let ghost: bool = i == 0i32 || j == 0i32;
        out.store(if ghost { zero } else { best });
    }

    /// Distance in pixels from each pixel to its seed.
    #[cutile::entry()]
    pub fn distance(out: &mut Tensor<f32, { [32, 32] }>, seeds: &Tensor<i32, { [-1, -1] }>) {
        let s: I = seeds.load_like(out);
        let (row, col) = pixel_coords();
        let d2: Tile<f32, { [32, 32] }> = convert_tile(dist2(s, row, col));
        out.store(sqrt(d2, rounding::NearestEven, ftz::Disabled));
    }

    /// Read `tensor[idx]` for a whole tile of flat indices.
    ///
    /// # Safety
    /// Every index must be inside `tensor`, which must be contiguous.
    unsafe fn gather_i32(tensor: &Tensor<i32, { [-1, -1] }>, idx: I) -> I {
        let base: PointerTile<*const i32, { [] }> = pointer_to_tile(tensor.as_ptr());
        let base: PointerTile<*const i32, { [1, 1] }> = base.reshape(const_shape![1, 1]);
        let ptrs: PointerTile<*const i32, { [32, 32] }> = base.broadcast(const_shape![32, 32]);
        let ptrs: PointerTile<*const i32, { [32, 32] }> = ptrs.offset_tile(idx);
        let (values, _token): (I, Token) = load_ptr_tko(
            ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        values
    }

    /// The scene value (type and color) of each pixel's nearest occupied
    /// pixel: a gather through the seed coordinates. 0 where there are no
    /// seeds at all.
    #[cutile::entry()]
    pub fn nearest(
        out: &mut Tensor<i32, { [32, 32] }>,
        seeds: &Tensor<i32, { [-1, -1] }>,
        scene: &Tensor<i32, { [-1, -1] }>,
    ) {
        let s: I = seeds.load_like(out);
        let shape = scene.shape();
        let cols: i32 = shape[1];
        let zero: I = fill_i(0i32);
        let (sr, sc) = decode(s);
        // "No seed" decodes off the grid; clamp it in, then mask it out.
        let sr: I = min_tile(max_tile(sr, zero), fill_i(shape[0] - 1i32));
        let sc: I = min_tile(max_tile(sc, zero), fill_i(cols - 1i32));
        let v: I = unsafe { gather_i32(scene, sr * fill_i(cols) + sc) };
        out.store(select(eq_tile(s, zero), zero, v));
    }

    // ---- Lighting ---------------------------------------------------------

    type F = Tile<f32, { [32, 32] }>;
    type Rgba = Tile<f32, { [32, 32, 4] }>;

    fn fill(v: f32) -> F {
        broadcast_scalar(v, const_shape![32, 32])
    }

    fn param(p: Tile<f32, { [16] }>, k: i32) -> F {
        let idx: Tile<i32, { [] }> = scalar_to_tile(k);
        let v: Tile<f32, { [1] }> = extract(p, [idx]);
        let v: Tile<f32, { [1, 1] }> = v.reshape(const_shape![1, 1]);
        v.broadcast(const_shape![32, 32])
    }

    fn params(t: &Tensor<f32, { [-1] }>) -> Tile<f32, { [16] }> {
        t.partition(const_shape![16]).load([0i32])
    }

    fn fract(x: F) -> F {
        x - floor(x)
    }

    fn clamp01(x: F) -> F {
        min_tile(max_tile(x, fill(0.0f32)), fill(1.0f32))
    }

    fn channel(rgba: Rgba, c: i32) -> F {
        let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let idx: Tile<i32, { [] }> = scalar_to_tile(c);
        let v: Tile<f32, { [32, 32, 1] }> = extract(rgba, [zero, zero, idx]);
        v.reshape(const_shape![32, 32])
    }

    fn rgba(r: F, g: F, b: F, a: F) -> Rgba {
        let r: Tile<f32, { [32, 32, 1] }> = r.reshape(const_shape![32, 32, 1]);
        let g: Tile<f32, { [32, 32, 1] }> = g.reshape(const_shape![32, 32, 1]);
        let b: Tile<f32, { [32, 32, 1] }> = b.reshape(const_shape![32, 32, 1]);
        let a: Tile<f32, { [32, 32, 1] }> = a.reshape(const_shape![32, 32, 1]);
        let rg: Tile<f32, { [32, 32, 2] }> = cat(r, g, 2i32);
        let ba: Tile<f32, { [32, 32, 2] }> = cat(b, a, 2i32);
        cat(rg, ba, 2i32)
    }

    /// Channel `shift` (16 red, 8 green, 0 blue) of a packed pixel, 0..1.
    fn color_of(v: I, shift: i32) -> F {
        let byte: I = andi(shri(v, fill_i(shift)), fill_i(255i32));
        let byte: F = convert_tile(byte);
        byte * fill(1.0f32 / 255.0f32)
    }

    /// Read `tensor[idx]` for a whole tile of flat indices.
    ///
    /// # Safety
    /// Every index must be inside `tensor`, which must be contiguous.
    unsafe fn gather_f32(tensor: &Tensor<f32, { [-1, -1] }>, idx: I) -> F {
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(tensor.as_ptr());
        let base: PointerTile<*const f32, { [1, 1] }> = base.reshape(const_shape![1, 1]);
        let ptrs: PointerTile<*const f32, { [32, 32] }> = base.broadcast(const_shape![32, 32]);
        let ptrs: PointerTile<*const f32, { [32, 32] }> = ptrs.offset_tile(idx);
        let (values, _token): (F, Token) = load_ptr_tko(
            ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        values
    }

    /// Pixel centers as floats.
    fn pixel_centers() -> (F, F) {
        let (row, col) = pixel_coords();
        let row: F = convert_tile(row);
        let col: F = convert_tile(col);
        (row + fill(0.5f32), col + fill(0.5f32))
    }

    /// 2D global illumination, blended into a running average.
    ///
    /// Every pixel casts `rays` rays at evenly spaced angles, rotated by a
    /// per-pixel, per-frame jitter. Each ray sphere-traces the distance
    /// field: sample the distance at its current pixel, step that far (at
    /// least one pixel), stop on landing in an occupied pixel or leaving
    /// the world. A ray that lands in a light adds the light's color.
    ///
    /// Sampling the distance at a computed position is a gather, not a
    /// stencil: no fixed offset view can express it. This uses cuTile's
    /// unsafe pointer loads, with indices clamped into the buffer.
    ///
    /// `params` (16 slots): 0 frame, 1 blend weight of the new frame,
    /// 2 light intensity, 3 view, 4 rays as a float, 5 exposure, 6 ambient.
    #[cutile::entry()]
    pub fn radiance(
        out: &mut Tensor<f32, { [32, 32, 4] }>,
        prev: &Tensor<f32, { [-1, -1, 4] }>,
        dist: &Tensor<f32, { [-1, -1] }>,
        scene: &Tensor<i32, { [-1, -1] }>,
        param_buf: &Tensor<f32, { [-1] }>,
        rays: i32,
        steps: i32,
        check_every: i32,
    ) {
        let p: Tile<f32, { [16] }> = params(param_buf);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let shape = dist.shape();
        let rows: i32 = shape[0];
        let cols: i32 = shape[1];
        let zero: F = fill(0.0f32);
        let one: F = fill(1.0f32);
        let (y, x) = pixel_centers();

        // Interleaved gradient noise, scrolled by the golden ratio each frame.
        let noise: F = fract(x * fill(0.06711056f32) + y * fill(0.00583715f32));
        let noise: F = fract(noise * fill(52.982918f32));
        let jitter: F = fract(noise + param(p, 0i32) * fill(0.618034f32));
        let sector: F = fill(6.2831855f32) / param(p, 4i32);

        let x_max: F = convert_tile(fill_i(cols));
        let y_max: F = convert_tile(fill_i(rows));
        let last_col: I = fill_i(cols - 1i32);
        let last_row: I = fill_i(rows - 1i32);
        let izero: I = fill_i(0i32);

        let mut sum_r: F = zero;
        let mut sum_g: F = zero;
        let mut sum_b: F = zero;
        let mut ray: f32 = 0.0f32;
        for _r in 0i32..rays {
            let angle: F = (fill(ray) + jitter) * sector;
            ray = ray + 1.0f32;
            let dx: F = cos(angle);
            let dy: F = sin(angle);

            let mut t: F = zero;
            let mut active: Mask = lt_tile(zero, one);
            let mut hit: Mask = lt_tile(one, zero);
            let mut idx: I = izero;
            let mut done: i32 = 0i32;
            while done < steps {
                let chunk: i32 = min(check_every, steps - done);
                for _s in 0i32..chunk {
                    let px: F = x + dx * t;
                    let py: F = y + dy * t;
                    let in_x: Mask =
                        select(ge_tile(px, zero), lt_tile(px, x_max), ge_tile(px, zero));
                    let in_y: Mask =
                        select(ge_tile(py, zero), lt_tile(py, y_max), ge_tile(py, zero));
                    let inside: Mask = select(in_x, in_y, in_x);
                    let col: I = convert_tile(floor(px));
                    let row: I = convert_tile(floor(py));
                    let col: I = min_tile(max_tile(col, izero), last_col);
                    let row: I = min_tile(max_tile(row, izero), last_row);
                    let here: I = row * fill_i(cols) + col;
                    let d: F = unsafe { gather_f32(dist, here) };
                    let landed: Mask = lt_tile(d, fill(0.5f32));
                    // Hit: still marching, inside the world, on a surface.
                    let hit_now: Mask = select(active, select(inside, landed, inside), active);
                    hit = select(hit_now, hit_now, hit);
                    idx = select(hit_now, here, idx);
                    // Keep marching: inside && !landed.
                    let keep: Mask = select(landed, lt_tile(one, zero), inside);
                    active = select(active, keep, active);
                    t = select(active, t + max_tile(d - one, one), t);
                }
                done = done + chunk;
                let alive: F = select(active, one, zero);
                let alive: Tile<f32, { [32] }> = reduce_max(alive, 1i32);
                let alive: Tile<f32, { [] }> = reduce_max(alive, 0i32);
                let alive: f32 = tile_to_scalar(alive);
                if alive < 0.5f32 {
                    break;
                }
            }

            // What did it land on?
            let v: I = unsafe { gather_i32(scene, idx) };
            let is_light: Mask = eq_tile(shri(v, fill_i(24i32)), fill_i(2i32));
            let lit: Mask = select(hit, is_light, hit);
            sum_r = sum_r + select(lit, color_of(v, 16i32), zero);
            sum_g = sum_g + select(lit, color_of(v, 8i32), zero);
            sum_b = sum_b + select(lit, color_of(v, 0i32), zero);
            // Rays that left the world see a faint ambient sky.
            let escaped: F = select(hit, zero, param(p, 6i32));
            sum_r = sum_r + escaped * fill(0.55f32);
            sum_g = sum_g + escaped * fill(0.65f32);
            sum_b = sum_b + escaped;
        }

        let scale: F = param(p, 2i32) / param(p, 4i32);
        let prev_tile: Rgba = prev
            .partition(const_shape![32, 32, 4])
            .load([pid.0, pid.1, 0i32]);
        let blend: F = param(p, 1i32);
        let r: F = channel(prev_tile, 0i32);
        let g: F = channel(prev_tile, 1i32);
        let b: F = channel(prev_tile, 2i32);
        let r: F = r + (sum_r * scale - r) * blend;
        let g: F = g + (sum_g * scale - g) * blend;
        let b: F = b + (sum_b * scale - b) * blend;
        out.store(rgba(r, g, b, one));
    }

    fn to_byte(c: F) -> I {
        // A bare `convert_tile(..)` tail expression has no type for the JIT
        // to read; the annotated `let` supplies it.
        let byte: I = convert_tile(clamp01(c) * fill(255.0f32) + fill(0.5f32));
        byte
    }

    /// Final 0x00RRGGBB image for the chosen view.
    ///
    /// View 0: floor albedo times radiance, walls and lights drawn on top.
    /// 1: radiance alone. 2: the distance field as contour bands.
    /// 3: nearest surface color (a Voronoi diagram of the scene pixels).
    #[cutile::entry()]
    pub fn compose(
        out: &mut Tensor<i32, { [32, 32] }>,
        light: &Tensor<f32, { [-1, -1, 4] }>,
        scene: &Tensor<i32, { [-1, -1] }>,
        dist: &Tensor<f32, { [-1, -1] }>,
        nearest: &Tensor<i32, { [-1, -1] }>,
        param_buf: &Tensor<f32, { [-1] }>,
    ) {
        let p: Tile<f32, { [16] }> = params(param_buf);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let zero: F = fill(0.0f32);
        let one: F = fill(1.0f32);
        let l: Rgba = light
            .partition(const_shape![32, 32, 4])
            .load([pid.0, pid.1, 0i32]);
        let s: I = scene.load_like(out);
        let d: F = dist.load_like(out);
        let n: I = nearest.load_like(out);
        let (y, x) = pixel_centers();

        // Lit view. Subtle 16 px checkerboard floor.
        let check: F = floor(x * fill(0.0625f32)) + floor(y * fill(0.0625f32));
        let check: F = check - fill(2.0f32) * floor(check * fill(0.5f32));
        let albedo: F = fill(0.72f32) + check * fill(0.08f32);
        let exposure: F = param(p, 5i32);
        let tone_r: F = one - exp(zero - channel(l, 0i32) * exposure);
        let tone_g: F = one - exp(zero - channel(l, 1i32) * exposure);
        let tone_b: F = one - exp(zero - channel(l, 2i32) * exposure);
        let kind: I = shri(s, fill_i(24i32));
        let is_wall: Mask = eq_tile(kind, fill_i(1i32));
        let is_light: Mask = eq_tile(kind, fill_i(2i32));
        let (sr, sg, sb) = (color_of(s, 16i32), color_of(s, 8i32), color_of(s, 0i32));
        let wall: F = fill(0.35f32);
        let lit_r: F = select(is_light, sr, select(is_wall, sr * wall, tone_r * albedo));
        let lit_g: F = select(is_light, sg, select(is_wall, sg * wall, tone_g * albedo));
        let lit_b: F = select(is_light, sb, select(is_wall, sb * wall, tone_b * albedo));

        // Distance bands: a dark line every 8 px, fading with distance.
        let band: F = fract(d * fill(0.125f32));
        let line: F = select(lt_tile(band, fill(0.12f32)), fill(0.55f32), one);
        let fade: F = exp(zero - d * fill(0.02f32));
        let near_c: F = select(lt_tile(d, fill(0.5f32)), one, fade * line);
        let dist_r: F = near_c * fill(0.95f32);
        let dist_g: F = near_c * fill(0.75f32);
        let dist_b: F = near_c * fill(0.35f32) + (one - near_c) * fill(0.12f32);

        // Voronoi: the nearest surface's color, darker away from it.
        let shade: F = fill(0.35f32) + fill(0.65f32) * fade;
        let vor_r: F = color_of(n, 16i32) * shade;
        let vor_g: F = color_of(n, 8i32) * shade;
        let vor_b: F = color_of(n, 0i32) * shade;

        let view: F = param(p, 3i32);
        let v1: Mask = gt_tile(view, fill(0.5f32));
        let v2: Mask = gt_tile(view, fill(1.5f32));
        let v3: Mask = gt_tile(view, fill(2.5f32));
        let r: F = select(v3, vor_r, select(v2, dist_r, select(v1, tone_r, lit_r)));
        let g: F = select(v3, vor_g, select(v2, dist_g, select(v1, tone_g, lit_g)));
        let b: F = select(v3, vor_b, select(v2, dist_b, select(v1, tone_b, lit_b)));

        // Gamma 2.
        let r: F = sqrt(clamp01(r), rounding::NearestEven, ftz::Disabled);
        let g: F = sqrt(clamp01(g), rounding::NearestEven, ftz::Disabled);
        let b: F = sqrt(clamp01(b), rounding::NearestEven, ftz::Disabled);
        out.store(to_byte(r) * fill_i(65536i32) + to_byte(g) * fill_i(256i32) + to_byte(b));
    }
}

/// The nine views of `src` for a jump of `k` pixels, in `jfa_step` order,
/// plus the block offsets (back, ahead).
fn jump_views(src: &Tensor<i32>, k: usize) -> Result<([TensorView<'_, i32>; 9], i32, i32), Error> {
    let (rows, cols) = (src.shape()[0] as usize, src.shape()[1] as usize);
    let (behind, back, front, ahead) = if k < TILE {
        (TILE - k, -1, k, 0)
    } else {
        (0, -((k / TILE) as i32), 0, (k / TILE) as i32)
    };
    let shifts = [behind, 0, front];
    let v = |r: usize, c: usize| src.slice(&[shifts[r]..rows, shifts[c]..cols]);
    let views = [
        v(0, 0)?,
        v(0, 1)?,
        v(0, 2)?,
        v(1, 0)?,
        v(1, 1)?,
        v(1, 2)?,
        v(2, 0)?,
        v(2, 1)?,
        v(2, 2)?,
    ];
    Ok((views, back, ahead))
}

/// Ray budget per pixel. Loop bounds are kernel scalars, so changing these
/// means recapturing the graphs.
#[derive(Clone, Copy, Debug)]
pub struct Quality {
    pub rays: i32,
    pub steps: i32,
    /// Steps between "is any ray in this tile still marching?" checks.
    pub check_every: i32,
}

pub const PARAMS: usize = 16;

/// Device buffers, all `buf_rows x buf_cols` (the world plus a leading ghost
/// ring of one tile).
pub struct Pipeline {
    pub layout: Layout,
    pub quality: Quality,
    stream: Arc<Stream>,
    params: Tensor<f32>,
    scene: Tensor<i32>,
    /// Ping-pong pair for the jump flood. Which one holds the result after
    /// the passes is fixed by the pass count.
    seeds: [Tensor<i32>; 2],
    dist: Tensor<f32>,
    nearest: Tensor<i32>,
    /// Running average of radiance (RGBA). Frames alternate which one is
    /// read and which is written, so there are two captured graphs.
    light: [Tensor<f32>; 2],
    frame: Tensor<i32>,
    /// Pinned host copies for the per-frame transfers: no allocation, and
    /// the GPU copies directly from and to them.
    params_host: Pinned<f32>,
    scene_host: Pinned<i32>,
    frame_host: Pinned<i32>,
    /// `frame_host` without the ghost ring.
    pixels: Vec<u32>,
}

impl Pipeline {
    pub fn new(stream: &Arc<Stream>, layout: Layout, quality: Quality) -> Result<Self, Error> {
        let shape = [layout.buf_rows(), layout.buf_cols()];
        let rgba = [layout.buf_rows(), layout.buf_cols(), 4];
        Ok(Pipeline {
            layout,
            quality,
            stream: stream.clone(),
            params: api::zeros::<f32>(&[PARAMS]).sync_on(stream)?,
            scene: api::zeros::<i32>(&shape).sync_on(stream)?,
            seeds: [
                api::zeros::<i32>(&shape).sync_on(stream)?,
                api::zeros::<i32>(&shape).sync_on(stream)?,
            ],
            dist: api::zeros::<f32>(&shape).sync_on(stream)?,
            nearest: api::zeros::<i32>(&shape).sync_on(stream)?,
            light: [
                api::zeros::<f32>(&rgba).sync_on(stream)?,
                api::zeros::<f32>(&rgba).sync_on(stream)?,
            ],
            frame: api::zeros::<i32>(&shape).sync_on(stream)?,
            params_host: Pinned::new(stream, PARAMS)?,
            scene_host: Pinned::new(stream, shape[0] * shape[1])?,
            frame_host: Pinned::new(stream, shape[0] * shape[1])?,
            pixels: vec![0; layout.rows * layout.cols],
        })
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Copy a host scene (`buf_rows x buf_cols`, packed) into the fixed buffer.
    pub fn upload_scene(&mut self, scene: &[i32]) -> Result<(), Error> {
        self.scene_host.as_mut_slice().copy_from_slice(scene);
        self.scene_host.upload(&mut self.scene, &self.stream)
    }

    pub fn set_params(&mut self, params: &[f32; PARAMS]) -> Result<(), Error> {
        self.params_host.as_mut_slice().copy_from_slice(params);
        self.params_host.upload(&mut self.params, &self.stream)
    }

    /// Seed, flood, then derive the distance and nearest-surface fields.
    pub fn flood(&mut self, sub: &impl Submit) -> Result<(), Error> {
        sub.submit(kernels::seed_init(
            (&mut self.seeds[0]).partition([TILE, TILE]),
            &self.scene,
        ))?;
        let [a, b] = &mut self.seeds;
        let (mut src, mut dst) = (a, b);
        for k in self.layout.jfa_offsets() {
            {
                let (v, back, ahead) = jump_views(src, k)?;
                sub.submit(kernels::jfa_step(
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
                    back,
                    ahead,
                ))?;
            }
            std::mem::swap(&mut src, &mut dst);
        }
        sub.submit(kernels::distance(
            (&mut self.dist).partition([TILE, TILE]),
            &*src,
        ))?;
        sub.submit(kernels::nearest(
            (&mut self.nearest).partition([TILE, TILE]),
            &*src,
            &self.scene,
        ))?;
        Ok(())
    }

    /// Trace, accumulate and compose. Even frames read `light[0]` and write
    /// `light[1]`; odd frames the reverse.
    pub fn shade(&mut self, sub: &impl Submit, parity: usize) -> Result<(), Error> {
        let q = self.quality;
        let [l0, l1] = &mut self.light;
        let (prev, next) = if parity == 0 { (l0, l1) } else { (l1, l0) };
        sub.submit(kernels::radiance(
            next.partition([TILE, TILE, 4]),
            &*prev,
            &self.dist,
            &self.scene,
            &self.params,
            q.rays,
            q.steps,
            q.check_every,
        ))?;
        sub.submit(kernels::compose(
            (&mut self.frame).partition([TILE, TILE]),
            &*next,
            &self.scene,
            &self.dist,
            &self.nearest,
            &self.params,
        ))?;
        Ok(())
    }

    pub fn run(&mut self, sub: &impl Submit, parity: usize) -> Result<(), Error> {
        self.flood(sub)?;
        self.shade(sub, parity)
    }

    pub fn run_eager(&mut self, parity: usize) -> Result<(), Error> {
        let stream = self.stream.clone();
        self.run(&Eager(&stream), parity)
    }

    pub fn flood_eager(&mut self) -> Result<(), Error> {
        let stream = self.stream.clone();
        self.flood(&Eager(&stream))
    }

    /// One whole frame per graph, for even and odd frames.
    pub fn capture(&mut self) -> Result<[CudaGraph<()>; 2], Error> {
        let stream = self.stream.clone();
        let even = CudaGraph::scope(&stream, |s| Ok(self.run(s, 0)?))?;
        let odd = CudaGraph::scope(&stream, |s| Ok(self.run(s, 1)?))?;
        Ok([even, odd])
    }

    /// Just the flood, as its own graph (for timing).
    pub fn capture_flood(&mut self) -> Result<CudaGraph<()>, Error> {
        let stream = self.stream.clone();
        Ok(CudaGraph::scope(&stream, |s| Ok(self.flood(s)?))?)
    }

    /// The radiance written by the last frame of the given parity, RGBA.
    pub fn download_light(&self, parity: usize) -> Result<Vec<f32>, Error> {
        Ok(self.light[1 - parity]
            .dup()
            .to_host_vec()
            .sync_on(&self.stream)?)
    }

    fn final_seeds(&self) -> &Tensor<i32> {
        &self.seeds[self.layout.jfa_offsets().len() % 2]
    }

    pub fn download_seeds(&self) -> Result<Vec<i32>, Error> {
        Ok(self
            .final_seeds()
            .dup()
            .to_host_vec()
            .sync_on(&self.stream)?)
    }

    pub fn download_dist(&self) -> Result<Vec<f32>, Error> {
        Ok(self.dist.dup().to_host_vec().sync_on(&self.stream)?)
    }

    pub fn download_nearest(&self) -> Result<Vec<i32>, Error> {
        Ok(self.nearest.dup().to_host_vec().sync_on(&self.stream)?)
    }

    /// The composed frame without the ghost ring, as 0x00RRGGBB.
    ///
    /// The slice is a host copy that the next download overwrites, so the
    /// caller may draw overlays into it.
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
