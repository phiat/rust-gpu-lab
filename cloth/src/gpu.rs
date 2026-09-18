//! GPU side: Verlet integration, constraint passes, normals and shading.
//!
//! The cloth is a grid of particles, so it is a stencil problem like
//! `life` and `sand`: every particle reads its neighbors from shifted views
//! of the same tensor. What is new is that the grid holds objects with
//! several fields (a position and an inverse mass) instead of one number,
//! and that the constraint solver is iterative: many cheap passes per frame
//! that each move every particle a little toward satisfying its links.
//!
//! Passes are in-place in spirit but ping-pong in practice: a pass reads
//! one buffer and writes the other, so neighbors always see a consistent
//! snapshot and no tile ever reads what another tile is writing.

use std::sync::Arc;

use cuda_core::Stream;
use cutile::prelude::*;
use tilekit::{Eager, Pinned, Submit};

use crate::world::{Batch, Layout, PARAMS};

pub const TILE: usize = 32;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    type F = Tile<f32, { [32, 32] }>;
    type I = Tile<i32, { [32, 32] }>;
    type Mask = Tile<bool, { [32, 32] }>;
    /// A tile of particles: x, y, z, inverse mass.
    type V = Tile<f32, { [32, 32, 4] }>;
    type P = Tile<f32, { [64] }>;

    fn fill(v: f32) -> F {
        broadcast_scalar(v, const_shape![32, 32])
    }

    fn fill_i(v: i32) -> I {
        broadcast_scalar(v, const_shape![32, 32])
    }

    fn or(a: Mask, b: Mask) -> Mask {
        select(a, a, b)
    }

    fn and(a: Mask, b: Mask) -> Mask {
        select(a, b, a)
    }

    fn param(p: P, k: i32) -> F {
        let idx: Tile<i32, { [] }> = scalar_to_tile(k);
        let v: Tile<f32, { [1] }> = extract(p, [idx]);
        let v: Tile<f32, { [1, 1] }> = v.reshape(const_shape![1, 1]);
        v.broadcast(const_shape![32, 32])
    }

    fn params(t: &Tensor<f32, { [-1] }>) -> P {
        t.partition(const_shape![64]).load([0i32])
    }

    /// Slot `k` of this pass's entry in the pass table, broadcast from a
    /// `[4]` view of it (see `sand`: an `i32` argument would specialize
    /// the kernel).
    fn pass_entry(t: &Tensor<i32, { [-1] }>, k: i32) -> I {
        let p: Tile<i32, { [4] }> = t.partition(const_shape![4]).load([0i32]);
        let idx: Tile<i32, { [] }> = scalar_to_tile(k);
        let v: Tile<i32, { [1] }> = extract(p, [idx]);
        let v: Tile<i32, { [1, 1] }> = v.reshape(const_shape![1, 1]);
        v.broadcast(const_shape![32, 32])
    }

    /// Buffer row and column of every particle in this tile.
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

    /// True where buffer position (r, c) is a real particle, not a ghost.
    fn in_world(r: I, c: I, rows: i32, cols: i32) -> Mask {
        let r_ok: Mask = and(ge_tile(r, fill_i(32i32)), lt_tile(r, fill_i(32i32 + rows)));
        let c_ok: Mask = and(ge_tile(c, fill_i(32i32)), lt_tile(c, fill_i(32i32 + cols)));
        and(r_ok, c_ok)
    }

    fn channel(v: V, c: i32) -> F {
        let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let idx: Tile<i32, { [] }> = scalar_to_tile(c);
        let t: Tile<f32, { [32, 32, 1] }> = extract(v, [zero, zero, idx]);
        t.reshape(const_shape![32, 32])
    }

    fn pack4(x: F, y: F, z: F, w: F) -> V {
        let x: Tile<f32, { [32, 32, 1] }> = x.reshape(const_shape![32, 32, 1]);
        let y: Tile<f32, { [32, 32, 1] }> = y.reshape(const_shape![32, 32, 1]);
        let z: Tile<f32, { [32, 32, 1] }> = z.reshape(const_shape![32, 32, 1]);
        let w: Tile<f32, { [32, 32, 1] }> = w.reshape(const_shape![32, 32, 1]);
        let xy: Tile<f32, { [32, 32, 2] }> = cat(x, y, 2i32);
        let zw: Tile<f32, { [32, 32, 2] }> = cat(z, w, 2i32);
        cat(xy, zw, 2i32)
    }

    fn load4(t: &Tensor<f32, { [-1, -1, 4] }>, i: i32, j: i32) -> V {
        t.partition(const_shape![32, 32, 4]).load([i, j, 0i32])
    }

    fn sqrt_f(x: F) -> F {
        sqrt(x, rounding::NearestEven, ftz::Disabled)
    }

    fn dot3(ax: F, ay: F, az: F, bx: F, by: F, bz: F) -> F {
        ax * bx + ay * by + az * bz
    }

    // ---- Kernels, mirroring cpu.rs -----------------------------------------

    /// Wind strength at a point and time (`cpu::gust`).
    fn gust(time: F, x: F, y: F) -> F {
        let a: F = sin(time * fill(1.7f32) + x * fill(0.05f32) + y * fill(0.03f32));
        let b: F = sin(time * fill(3.1f32) - y * fill(0.08f32) + x * fill(0.02f32));
        fill(0.55f32) + fill(0.3f32) * a + fill(0.15f32) * b
    }

    /// Verlet step. Writes the moved particle and, to `out_prev`, the
    /// particle as it was (the next step's "previous").
    #[cutile::entry()]
    pub fn integrate(
        out: &mut Tensor<f32, { [32, 32, 4] }>,
        out_prev: &mut Tensor<f32, { [32, 32, 4] }>,
        pos: &Tensor<f32, { [-1, -1, 4] }>,
        prev: &Tensor<f32, { [-1, -1, 4] }>,
        nrm: &Tensor<f32, { [-1, -1, 4] }>,
        param_buf: &Tensor<f32, { [-1] }>,
    ) {
        let p: P = params(param_buf);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let cur: V = load4(pos, pid.0, pid.1);
        let old: V = load4(prev, pid.0, pid.1);
        let n: V = load4(nrm, pid.0, pid.1);
        let x: F = channel(cur, 0i32);
        let y: F = channel(cur, 1i32);
        let z: F = channel(cur, 2i32);
        let w: F = channel(cur, 3i32);
        let nx: F = channel(n, 0i32);
        let ny: F = channel(n, 1i32);
        let nz: F = channel(n, 2i32);
        let g: F = gust(param(p, 6i32), x, y);
        let wd: F = (nx * param(p, 3i32) + ny * param(p, 4i32) + nz * param(p, 5i32)) * g;
        let ax: F = nx * wd;
        let ay: F = param(p, 1i32) + ny * wd;
        let az: F = nz * wd;
        let damping: F = param(p, 2i32);
        let dt: F = param(p, 0i32);
        let dt2: F = dt * dt;
        let vx: F = (x - channel(old, 0i32)) * damping;
        let vy: F = (y - channel(old, 1i32)) * damping;
        let vz: F = (z - channel(old, 2i32)) * damping;
        let free: Mask = gt_tile(w, fill(0.0f32));
        let x2: F = select(free, x + vx + ax * dt2, x);
        let y2: F = select(free, y + vy + ay * dt2, y);
        let z2: F = select(free, z + vz + az * dt2, z);
        out.store(pack4(x2, y2, z2, w));
        out_prev.store(cur);
    }

    /// Move (px, py, pz) toward satisfying one constraint with neighbor `q`.
    fn pull(p: (F, F, F), w: F, q: V, valid: Mask, rest: F, k: F) -> (F, F, F) {
        let zero: F = fill(0.0f32);
        let dx: F = channel(q, 0i32) - p.0;
        let dy: F = channel(q, 1i32) - p.1;
        let dz: F = channel(q, 2i32) - p.2;
        let len: F = max_tile(sqrt_f(dx * dx + dy * dy + dz * dz), fill(1e-6f32));
        let wsum: F = w + channel(q, 3i32);
        let share: F = select(gt_tile(wsum, zero), w / wsum, zero);
        let s: F = (len - rest) / len * share * k;
        (
            p.0 + select(valid, dx * s, zero),
            p.1 + select(valid, dy * s, zero),
            p.2 + select(valid, dz * s, zero),
        )
    }

    /// One constraint pass (`cpu::pair`). `minus` and `plus` are views of
    /// `mm` shifted by minus and plus the batch's step (see `views`), so
    /// every particle finds its partner in one of them; which one depends
    /// on the batch parity.
    #[cutile::entry()]
    pub fn pair(
        out: &mut Tensor<f32, { [32, 32, 4] }>,
        mm: &Tensor<f32, { [-1, -1, 4] }>,
        minus: &Tensor<f32, { [-1, -1, 4] }>,
        plus: &Tensor<f32, { [-1, -1, 4] }>,
        anchor: &Tensor<f32, { [-1, -1, 4] }>,
        param_buf: &Tensor<f32, { [-1] }>,
        pass_buf: &Tensor<i32, { [-1] }>,
    ) {
        let p: P = params(param_buf);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let (i, j) = (pid.0, pid.1);
        // Views load at block - 1 (clamped: ghost tiles keep their
        // particles anyway, since those have no mass).
        let up: i32 = max(i - 1i32, 0i32);
        let left: i32 = max(j - 1i32, 0i32);
        let shape = mm.shape();
        let rows: i32 = shape[0] - 64i32;
        let cols: i32 = shape[1] - 64i32;
        let (row, col) = coords();
        let zero: F = fill(0.0f32);
        let izero: I = fill_i(0i32);
        let ione: I = fill_i(1i32);

        let cur: V = load4(mm, i, j);
        let x0: F = channel(cur, 0i32);
        let y0: F = channel(cur, 1i32);
        let z0: F = channel(cur, 2i32);
        let w: F = channel(cur, 3i32);

        // Which way is my partner? (`Batch::partner`)
        let b: I = pass_entry(pass_buf, 0i32);
        let kind: I = pass_entry(pass_buf, 1i32);
        let lg: I = pass_entry(pass_buf, 2i32);
        let k_slot: I = pass_entry(pass_buf, 3i32);
        let horizontal: Mask = eq_tile(kind, ione);
        let diagonal: Mask = ge_tile(kind, fill_i(2i32));
        let coord: I = select(horizontal, col, row);
        let group: I = shri(coord, lg);
        let plus_side: Mask = eq_tile(andi(group + b, ione), izero);
        let s: I = ione + lg;
        let step_r: I = select(horizontal, izero, select(diagonal, ione, s));
        let step_c: I = select(
            horizontal,
            s,
            select(
                diagonal,
                select(eq_tile(kind, fill_i(2i32)), ione, fill_i(-1i32)),
                izero,
            ),
        );
        let sign: I = select(plus_side, ione, fill_i(-1i32));
        let valid: Mask = in_world(row + sign * step_r, col + sign * step_c, rows, cols);
        let qp: V = load4(plus, up, left);
        let qm: V = load4(minus, up, left);
        let qx: F = select(plus_side, channel(qp, 0i32), channel(qm, 0i32));
        let qy: F = select(plus_side, channel(qp, 1i32), channel(qm, 1i32));
        let qz: F = select(plus_side, channel(qp, 2i32), channel(qm, 2i32));
        let qw: F = select(plus_side, channel(qp, 3i32), channel(qm, 3i32));
        let spacing: F = param(p, 15i32);
        let s_f: F = convert_tile(s);
        let rest: F = spacing * select(diagonal, fill(1.4142135f32), s_f);
        let k: F = select(
            eq_tile(k_slot, izero),
            param(p, 12i32),
            select(eq_tile(k_slot, ione), param(p, 13i32), param(p, 14i32)),
        );

        // Project the link: both ends move by their share of the mass.
        let dx: F = qx - x0;
        let dy: F = qy - y0;
        let dz: F = qz - z0;
        let len: F = max_tile(sqrt_f(dx * dx + dy * dy + dz * dz), fill(1e-6f32));
        let wsum: F = w + qw;
        let share: F = select(gt_tile(wsum, zero), w / wsum, zero);
        let sc: F = (len - rest) / len * share * k;
        let x: F = select(valid, x0 + dx * sc, x0);
        let y: F = select(valid, y0 + dy * sc, y0);
        let z: F = select(valid, z0 + dz * sc, z0);

        // Long range attachment: never farther from the anchor than the
        // flat cloth allows.
        let free: Mask = gt_tile(w, zero);
        let a: V = load4(anchor, i, j);
        let ax: F = channel(a, 0i32);
        let ay: F = channel(a, 1i32);
        let az: F = channel(a, 2i32);
        let reach: F = channel(a, 3i32);
        let vx: F = x - ax;
        let vy: F = y - ay;
        let vz: F = z - az;
        let dist: F = sqrt_f(vx * vx + vy * vy + vz * vz);
        let pull: F = reach / max_tile(dist, fill(1e-6f32));
        let far: Mask = and(free, and(gt_tile(reach, zero), gt_tile(dist, reach)));
        let x: F = select(far, ax + vx * pull, x);
        let y: F = select(far, ay + vy * pull, y);
        let z: F = select(far, az + vz * pull, z);

        // Collisions: out of the sphere, above the floor.
        let sx: F = param(p, 7i32);
        let sy: F = param(p, 8i32);
        let sz: F = param(p, 9i32);
        let sr: F = param(p, 10i32);
        let vx: F = x - sx;
        let vy: F = y - sy;
        let vz: F = z - sz;
        let dist: F = sqrt_f(vx * vx + vy * vy + vz * vz);
        let push: F = sr / max_tile(dist, fill(1e-6f32));
        let inside: Mask = and(free, lt_tile(dist, sr));
        let x: F = select(inside, sx + vx * push, x);
        let y: F = select(inside, sy + vy * push, y);
        let z: F = select(inside, sz + vz * push, z);
        let y: F = select(free, max_tile(y, param(p, 11i32)), y);
        out.store(pack4(x, y, z, w));
    }

    /// Unit normals from the four structural neighbors (`cpu::normals`),
    /// read from shifted views (see `views`).
    #[cutile::entry()]
    pub fn normals(
        out: &mut Tensor<f32, { [32, 32, 4] }>,
        mm: &Tensor<f32, { [-1, -1, 4] }>,
        u1: &Tensor<f32, { [-1, -1, 4] }>,
        d1: &Tensor<f32, { [-1, -1, 4] }>,
        l1: &Tensor<f32, { [-1, -1, 4] }>,
        r1: &Tensor<f32, { [-1, -1, 4] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let (i, j) = (pid.0, pid.1);
        let up: i32 = max(i - 1i32, 0i32);
        let left: i32 = max(j - 1i32, 0i32);
        let shape = mm.shape();
        let rows: i32 = shape[0] - 64i32;
        let cols: i32 = shape[1] - 64i32;
        let (row, col) = coords();
        let one: I = fill_i(1i32);

        let cur: V = load4(mm, i, j);
        let l: V = load4(l1, up, left);
        let r: V = load4(r1, up, left);
        let u: V = load4(u1, up, left);
        let d: V = load4(d1, up, left);
        let vl: Mask = in_world(row, col - one, rows, cols);
        let vr: Mask = in_world(row, col + one, rows, cols);
        let vu: Mask = in_world(row - one, col, rows, cols);
        let vd: Mask = in_world(row + one, col, rows, cols);
        let px: F = channel(cur, 0i32);
        let py: F = channel(cur, 1i32);
        let pz: F = channel(cur, 2i32);
        let tx0: F = select(vr, channel(r, 0i32), px) - select(vl, channel(l, 0i32), px);
        let tx1: F = select(vr, channel(r, 1i32), py) - select(vl, channel(l, 1i32), py);
        let tx2: F = select(vr, channel(r, 2i32), pz) - select(vl, channel(l, 2i32), pz);
        let ty0: F = select(vd, channel(d, 0i32), px) - select(vu, channel(u, 0i32), px);
        let ty1: F = select(vd, channel(d, 1i32), py) - select(vu, channel(u, 1i32), py);
        let ty2: F = select(vd, channel(d, 2i32), pz) - select(vu, channel(u, 2i32), pz);
        let nx: F = tx1 * ty2 - tx2 * ty1;
        let ny: F = tx2 * ty0 - tx0 * ty2;
        let nz: F = tx0 * ty1 - tx1 * ty0;
        let len: F = max_tile(sqrt_f(dot3(nx, ny, nz, nx, ny, nz)), fill(1e-9f32));
        out.store(pack4(nx / len, ny / len, nz / len, fill(0.0f32)));
    }

    /// Project and light every particle (`cpu::shade`): screen x, screen
    /// y, depth, and the color packed as 0xRRGGBB in an f32.
    #[cutile::entry()]
    pub fn shade(
        out: &mut Tensor<f32, { [32, 32, 4] }>,
        pos: &Tensor<f32, { [-1, -1, 4] }>,
        nrm: &Tensor<f32, { [-1, -1, 4] }>,
        param_buf: &Tensor<f32, { [-1] }>,
    ) {
        let p: P = params(param_buf);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let cur: V = load4(pos, pid.0, pid.1);
        let n: V = load4(nrm, pid.0, pid.1);
        let zero: F = fill(0.0f32);
        let d0: F = channel(cur, 0i32) - param(p, 29i32);
        let d1: F = channel(cur, 1i32) - param(p, 30i32);
        let d2: F = channel(cur, 2i32) - param(p, 31i32);
        let qx: F = param(p, 20i32) * d0 + param(p, 21i32) * d1 + param(p, 22i32) * d2;
        let qy: F = param(p, 23i32) * d0 + param(p, 24i32) * d1 + param(p, 25i32) * d2;
        let qz: F =
            param(p, 26i32) * d0 + param(p, 27i32) * d1 + param(p, 28i32) * d2 + param(p, 32i32);
        let focal: F = param(p, 33i32);
        let sx: F = param(p, 34i32) + focal * qx / qz;
        let sy: F = param(p, 35i32) - focal * qy / qz;
        let nx: F = channel(n, 0i32);
        let ny: F = channel(n, 1i32);
        let nz: F = channel(n, 2i32);
        let n0: F = param(p, 20i32) * nx + param(p, 21i32) * ny + param(p, 22i32) * nz;
        let n1: F = param(p, 23i32) * nx + param(p, 24i32) * ny + param(p, 25i32) * nz;
        let n2: F = param(p, 26i32) * nx + param(p, 27i32) * ny + param(p, 28i32) * nz;
        let vl: F = max_tile(sqrt_f(dot3(qx, qy, qz, qx, qy, qz)), fill(1e-6f32));
        let vx: F = zero - qx / vl;
        let vy: F = zero - qy / vl;
        let vz: F = zero - qz / vl;
        let facing: F = dot3(n0, n1, n2, vx, vy, vz);
        let back: Mask = lt_tile(facing, zero);
        let n0: F = select(back, zero - n0, n0);
        let n1: F = select(back, zero - n1, n1);
        let n2: F = select(back, zero - n2, n2);
        let lx: F = param(p, 36i32);
        let ly: F = param(p, 37i32);
        let lz: F = param(p, 38i32);
        let diff: F = max_tile(dot3(n0, n1, n2, lx, ly, lz), zero);
        let hx: F = lx + vx;
        let hy: F = ly + vy;
        let hz: F = lz + vz;
        let hl: F = max_tile(sqrt_f(dot3(hx, hy, hz, hx, hy, hz)), fill(1e-6f32));
        let spec: F = pow(
            max_tile(dot3(n0, n1, n2, hx, hy, hz) / hl, zero),
            fill(32.0f32),
        );
        let tint: F = select(back, fill(0.7f32), fill(1.0f32));
        // Checkerboard of 16x16 squares in the two cloth colors
        // (`world::base_color`).
        let (row, col) = coords();
        let sq: I = andi(
            shri(row - fill_i(32i32), fill_i(4i32)) + shri(col - fill_i(32i32), fill_i(4i32)),
            fill_i(1i32),
        );
        let a: Mask = eq_tile(sq, fill_i(0i32));
        let br: F = select(a, fill(0.82f32), fill(0.92f32));
        let bg: F = select(a, fill(0.22f32), fill(0.85f32));
        let bb: F = select(a, fill(0.18f32), fill(0.72f32));
        let lit: F = fill(0.2f32) + fill(0.8f32) * diff;
        let hi: F = fill(0.35f32) * spec;
        let one: F = fill(1.0f32);
        let r: F = min_tile(max_tile(br * tint * lit + hi, zero), one);
        let g: F = min_tile(max_tile(bg * tint * lit + hi, zero), one);
        let b: F = min_tile(max_tile(bb * tint * lit + hi, zero), one);
        let half: F = fill(0.5f32);
        let k255: F = fill(255.0f32);
        let rgb: F = floor(r * k255 + half) * fill(65536.0f32);
        let rgb: F = rgb + floor(g * k255 + half) * fill(256.0f32);
        let rgb: F = rgb + floor(b * k255 + half);
        out.store(pack4(sx, sy, qz, rgb));
    }
}

/// Views of `src` shifted by each `(dr, dc)` offset, so that block
/// `[i - 1, j - 1]` of every view holds the `(dr, dc)` neighbors of block
/// `[i, j]` of `src`: view (dr, dc) starts at row `TILE + dr`, column
/// `TILE + dc`, which the ghost ring makes possible for offsets of either
/// sign. Every view has the same shape, so a kernel's argument pattern is
/// the same whichever offsets a launch uses, and the JIT (which
/// specializes on shape divisibility) builds it once. The length leaves
/// the last ghost block a partial tile, which loads zero-padded rather
/// than asserting.
fn views<'a>(
    src: &'a Tensor<f32>,
    offsets: &[(i32, i32)],
) -> Result<Vec<TensorView<'a, f32>>, Error> {
    let (rows, cols) = (src.shape()[0] as usize, src.shape()[1] as usize);
    let (lr, lc) = (rows - 2 * TILE + 1, cols - 2 * TILE + 1);
    offsets
        .iter()
        .map(|&(dr, dc)| {
            let (r, c) = ((TILE as i32 + dr) as usize, (TILE as i32 + dc) as usize);
            src.slice(&[r..r + lr, c..c + lc, 0..4])
        })
        .collect()
}

/// Device buffers: ping-pong positions and previous positions, normals,
/// the shaded output, and the parameter and pass tables.
pub struct Pipeline {
    pub substeps: usize,
    /// Solver iterations per substep; each is 12 batch passes.
    pub iterations: usize,
    stream: Arc<Stream>,
    pos: [Tensor<f32>; 2],
    prev: [Tensor<f32>; 2],
    nrm: Tensor<f32>,
    anchor: Tensor<f32>,
    screen: Tensor<f32>,
    params: Tensor<f32>,
    /// One `Batch::entry` per pass of an iteration, 4 elements each, so
    /// each pass's view has the same alignment and shape (one kernel
    /// variant; see `sand`).
    pass_table: Tensor<i32>,
    pos_host: Pinned<f32>,
    params_host: Pinned<f32>,
    screen_host: Pinned<f32>,
}

impl Pipeline {
    pub fn new(
        stream: &Arc<Stream>,
        layout: Layout,
        substeps: usize,
        iterations: usize,
    ) -> Result<Self, Error> {
        assert!(substeps >= 1 && iterations >= 1);
        let shape = [layout.buf_rows(), layout.buf_cols(), 4];
        let n = layout.len() * 4;
        let zeros = || api::zeros::<f32>(&shape).sync_on(stream);
        Ok(Pipeline {
            substeps,
            iterations,
            stream: stream.clone(),
            pos: [zeros()?, zeros()?],
            prev: [zeros()?, zeros()?],
            nrm: zeros()?,
            anchor: zeros()?,
            screen: zeros()?,
            params: api::zeros::<f32>(&[PARAMS]).sync_on(stream)?,
            pass_table: {
                let table: Vec<i32> = Batch::iteration().iter().flat_map(|b| b.entry()).collect();
                api::copy_host_vec_to_device(&Arc::new(table)).sync_on(stream)?
            },
            pos_host: Pinned::new(stream, n)?,
            params_host: Pinned::new(stream, PARAMS)?,
            screen_host: Pinned::new(stream, n)?,
        })
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Load a cloth at rest and its anchors into buffer `parity` (the one
    /// the next frame of that parity reads) and compute its normals.
    pub fn upload_cloth(
        &mut self,
        parity: usize,
        cloth: &[f32],
        anchors: &[f32],
    ) -> Result<(), Error> {
        self.pos_host.as_mut_slice().copy_from_slice(anchors);
        self.pos_host.upload(&mut self.anchor, &self.stream)?;
        self.pos_host.as_mut_slice().copy_from_slice(cloth);
        self.pos_host.upload(&mut self.pos[parity], &self.stream)?;
        self.pos_host.upload(&mut self.prev[parity], &self.stream)?;
        let stream = self.stream.clone();
        let v = views(&self.pos[parity], &STENCIL)?;
        kernels::normals(
            (&mut self.nrm).partition([TILE, TILE, 4]),
            &self.pos[parity],
            &v[0],
            &v[1],
            &v[2],
            &v[3],
        )
        .sync_on(&stream)?;
        Ok(())
    }

    pub fn params_mut(&mut self) -> &mut [f32] {
        self.params_host.as_mut_slice()
    }

    pub fn upload_params(&mut self) -> Result<(), Error> {
        self.params_host.upload(&mut self.params, &self.stream)
    }

    /// The buffer a frame starting in `parity` leaves its result in.
    pub fn end_parity(&self, parity: usize) -> usize {
        (parity + self.substeps) % 2
    }

    /// One frame: `substeps` x (integrate, `iterations` x 12 batch passes),
    /// then normals and shading. Every pass flips buffers.
    pub fn run(&mut self, sub: &impl Submit, parity: usize) -> Result<(), Error> {
        let [p0, p1] = &mut self.pos;
        let [q0, q1] = &mut self.prev;
        let (mut src, mut dst, mut src_prev, mut dst_prev) = if parity == 0 {
            (p0, p1, q0, q1)
        } else {
            (p1, p0, q1, q0)
        };
        for _ in 0..self.substeps {
            sub.submit(kernels::integrate(
                dst.partition([TILE, TILE, 4]),
                dst_prev.partition([TILE, TILE, 4]),
                &*src,
                &*src_prev,
                &self.nrm,
                &self.params,
            ))?;
            std::mem::swap(&mut src, &mut dst);
            std::mem::swap(&mut src_prev, &mut dst_prev);
            for _ in 0..self.iterations {
                for (pass, batch) in Batch::iteration().into_iter().enumerate() {
                    let (sr, sc) = batch.step();
                    let v = views(src, &[(-sr, -sc), (sr, sc)])?;
                    let range = 4 * pass..4 * pass + 4;
                    let pass_view = self.pass_table.slice(std::slice::from_ref(&range))?;
                    sub.submit(kernels::pair(
                        dst.partition([TILE, TILE, 4]),
                        &*src,
                        &v[0],
                        &v[1],
                        &self.anchor,
                        &self.params,
                        &pass_view,
                    ))?;
                    drop(v);
                    std::mem::swap(&mut src, &mut dst);
                }
            }
        }
        let v = views(src, &STENCIL)?;
        sub.submit(kernels::normals(
            (&mut self.nrm).partition([TILE, TILE, 4]),
            &*src,
            &v[0],
            &v[1],
            &v[2],
            &v[3],
        ))?;
        sub.submit(kernels::shade(
            (&mut self.screen).partition([TILE, TILE, 4]),
            &*src,
            &self.nrm,
            &self.params,
        ))?;
        Ok(())
    }

    pub fn run_eager(&mut self, parity: usize) -> Result<(), Error> {
        let stream = self.stream.clone();
        self.run(&Eager(&stream), parity)
    }

    /// One graph per starting parity.
    pub fn capture(&mut self) -> Result<[CudaGraph<()>; 2], Error> {
        let stream = self.stream.clone();
        let even = CudaGraph::scope(&stream, |s| Ok(self.run(s, 0)?))?;
        let odd = CudaGraph::scope(&stream, |s| Ok(self.run(s, 1)?))?;
        Ok([even, odd])
    }

    /// The particles in buffer `parity` (use `end_parity`).
    pub fn download_pos(&mut self, parity: usize) -> Result<&[f32], Error> {
        self.pos_host.download(&self.pos[parity], &self.stream)?;
        Ok(self.pos_host.as_slice())
    }

    /// The shaded particles of the last frame.
    pub fn download_screen(&mut self) -> Result<&[f32], Error> {
        self.screen_host.download(&self.screen, &self.stream)?;
        Ok(self.screen_host.as_slice())
    }
}

/// View offsets for `normals`: up, down, left, right.
const STENCIL: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
