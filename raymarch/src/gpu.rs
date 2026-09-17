//! GPU ray marcher: one cuTile tile kernel that sphere-traces, shades and
//! packs a 0RGB pixel for every pixel of its 32x32 tile.

use std::sync::Arc;

use cuda_core::Stream;
use cutile::prelude::*;

use crate::scene::PARAMS;

/// The kernel spells its tile shape as a literal, so the host has to
/// partition with exactly this size.
pub const TILE: usize = 32;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    // Every shape here is the literal `32`, not a `const B: i32` generic.
    // With a generic entry point, device-function results come back typed
    // `{[32, 32]}` while locals stay `{[B, B]}`, and `+ - *` reject the mix
    // (see filters/README.md). With no generic there is only one spelling,
    // so helper functions compose freely. Aliases keep it readable.
    type F = Tile<f32, { [32, 32] }>;
    type Mask = Tile<bool, { [32, 32] }>;
    type Params = Tile<f32, { [32] }>;

    fn fill(v: f32) -> F {
        broadcast_scalar(v, const_shape![32, 32])
    }

    /// `params[k]`, broadcast over the tile.
    fn param(p: Params, k: i32) -> F {
        let idx: Tile<i32, { [] }> = scalar_to_tile(k);
        let v: Tile<f32, { [1] }> = extract(p, [idx]);
        let v: Tile<f32, { [1, 1] }> = v.reshape(const_shape![1, 1]);
        v.broadcast(const_shape![32, 32])
    }

    fn clamp01(x: F) -> F {
        min_tile(max_tile(x, fill(0.0f32)), fill(1.0f32))
    }

    fn mix(a: F, b: F, t: F) -> F {
        a + (b - a) * t
    }

    fn length2(x: F, y: F) -> F {
        sqrt(x * x + y * y, rounding::NearestEven, ftz::Disabled)
    }

    fn length3(x: F, y: F, z: F) -> F {
        sqrt(x * x + y * y + z * z, rounding::NearestEven, ftz::Disabled)
    }

    fn normalize3(x: F, y: F, z: F) -> (F, F, F) {
        let len: F = length3(x, y, z);
        (x / len, y / len, z / len)
    }

    /// Row and column of each pixel center in this tile.
    fn pixel_centers() -> (F, F) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let idx: Tile<i32, { [32] }> = iota(const_shape![32]);
        let row0: Tile<i32, { [32] }> = broadcast_scalar(pid.0 * 32i32, const_shape![32]);
        let col0: Tile<i32, { [32] }> = broadcast_scalar(pid.1 * 32i32, const_shape![32]);
        let half: Tile<f32, { [32] }> = broadcast_scalar(0.5f32, const_shape![32]);
        let rows: Tile<f32, { [32] }> = convert_tile(idx + row0);
        let rows: Tile<f32, { [32] }> = rows + half;
        let cols: Tile<f32, { [32] }> = convert_tile(idx + col0);
        let cols: Tile<f32, { [32] }> = cols + half;
        let rows: Tile<f32, { [32, 1] }> = rows.reshape(const_shape![32, 1]);
        let cols: Tile<f32, { [1, 32] }> = cols.reshape(const_shape![1, 32]);
        (
            rows.broadcast(const_shape![32, 32]),
            cols.broadcast(const_shape![32, 32]),
        )
    }

    // ---- Signed distance functions -------------------------------------
    //
    // Each returns, for every pixel's sample point, a lower bound on the
    // distance to that shape. `rs`/`rc` (sin/cos of the spin angle) and `bob`
    // are animation values from the parameter block.

    fn sd_round_box(x: F, y: F, z: F, half: f32, radius: f32) -> F {
        let zero: F = fill(0.0f32);
        let inner: F = fill(half - radius);
        let qx: F = absf(x) - inner;
        let qy: F = absf(y) - inner;
        let qz: F = absf(z) - inner;
        let outside: F = length3(max_tile(qx, zero), max_tile(qy, zero), max_tile(qz, zero));
        let inside: F = min_tile(max_tile(qx, max_tile(qy, qz)), zero);
        outside + inside - fill(radius)
    }

    /// Polynomial smooth minimum: blends two shapes within distance `k`.
    fn smin(a: F, b: F, k: f32) -> F {
        let h: F = clamp01(fill(0.5f32) + (b - a) * fill(0.5f32 / k));
        mix(b, a, h) - fill(k) * h * (fill(1.0f32) - h)
    }

    /// Mirror the xz half-plane on the far side of the line at angle
    /// (cos, sin) onto the near side: p -= 2 * max(dot(p, n), 0) * n.
    fn fold(x: F, z: F, cos: f32, sin: f32) -> (F, F) {
        let d: F = max_tile(z * fill(cos) - x * fill(sin), fill(0.0f32));
        let d2: F = d * fill(2.0f32);
        (x + d2 * fill(sin), z - d2 * fill(cos))
    }

    fn sd_column(x: F, y: F, z: F, cx: f32, cz: f32) -> F {
        let flat: F = length2(x - fill(cx), z - fill(cz)) - fill(0.24f32);
        max_tile(flat, y - fill(2.6f32))
    }

    /// Two rings of columns (16 and 32) around the courtyard.
    ///
    /// Polar repetition without trig: `abs` folds space into one quadrant,
    /// a min/max swap into 45 degrees, and each mirror fold halves the
    /// wedge again. One column in the final wedge stands in for all of them.
    /// (`atan2` + `cos` + `sin` did the same job at 3.5x the frame time.)
    fn sd_pillars(x: F, y: F, z: F) -> F {
        let ax: F = absf(x);
        let az: F = absf(z);
        let (x45, z45) = (max_tile(ax, az), min_tile(ax, az));
        // 22.5 degree wedge: 16 columns at radius 6.5, centered at 11.25 degrees.
        let (x22, z22) = fold(x45, z45, 0.92387953f32, 0.38268343f32);
        let inner: F = sd_column(x22, y, z22, 6.3751044f32, 1.2680871f32);
        // 11.25 degree wedge: 32 columns at radius 10.5, centered at 5.625 degrees.
        let (x11, z11) = fold(x22, z22, 0.98078528f32, 0.19509032f32);
        let outer: F = sd_column(x11, y, z11, 10.449440f32, 1.0291800f32);
        min_tile(inner, outer)
    }

    /// A bobbing sphere melted into a spinning rounded cube.
    fn sd_blob(x: F, y: F, z: F, rs: F, rc: F, bob: F) -> F {
        let ball: F = length3(x, y - bob, z) - fill(0.5f32);
        let bx: F = rc * x + rs * z;
        let bz: F = rc * z - rs * x;
        let cube: F = sd_round_box(bx, y - fill(0.55f32), bz, 0.45f32, 0.08f32);
        smin(ball, cube, 0.35f32)
    }

    /// An upright torus spinning about the vertical axis.
    fn sd_ring(x: F, y: F, z: F, rs: F, rc: F) -> F {
        let lx: F = x - fill(1.9f32);
        let ly: F = y - fill(0.75f32);
        let lz: F = z + fill(0.9f32);
        let tx: F = rc * lx - rs * lz;
        let tz: F = rs * lx + rc * lz;
        length2(length2(tx, ly) - fill(0.55f32), tz) - fill(0.2f32)
    }

    fn sd_ball(x: F, y: F, z: F) -> F {
        length3(x + fill(1.7f32), y - fill(0.5f32), z - fill(1.0f32)) - fill(0.5f32)
    }

    fn scene(x: F, y: F, z: F, rs: F, rc: F, bob: F) -> F {
        let d: F = min_tile(y, sd_pillars(x, y, z));
        let d: F = min_tile(d, sd_blob(x, y, z, rs, rc, bob));
        let d: F = min_tile(d, sd_ring(x, y, z, rs, rc));
        min_tile(d, sd_ball(x, y, z))
    }

    /// Id of the closest shape: 0 ground, 1 pillars, 2 blob, 3 ring, 4 ball.
    fn material(x: F, y: F, z: F, rs: F, rc: F, bob: F) -> F {
        let d: F = y;
        let m: F = fill(0.0f32);
        let dp: F = sd_pillars(x, y, z);
        let closer: Mask = lt_tile(dp, d);
        let d: F = select(closer, dp, d);
        let m: F = select(closer, fill(1.0f32), m);
        let db: F = sd_blob(x, y, z, rs, rc, bob);
        let closer: Mask = lt_tile(db, d);
        let d: F = select(closer, db, d);
        let m: F = select(closer, fill(2.0f32), m);
        let dr: F = sd_ring(x, y, z, rs, rc);
        let closer: Mask = lt_tile(dr, d);
        let d: F = select(closer, dr, d);
        let m: F = select(closer, fill(3.0f32), m);
        let closer: Mask = lt_tile(sd_ball(x, y, z), d);
        select(closer, fill(4.0f32), m)
    }

    /// Pick one of five per-material constants.
    fn by_material(m: F, ground: F, pillars: f32, blob: f32, ring: f32, ball: f32) -> F {
        let v: F = select(lt_tile(m, fill(3.5f32)), fill(ring), fill(ball));
        let v: F = select(lt_tile(m, fill(2.5f32)), fill(blob), v);
        let v: F = select(lt_tile(m, fill(1.5f32)), fill(pillars), v);
        select(lt_tile(m, fill(0.5f32)), ground, v)
    }

    // ---- Shading ----------------------------------------------------------

    /// Surface normal from 4 scene samples at the corners of a tetrahedron.
    ///
    /// Written as a loop so the JIT inlines `scene` once instead of four
    /// times: every inlined copy adds seconds of `tileiras` compile time.
    fn normal(x: F, y: F, z: F, rs: F, rc: F, bob: F) -> (F, F, F) {
        let e: f32 = 0.0005f32;
        let mut nx: F = fill(0.0f32);
        let mut ny: F = fill(0.0f32);
        let mut nz: F = fill(0.0f32);
        for i in 0i32..4i32 {
            // Corners (1,-1,-1), (-1,-1,1), (-1,1,-1), (1,1,1).
            let kx: f32 = if i == 0i32 || i == 3i32 {
                1.0f32
            } else {
                -1.0f32
            };
            let ky: f32 = if i >= 2i32 { 1.0f32 } else { -1.0f32 };
            let kz: f32 = if i == 1i32 || i == 3i32 {
                1.0f32
            } else {
                -1.0f32
            };
            let d: F = scene(
                x + fill(kx * e),
                y + fill(ky * e),
                z + fill(kz * e),
                rs,
                rc,
                bob,
            );
            nx = nx + fill(kx) * d;
            ny = ny + fill(ky) * d;
            nz = nz + fill(kz) * d;
        }
        normalize3(nx, ny, nz)
    }

    /// Soft shadow: march toward the light and keep the narrowest miss.
    ///
    /// Uses the previous step's distance to estimate the closest approach
    /// between samples, which removes the banding of the plain
    /// `min(res, k * h / t)` version. A fixed step count with a mask
    /// replaces the usual `break`.
    fn soft_shadow(x: F, y: F, z: F, lx: F, ly: F, lz: F, rs: F, rc: F, bob: F, steps: i32) -> F {
        let zero: F = fill(0.0f32);
        let mut t: F = fill(0.02f32);
        let mut res: F = fill(1.0f32);
        let mut prev: F = fill(1.0e10f32);
        let tmax: F = fill(12.0f32);
        for _i in 0i32..steps {
            let h: F = scene(x + lx * t, y + ly * t, z + lz * t, rs, rc, bob);
            let back: F = h * h / (fill(2.0f32) * prev);
            let across: F = sqrt(
                max_tile(h * h - back * back, zero),
                rounding::NearestEven,
                ftz::Disabled,
            );
            let est: F = fill(10.0f32) * across / max_tile(t - back, fill(0.0001f32));
            let in_range: Mask = lt_tile(t, tmax);
            res = select(in_range, min_tile(res, est), res);
            prev = h;
            t = t + min_tile(max_tile(h, fill(0.01f32)), fill(0.25f32));
        }
        let res: F = clamp01(res);
        res * res * (fill(3.0f32) - fill(2.0f32) * res)
    }

    /// Ambient occlusion from 5 samples along the normal.
    fn occlusion(x: F, y: F, z: F, nx: F, ny: F, nz: F, rs: F, rc: F, bob: F) -> F {
        let mut occ: F = fill(0.0f32);
        let mut weight: f32 = 1.0f32;
        let mut h: f32 = 0.01f32;
        for _i in 0i32..5i32 {
            let hh: F = fill(h);
            let d: F = scene(x + nx * hh, y + ny * hh, z + nz * hh, rs, rc, bob);
            occ = occ + (hh - d) * fill(weight);
            weight = weight * 0.95f32;
            h = h + 0.03f32;
        }
        clamp01(fill(1.0f32) - fill(3.0f32) * occ)
    }

    /// Black -> red -> yellow -> white, for the debug views.
    fn heat(v: F) -> (F, F, F) {
        let v3: F = clamp01(v) * fill(3.0f32);
        (
            clamp01(v3),
            clamp01(v3 - fill(1.0f32)),
            clamp01(v3 - fill(2.0f32)),
        )
    }

    fn to_byte(c: F) -> Tile<i32, { [32, 32] }> {
        convert_tile(c * fill(255.0f32) + fill(0.5f32))
    }

    /// Render one tile into packed 0x00RRGGBB pixels.
    ///
    /// `params` layout (see `scene::slot`): 0 width, 1 height, 2-4 eye,
    /// 5-7 forward, 8-10 right, 11-13 up, 14 focal length, 15 time,
    /// 16-18 sun direction, 19 spin sin, 20 spin cos, 21 bob height,
    /// 22 view mode, 23 max steps.
    #[cutile::entry()]
    pub fn render(
        out: &mut Tensor<i32, { [32, 32] }>,
        params: &Tensor<f32, { [-1] }>,
        max_steps: i32,
        check_every: i32,
        shadow_steps: i32,
    ) {
        let part = params.partition(const_shape![32]);
        let p: Params = part.load([0i32]);
        let zero: F = fill(0.0f32);
        let one: F = fill(1.0f32);

        // A camera ray through every pixel center.
        let (row, col) = pixel_centers();
        let width: F = param(p, 0i32);
        let height: F = param(p, 1i32);
        let two: F = fill(2.0f32);
        let u: F = (col * two - width) / height;
        let v: F = (height - row * two) / height;
        let focal: F = param(p, 14i32);
        let (dx, dy, dz) = normalize3(
            param(p, 5i32) * focal + param(p, 8i32) * u + param(p, 11i32) * v,
            param(p, 6i32) * focal + param(p, 9i32) * u + param(p, 12i32) * v,
            param(p, 7i32) * focal + param(p, 10i32) * u + param(p, 13i32) * v,
        );
        let ox: F = param(p, 2i32);
        let oy: F = param(p, 3i32);
        let oz: F = param(p, 4i32);
        let rs: F = param(p, 19i32);
        let rc: F = param(p, 20i32);
        let bob: F = param(p, 21i32);

        // Sphere tracing. Each ray advances by the scene distance until it
        // is within eps * t of a surface or passes tmax. Finished rays are
        // frozen with `select`; the tile stops once all of its rays are.
        let eps: F = fill(0.0004f32);
        let tmax: F = fill(60.0f32);
        let mut t: F = fill(0.05f32);
        let mut steps: F = zero;
        let mut active: Mask = lt_tile(zero, one);
        let mut done: i32 = 0i32;
        let mut tile_steps: f32 = 0.0f32;
        while done < max_steps {
            let chunk: i32 = min(check_every, max_steps - done);
            for _i in 0i32..chunk {
                let d: F = scene(ox + dx * t, oy + dy * t, oz + dz * t, rs, rc, bob);
                let not_hit: Mask = ge_tile(d, eps * t);
                let in_range: Mask = le_tile(t, tmax);
                active = select(not_hit, select(in_range, active, in_range), not_hit);
                t = select(active, t + d, t);
                steps = select(active, steps + one, steps);
                tile_steps = tile_steps + 1.0f32;
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

        // Sky: a vertical gradient plus a sun disc and halo.
        let lx: F = param(p, 16i32);
        let ly: F = param(p, 17i32);
        let lz: F = param(p, 18i32);
        let sky_t: F = clamp01(dy * fill(1.4f32) + fill(0.1f32));
        let sun: F = max_tile(dx * lx + dy * ly + dz * lz, zero);
        let s2: F = sun * sun;
        let s4: F = s2 * s2;
        let s8: F = s4 * s4;
        let s16: F = s8 * s8;
        let s64: F = s16 * s16 * s16 * s16;
        let s256: F = s64 * s64 * s64 * s64;
        let halo: F = s8 * fill(0.2f32);
        let disc: F = s256 * fill(1.5f32);
        let fog_r: F = fill(0.78f32) + halo;
        let fog_g: F = fill(0.80f32) + halo * fill(0.85f32);
        let fog_b: F = fill(0.84f32) + halo * fill(0.6f32);
        let mut r: F = mix(fog_r, fill(0.30f32), sky_t) + disc;
        let mut g: F = mix(fog_g, fill(0.46f32), sky_t) + disc * fill(0.9f32);
        let mut b: F = mix(fog_b, fill(0.78f32), sky_t) + disc * fill(0.7f32);

        // Surfaces. Tiles that are all sky skip this whole block, which is
        // most of the per-pixel cost (normals, shadow march, occlusion).
        let hit: Mask = le_tile(t, tmax);
        let any_hit: F = select(hit, one, zero);
        let any_hit: Tile<f32, { [32] }> = reduce_max(any_hit, 1i32);
        let any_hit: Tile<f32, { [] }> = reduce_max(any_hit, 0i32);
        let any_hit: f32 = tile_to_scalar(any_hit);
        if any_hit > 0.5f32 {
            let px: F = ox + dx * t;
            let py: F = oy + dy * t;
            let pz: F = oz + dz * t;
            let (nx, ny, nz) = normal(px, py, pz, rs, rc, bob);
            let m: F = material(px, py, pz, rs, rc, bob);

            // Checkerboard ground: floor(x) + floor(z), mod 2.
            let check: F = floor(px) + floor(pz);
            let check: F = check - two * floor(check * fill(0.5f32));
            let ground_r: F = mix(fill(0.26f32), fill(0.62f32), check);
            let ground_g: F = mix(fill(0.25f32), fill(0.58f32), check);
            let ground_b: F = mix(fill(0.24f32), fill(0.52f32), check);
            let ar: F = by_material(m, ground_r, 0.80f32, 0.90f32, 0.06f32, 0.85f32);
            let ag: F = by_material(m, ground_g, 0.76f32, 0.32f32, 0.45f32, 0.62f32);
            let ab: F = by_material(m, ground_b, 0.70f32, 0.18f32, 0.52f32, 0.18f32);
            let shine: F = by_material(m, fill(0.1f32), 0.15f32, 0.5f32, 0.9f32, 1.2f32);

            let off: F = fill(0.002f32);
            let shadow: F = soft_shadow(
                px + nx * off,
                py + ny * off,
                pz + nz * off,
                lx,
                ly,
                lz,
                rs,
                rc,
                bob,
                shadow_steps,
            );
            let ao: F = occlusion(px, py, pz, nx, ny, nz, rs, rc, bob);
            let diffuse: F = clamp01(nx * lx + ny * ly + nz * lz) * shadow;
            let ambient: F = (fill(0.5f32) + fill(0.5f32) * ny) * ao;
            let bounce: F = clamp01(fill(0.0f32) - ny) * ao * fill(0.3f32);

            // Blinn-Phong highlight: half vector between the light and the eye.
            let (hx, hy, hz) = normalize3(lx - dx, ly - dy, lz - dz);
            let nh: F = clamp01(nx * hx + ny * hy + nz * hz);
            let n2: F = nh * nh;
            let n4: F = n2 * n2;
            let n16: F = n4 * n4 * n4 * n4;
            let spec: F = n16 * n16 * n16 * n16 * shine * diffuse;

            let lit_r: F = ar * (fill(1.40f32) * diffuse + fill(0.35f32) * ambient + bounce) + spec;
            let lit_g: F = ag * (fill(1.25f32) * diffuse + fill(0.45f32) * ambient + bounce)
                + spec * fill(0.9f32);
            let lit_b: F = ab
                * (fill(1.00f32) * diffuse + fill(0.60f32) * ambient + bounce * fill(0.7f32))
                + spec * fill(0.75f32);

            // Distance fog toward the horizon color.
            let fog: F = one - exp(t * t * fill(-0.002f32));
            r = select(hit, mix(lit_r, fog_r, fog), r);
            g = select(hit, mix(lit_g, fog_g, fog), g);
            b = select(hit, mix(lit_b, fog_b, fog), b);
        }

        // Gamma 2 (sqrt), then optionally replace with a debug view.
        let r: F = sqrt(clamp01(r), rounding::NearestEven, ftz::Disabled);
        let g: F = sqrt(clamp01(g), rounding::NearestEven, ftz::Disabled);
        let b: F = sqrt(clamp01(b), rounding::NearestEven, ftz::Disabled);
        let view: F = param(p, 22i32);
        let inv_max: F = one / param(p, 23i32);
        let (sr, sg, sb) = heat(steps * inv_max);
        let (tr, tg, tb) = heat(fill(tile_steps) * inv_max);
        let show_steps: Mask = gt_tile(view, fill(0.5f32));
        let show_tiles: Mask = gt_tile(view, fill(1.5f32));
        let r: F = select(show_tiles, tr, select(show_steps, sr, r));
        let g: F = select(show_tiles, tg, select(show_steps, sg, g));
        let b: F = select(show_tiles, tb, select(show_steps, sb, b));

        let r: Tile<i32, { [32, 32] }> = to_byte(r);
        let g: Tile<i32, { [32, 32] }> = to_byte(g);
        let b: Tile<i32, { [32, 32] }> = to_byte(b);
        let shift_r: Tile<i32, { [32, 32] }> = broadcast_scalar(65536i32, const_shape![32, 32]);
        let shift_g: Tile<i32, { [32, 32] }> = broadcast_scalar(256i32, const_shape![32, 32]);
        out.store(r * shift_r + g * shift_g + b);
    }
}

/// March and shadow step budgets. Loop bounds are kernel scalars, so they
/// are baked into a captured graph; changing quality means recapturing.
#[derive(Clone, Copy, Debug)]
pub struct Quality {
    pub max_steps: i32,
    /// Steps between "is any ray in this tile still marching?" checks.
    pub check_every: i32,
    pub shadow_steps: i32,
}

impl Quality {
    pub const PRESETS: [Quality; 3] = [
        Quality {
            max_steps: 64,
            check_every: 16,
            shadow_steps: 16,
        },
        Quality {
            max_steps: 128,
            check_every: 16,
            shadow_steps: 32,
        },
        Quality {
            max_steps: 256,
            check_every: 16,
            shadow_steps: 64,
        },
    ];
}

/// Device buffers for one output size: the parameter block and the frame.
pub struct Renderer {
    stream: Arc<Stream>,
    pub quality: Quality,
    params: Tensor<f32>,
    frame: Tensor<i32>,
}

impl Renderer {
    pub fn new(
        stream: &Arc<Stream>,
        width: usize,
        height: usize,
        quality: Quality,
    ) -> Result<Self, Error> {
        Ok(Renderer {
            stream: stream.clone(),
            quality,
            params: api::zeros::<f32>(&[PARAMS]).sync_on(stream)?,
            frame: api::zeros::<i32>(&[height, width]).sync_on(stream)?,
        })
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Copy a new parameter block into the existing device buffer, so a
    /// captured graph sees it on its next launch.
    pub fn set_params(&mut self, params: &[f32; PARAMS]) -> Result<(), Error> {
        let src = api::copy_host_vec_to_device(&Arc::new(params.to_vec())).sync_on(&self.stream)?;
        api::memcpy(&mut self.params, &src).sync_on(&self.stream)?;
        Ok(())
    }

    fn op(&mut self) -> impl GraphNode + DeviceOp + '_ {
        let q = self.quality;
        kernels::render(
            (&mut self.frame).partition([TILE, TILE]),
            &self.params,
            q.max_steps,
            q.check_every,
            q.shadow_steps,
        )
    }

    pub fn render_eager(&mut self) -> Result<(), Error> {
        let stream = self.stream.clone();
        self.op().sync_on(&stream)?;
        Ok(())
    }

    /// Record one render as a CUDA graph. Replays read whatever is in the
    /// parameter buffer at launch time.
    pub fn capture(&mut self) -> Result<CudaGraph<()>, Error> {
        let stream = self.stream.clone();
        Ok(CudaGraph::scope(&stream, |s| {
            s.record(self.op())?;
            Ok(())
        })?)
    }

    /// Copy the frame to the host as 0x00RRGGBB pixels.
    pub fn download(&self) -> Result<Vec<u32>, Error> {
        let pixels = self.frame.dup().to_host_vec().sync_on(&self.stream)?;
        Ok(pixels.into_iter().map(|p| p as u32).collect())
    }
}
