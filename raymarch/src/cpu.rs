//! CPU reference: the same scene, march and shading as the GPU kernel,
//! one pixel at a time with rayon across rows. Only the shaded view.
//!
//! Operations are kept in the kernel's order, but results are not bit
//! exact: the GPU compiler may fuse `a * b + c` into one fused multiply-add,
//! and its `exp` differs from libm in the last bits. Sphere tracing
//! amplifies those differences near silhouettes.

use rayon::prelude::*;

use crate::gpu::Quality;
use crate::scene::{slot, PARAMS};

#[derive(Clone, Copy)]
struct Anim {
    rs: f32,
    rc: f32,
    bob: f32,
}

fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn length2(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

fn length3(x: f32, y: f32, z: f32) -> f32 {
    (x * x + y * y + z * z).sqrt()
}

fn normalize3(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let len = length3(x, y, z);
    (x / len, y / len, z / len)
}

fn sd_round_box(x: f32, y: f32, z: f32, half: f32, radius: f32) -> f32 {
    let inner = half - radius;
    let (qx, qy, qz) = (x.abs() - inner, y.abs() - inner, z.abs() - inner);
    let outside = length3(qx.max(0.0), qy.max(0.0), qz.max(0.0));
    let inside = qx.max(qy.max(qz)).min(0.0);
    outside + inside - radius
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp01(0.5 + (b - a) * (0.5 / k));
    mix(b, a, h) - k * h * (1.0 - h)
}

fn fold(x: f32, z: f32, cos: f32, sin: f32) -> (f32, f32) {
    let d = (z * cos - x * sin).max(0.0);
    let d2 = d * 2.0;
    (x + d2 * sin, z - d2 * cos)
}

fn sd_column(x: f32, y: f32, z: f32, cx: f32, cz: f32) -> f32 {
    (length2(x - cx, z - cz) - 0.24).max(y - 2.6)
}

fn sd_pillars(x: f32, y: f32, z: f32) -> f32 {
    let (ax, az) = (x.abs(), z.abs());
    let (x45, z45) = (ax.max(az), ax.min(az));
    let (x22, z22) = fold(x45, z45, 0.92387953, 0.38268343);
    let inner = sd_column(x22, y, z22, 6.3751044, 1.2680871);
    let (x11, z11) = fold(x22, z22, 0.98078528, 0.19509032);
    let outer = sd_column(x11, y, z11, 10.449440, 1.0291800);
    inner.min(outer)
}

fn sd_blob(x: f32, y: f32, z: f32, a: Anim) -> f32 {
    let ball = length3(x, y - a.bob, z) - 0.5;
    let bx = a.rc * x + a.rs * z;
    let bz = a.rc * z - a.rs * x;
    let cube = sd_round_box(bx, y - 0.55, bz, 0.45, 0.08);
    smin(ball, cube, 0.35)
}

fn sd_ring(x: f32, y: f32, z: f32, a: Anim) -> f32 {
    let (lx, ly, lz) = (x - 1.9, y - 0.75, z + 0.9);
    let tx = a.rc * lx - a.rs * lz;
    let tz = a.rs * lx + a.rc * lz;
    length2(length2(tx, ly) - 0.55, tz) - 0.2
}

fn sd_ball(x: f32, y: f32, z: f32) -> f32 {
    length3(x + 1.7, y - 0.5, z - 1.0) - 0.5
}

fn scene(x: f32, y: f32, z: f32, a: Anim) -> f32 {
    y.min(sd_pillars(x, y, z))
        .min(sd_blob(x, y, z, a))
        .min(sd_ring(x, y, z, a))
        .min(sd_ball(x, y, z))
}

fn material(x: f32, y: f32, z: f32, a: Anim) -> usize {
    let (mut d, mut m) = (y, 0);
    for (id, dist) in [
        (1, sd_pillars(x, y, z)),
        (2, sd_blob(x, y, z, a)),
        (3, sd_ring(x, y, z, a)),
        (4, sd_ball(x, y, z)),
    ] {
        if dist < d {
            (d, m) = (dist, id);
        }
    }
    m
}

fn normal(x: f32, y: f32, z: f32, a: Anim) -> (f32, f32, f32) {
    let e = 0.0005f32;
    let (mut nx, mut ny, mut nz) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..4 {
        let kx: f32 = if i == 0 || i == 3 { 1.0 } else { -1.0 };
        let ky: f32 = if i >= 2 { 1.0 } else { -1.0 };
        let kz: f32 = if i == 1 || i == 3 { 1.0 } else { -1.0 };
        let d = scene(x + kx * e, y + ky * e, z + kz * e, a);
        nx = nx + kx * d;
        ny = ny + ky * d;
        nz = nz + kz * d;
    }
    normalize3(nx, ny, nz)
}

#[allow(clippy::too_many_arguments)]
fn soft_shadow(x: f32, y: f32, z: f32, lx: f32, ly: f32, lz: f32, a: Anim, steps: i32) -> f32 {
    let (mut t, mut res, mut prev) = (0.02f32, 1.0f32, 1.0e10f32);
    for _ in 0..steps {
        let h = scene(x + lx * t, y + ly * t, z + lz * t, a);
        let back = h * h / (2.0 * prev);
        let across = (h * h - back * back).max(0.0).sqrt();
        let est = 10.0 * across / (t - back).max(0.0001);
        if t < 12.0 {
            res = res.min(est);
        }
        prev = h;
        t = t + h.max(0.01).min(0.25);
    }
    let res = clamp01(res);
    res * res * (3.0 - 2.0 * res)
}

#[allow(clippy::too_many_arguments)]
fn occlusion(x: f32, y: f32, z: f32, nx: f32, ny: f32, nz: f32, a: Anim) -> f32 {
    let (mut occ, mut weight, mut h) = (0.0f32, 1.0f32, 0.01f32);
    for _ in 0..5 {
        let d = scene(x + nx * h, y + ny * h, z + nz * h, a);
        occ = occ + (h - d) * weight;
        weight = weight * 0.95;
        h = h + 0.03;
    }
    clamp01(1.0 - 3.0 * occ)
}

fn to_byte(c: f32) -> u32 {
    (c * 255.0 + 0.5) as i32 as u32
}

fn pixel(p: &[f32; PARAMS], row: usize, col: usize, q: Quality) -> u32 {
    let (row, col) = (row as f32 + 0.5, col as f32 + 0.5);
    let (width, height) = (p[slot::WIDTH], p[slot::HEIGHT]);
    let u = (col * 2.0 - width) / height;
    let v = (height - row * 2.0) / height;
    let (f, r, up, focal) = (slot::FORWARD, slot::RIGHT, slot::UP, p[slot::FOCAL]);
    let (dx, dy, dz) = normalize3(
        p[f] * focal + p[r] * u + p[up] * v,
        p[f + 1] * focal + p[r + 1] * u + p[up + 1] * v,
        p[f + 2] * focal + p[r + 2] * u + p[up + 2] * v,
    );
    let (ox, oy, oz) = (p[slot::EYE], p[slot::EYE + 1], p[slot::EYE + 2]);
    let a = Anim {
        rs: p[slot::SPIN_SIN],
        rc: p[slot::SPIN_COS],
        bob: p[slot::BOB],
    };

    let (eps, tmax) = (0.0004f32, 60.0f32);
    let mut t = 0.05f32;
    for _ in 0..q.max_steps {
        let d = scene(ox + dx * t, oy + dy * t, oz + dz * t, a);
        if d < eps * t || t > tmax {
            break;
        }
        t = t + d;
    }

    let (lx, ly, lz) = (p[slot::SUN], p[slot::SUN + 1], p[slot::SUN + 2]);
    let sky_t = clamp01(dy * 1.4 + 0.1);
    let sun = (dx * lx + dy * ly + dz * lz).max(0.0);
    let s2 = sun * sun;
    let s4 = s2 * s2;
    let s8 = s4 * s4;
    let s16 = s8 * s8;
    let s64 = s16 * s16 * s16 * s16;
    let s256 = s64 * s64 * s64 * s64;
    let (halo, disc) = (s8 * 0.2, s256 * 1.5);
    let (fog_r, fog_g, fog_b) = (0.78 + halo, 0.80 + halo * 0.85, 0.84 + halo * 0.6);
    let mut r = mix(fog_r, 0.30, sky_t) + disc;
    let mut g = mix(fog_g, 0.46, sky_t) + disc * 0.9;
    let mut b = mix(fog_b, 0.78, sky_t) + disc * 0.7;

    if t <= tmax {
        let (px, py, pz) = (ox + dx * t, oy + dy * t, oz + dz * t);
        let (nx, ny, nz) = normal(px, py, pz, a);
        let check = px.floor() + pz.floor();
        let check = check - 2.0 * (check * 0.5).floor();
        let (ar, ag, ab, shine) = match material(px, py, pz, a) {
            0 => (
                mix(0.26, 0.62, check),
                mix(0.25, 0.58, check),
                mix(0.24, 0.52, check),
                0.1,
            ),
            1 => (0.80, 0.76, 0.70, 0.15),
            2 => (0.90, 0.32, 0.18, 0.5),
            3 => (0.06, 0.45, 0.52, 0.9),
            _ => (0.85, 0.62, 0.18, 1.2),
        };

        let off = 0.002f32;
        let shadow = soft_shadow(
            px + nx * off,
            py + ny * off,
            pz + nz * off,
            lx,
            ly,
            lz,
            a,
            q.shadow_steps,
        );
        let ao = occlusion(px, py, pz, nx, ny, nz, a);
        let diffuse = clamp01(nx * lx + ny * ly + nz * lz) * shadow;
        let ambient = (0.5 + 0.5 * ny) * ao;
        let bounce = clamp01(0.0 - ny) * ao * 0.3;

        let (hx, hy, hz) = normalize3(lx - dx, ly - dy, lz - dz);
        let nh = clamp01(nx * hx + ny * hy + nz * hz);
        let n2 = nh * nh;
        let n4 = n2 * n2;
        let n16 = n4 * n4 * n4 * n4;
        let spec = n16 * n16 * n16 * n16 * shine * diffuse;

        let lit_r = ar * (1.40 * diffuse + 0.35 * ambient + bounce) + spec;
        let lit_g = ag * (1.25 * diffuse + 0.45 * ambient + bounce) + spec * 0.9;
        let lit_b = ab * (1.00 * diffuse + 0.60 * ambient + bounce * 0.7) + spec * 0.75;

        let fog = 1.0 - (t * t * -0.002).exp();
        r = mix(lit_r, fog_r, fog);
        g = mix(lit_g, fog_g, fog);
        b = mix(lit_b, fog_b, fog);
    }

    let (r, g, b) = (clamp01(r).sqrt(), clamp01(g).sqrt(), clamp01(b).sqrt());
    to_byte(r) * 65536 + to_byte(g) * 256 + to_byte(b)
}

pub fn render(params: &[f32; PARAMS], width: usize, height: usize, q: Quality) -> Vec<u32> {
    let mut out = vec![0u32; width * height];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, line)| {
            for (col, px) in line.iter_mut().enumerate() {
                *px = pixel(params, row, col, q);
            }
        });
    out
}
