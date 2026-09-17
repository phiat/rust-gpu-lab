//! CPU references: jump flooding with the kernel's exact integer math, and
//! an exact Euclidean distance transform to measure how far JFA is off.

// The math below mirrors the GPU kernel expression for expression (same
// operation order, same `min(max(..))` clamps, same literals), so that the
// two can be compared. Clippy's tidier spellings would hide that.
#![allow(
    clippy::assign_op_pattern,
    clippy::manual_clamp,
    clippy::excessive_precision
)]

use rayon::prelude::*;

use crate::gpu::TILE;
use crate::Layout;

const BIAS: i32 = 8192;

pub fn encode(row: i32, col: i32) -> i32 {
    (row + BIAS) * 32768 + col + BIAS
}

pub fn decode(seed: i32) -> (i32, i32) {
    ((seed >> 15) - BIAS, (seed & 32767) - BIAS)
}

pub fn dist2(seed: i32, row: i32, col: i32) -> i32 {
    let (sr, sc) = decode(seed);
    (sr - row) * (sr - row) + (sc - col) * (sc - col)
}

pub fn seed_init(lay: &Layout, scene: &[i32]) -> Vec<i32> {
    let cols = lay.buf_cols();
    let mut out = vec![0; scene.len()];
    out.par_chunks_mut(cols).enumerate().for_each(|(r, line)| {
        for (c, px) in line.iter_mut().enumerate() {
            if scene[r * cols + c] != 0 {
                *px = encode(r as i32, c as i32);
            }
        }
    });
    out
}

/// One pass: the 9 candidates at (-k, 0, +k)^2, center first, then row by
/// row as in `jfa_step`. Out-of-buffer candidates are "no seed" (0), and the
/// ghost ring stays 0.
pub fn jfa_step(lay: &Layout, src: &[i32], k: usize) -> Vec<i32> {
    let (rows, cols) = (lay.buf_rows() as i32, lay.buf_cols() as i32);
    let k = k as i32;
    let mut out = vec![0; src.len()];
    out.par_chunks_mut(cols as usize)
        .enumerate()
        .skip(TILE)
        .for_each(|(r, line)| {
            let r = r as i32;
            for c in TILE as i32..cols {
                let at = |dr: i32, dc: i32| {
                    let (rr, cc) = (r + dr, c + dc);
                    if (0..rows).contains(&rr) && (0..cols).contains(&cc) {
                        src[(rr * cols + cc) as usize]
                    } else {
                        0
                    }
                };
                let mut best = at(0, 0);
                let mut best_d = dist2(best, r, c);
                for (dr, dc) in [
                    (-k, -k),
                    (-k, 0),
                    (-k, k),
                    (0, -k),
                    (0, k),
                    (k, -k),
                    (k, 0),
                    (k, k),
                ] {
                    let cand = at(dr, dc);
                    let d = dist2(cand, r, c);
                    if d < best_d {
                        (best, best_d) = (cand, d);
                    }
                }
                line[c as usize] = best;
            }
        });
    out
}

pub fn jump_flood(lay: &Layout, scene: &[i32]) -> Vec<i32> {
    let mut seeds = seed_init(lay, scene);
    for k in lay.jfa_offsets() {
        seeds = jfa_step(lay, &seeds, k);
    }
    seeds
}

pub fn distance(lay: &Layout, seeds: &[i32]) -> Vec<f32> {
    let cols = lay.buf_cols();
    (0..seeds.len())
        .into_par_iter()
        .map(|i| (dist2(seeds[i], (i / cols) as i32, (i % cols) as i32) as f32).sqrt())
        .collect()
}

pub fn nearest(seeds: &[i32], scene: &[i32], cols: usize) -> Vec<i32> {
    seeds
        .par_iter()
        .map(|&s| {
            if s == 0 {
                0
            } else {
                let (r, c) = decode(s);
                scene[r as usize * cols + c as usize]
            }
        })
        .collect()
}

/// Exact squared distance to the nearest occupied pixel for every pixel
/// (Felzenszwalb & Huttenlocher: a 1-D lower envelope of parabolas per
/// column, then per row). `f64::INFINITY` when nothing is occupied.
pub fn exact_dist2(lay: &Layout, scene: &[i32]) -> Vec<f64> {
    let (rows, cols) = (lay.buf_rows(), lay.buf_cols());
    let mut grid: Vec<f64> = scene
        .iter()
        .map(|&s| if s != 0 { 0.0 } else { f64::INFINITY })
        .collect();
    // Columns.
    let mut columns: Vec<Vec<f64>> = (0..cols)
        .map(|c| (0..rows).map(|r| grid[r * cols + c]).collect())
        .collect();
    columns.par_iter_mut().for_each(|col| *col = edt_1d(col));
    for (c, col) in columns.iter().enumerate() {
        for r in 0..rows {
            grid[r * cols + c] = col[r];
        }
    }
    // Rows.
    grid.par_chunks_mut(cols).for_each(|row| {
        let done = edt_1d(row);
        row.copy_from_slice(&done);
    });
    grid
}

fn edt_1d(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let mut out = vec![f64::INFINITY; n];
    let mut v = vec![0usize; n]; // parabola vertices in the envelope
    let mut z = vec![0f64; n + 1]; // boundaries between them
    let mut k = 0usize;
    let first = match f.iter().position(|x| x.is_finite()) {
        Some(i) => i,
        None => return out,
    };
    v[0] = first;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let sq = |x: usize| (x * x) as f64;
    for q in first + 1..n {
        if !f[q].is_finite() {
            continue;
        }
        loop {
            let p = v[k];
            let s = ((f[q] + sq(q)) - (f[p] + sq(p))) / (2.0 * (q as f64 - p as f64));
            if s <= z[k] {
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
        }
    }
    let mut k = 0;
    for (q, o) in out.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        *o = (q as f64 - p as f64).powi(2) + f[p];
    }
    out
}

fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// One frame of the radiance kernel with no history (blend 1), in the
/// kernel's operation order. `params` as in `kernels::radiance`.
pub fn radiance(
    lay: &Layout,
    dist: &[f32],
    scene: &[i32],
    params: &[f32; crate::gpu::PARAMS],
    q: crate::gpu::Quality,
) -> Vec<[f32; 3]> {
    let (rows, cols) = (lay.buf_rows() as i32, lay.buf_cols() as i32);
    let (x_max, y_max) = (cols as f32, rows as f32);
    let (frame, intensity, rays_f, ambient) = (params[0], params[2], params[4], params[6]);
    let byte = |v: i32, shift: i32| ((v >> shift) & 255) as f32 * (1.0f32 / 255.0f32);
    (0..(rows * cols) as usize)
        .into_par_iter()
        .map(|i| {
            let y = (i as i32 / cols) as f32 + 0.5;
            let x = (i as i32 % cols) as f32 + 0.5;
            let noise = fract(x * 0.06711056 + y * 0.00583715);
            let noise = fract(noise * 52.982918);
            let jitter = fract(noise + frame * 0.618034);
            let sector = 6.2831855f32 / rays_f;
            let mut sum = [0f32; 3];
            let mut ray = 0f32;
            for _ in 0..q.rays {
                let angle = (ray + jitter) * sector;
                ray += 1.0;
                let (dx, dy) = (angle.cos(), angle.sin());
                let (mut t, mut hit, mut idx) = (0f32, false, 0usize);
                for _ in 0..q.steps {
                    let (px, py) = (x + dx * t, y + dy * t);
                    let inside = px >= 0.0 && px < x_max && py >= 0.0 && py < y_max;
                    let col = (px.floor() as i32).clamp(0, cols - 1);
                    let row = (py.floor() as i32).clamp(0, rows - 1);
                    let here = (row * cols + col) as usize;
                    let d = dist[here];
                    let landed = d < 0.5;
                    if inside && landed {
                        (hit, idx) = (true, here);
                    }
                    if !inside || landed {
                        break;
                    }
                    t = t + (d - 1.0).max(1.0);
                }
                let v = scene[idx];
                if hit && v >> 24 == 2 {
                    sum[0] = sum[0] + byte(v, 16);
                    sum[1] = sum[1] + byte(v, 8);
                    sum[2] = sum[2] + byte(v, 0);
                }
                if !hit {
                    sum[0] = sum[0] + ambient * 0.55;
                    sum[1] = sum[1] + ambient * 0.65;
                    sum[2] = sum[2] + ambient;
                }
            }
            let scale = intensity / rays_f;
            sum.map(|s| s * scale)
        })
        .collect()
}
