//! CPU reference for every kernel, with the same operations in the same
//! order, so the GPU can be checked particle for particle.
//!
//! Buffers are flat `f32` slices, four per particle, in the GPU layout
//! (ghost ring included). Each function computes every buffer particle,
//! ghosts too, exactly as the kernels do.

// Mirrors the kernels expression for expression.
#![allow(clippy::assign_op_pattern)]

use rayon::prelude::*;

use crate::world::*;

fn get(buf: &[f32], i: usize) -> [f32; 4] {
    [buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]]
}

fn put(buf: &mut [f32], i: usize, v: [f32; 4]) {
    buf[4 * i..4 * i + 4].copy_from_slice(&v);
}

/// The neighbor at (r + dr, c + dc), or zeros outside the buffer. `valid`
/// says whether it is a real (non-ghost) particle.
fn neighbor(lay: &Layout, buf: &[f32], r: usize, c: usize, dr: i32, dc: i32) -> ([f32; 4], bool) {
    let (nr, nc) = (r as i32 + dr, c as i32 + dc);
    if nr < 0 || nc < 0 || nr as usize >= lay.buf_rows() || nc as usize >= lay.buf_cols() {
        return ([0.0; 4], false);
    }
    let (nr, nc) = (nr as usize, nc as usize);
    (get(buf, nr * lay.buf_cols() + nc), lay.is_world(nr, nc))
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Wind strength at a point and time: a steady part and two rolling waves.
pub fn gust(time: f32, x: f32, y: f32) -> f32 {
    0.55 + 0.3 * (time * 1.7 + x * 0.05 + y * 0.03).sin()
        + 0.15 * (time * 3.1 - y * 0.08 + x * 0.02).sin()
}

/// Verlet step: velocity from the previous position, gravity, and wind
/// pushing along the normal. Pinned particles (and ghosts) stay put.
pub fn integrate(
    lay: &Layout,
    pos: &[f32],
    prev: &[f32],
    nrm: &[f32],
    ph: &Physics,
    out: &mut [f32],
) {
    let bc = lay.buf_cols();
    let dt2 = ph.dt * ph.dt;
    out.par_chunks_mut(4 * bc)
        .enumerate()
        .for_each(|(r, line)| {
            for c in 0..bc {
                let i = r * bc + c;
                let [x, y, z, w] = get(pos, i);
                let [px, py, pz, _] = get(prev, i);
                let [nx, ny, nz, _] = get(nrm, i);
                let g = gust(ph.time, x, y);
                let wd = (nx * ph.wind[0] + ny * ph.wind[1] + nz * ph.wind[2]) * g;
                let ax = nx * wd;
                let ay = ph.gravity + ny * wd;
                let az = nz * wd;
                let vx = (x - px) * ph.damping;
                let vy = (y - py) * ph.damping;
                let vz = (z - pz) * ph.damping;
                let moved = [x + vx + ax * dt2, y + vy + ay * dt2, z + vz + az * dt2, w];
                put(line, c, if w > 0.0 { moved } else { [x, y, z, w] });
            }
        });
}

/// One constraint pass: every particle projects its one link of `batch`
/// (both ends move by their share of the mass), then is kept within reach
/// of its anchor, out of the sphere and above the floor.
pub fn pair(
    lay: &Layout,
    pos: &[f32],
    anchor: &[f32],
    ph: &Physics,
    batch: Batch,
    out: &mut [f32],
) {
    let bc = lay.buf_cols();
    let [sx, sy, sz, sr] = ph.sphere;
    let rest = batch.rest(ph.spacing);
    let k = ph.stiffness[batch.k];
    out.par_chunks_mut(4 * bc)
        .enumerate()
        .for_each(|(r, line)| {
            for c in 0..bc {
                let i = r * bc + c;
                let [mut x, mut y, mut z, w] = get(pos, i);
                let (dr, dc) = batch.partner(r, c);
                let ([qx, qy, qz, qw], valid) = neighbor(lay, pos, r, c, dr, dc);
                let (dx, dy, dz) = (qx - x, qy - y, qz - z);
                let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                let wsum = w + qw;
                let share = if wsum > 0.0 { w / wsum } else { 0.0 };
                let s = (len - rest) / len * share * k;
                if valid {
                    x = x + dx * s;
                    y = y + dy * s;
                    z = z + dz * s;
                }
                if w > 0.0 {
                    let [ax, ay, az, reach] = get(anchor, i);
                    let (vx, vy, vz) = (x - ax, y - ay, z - az);
                    let dist = (vx * vx + vy * vy + vz * vz).sqrt();
                    let pull = reach / dist.max(1e-6);
                    if reach > 0.0 && dist > reach {
                        x = ax + vx * pull;
                        y = ay + vy * pull;
                        z = az + vz * pull;
                    }
                    let (vx, vy, vz) = (x - sx, y - sy, z - sz);
                    let dist = (vx * vx + vy * vy + vz * vz).sqrt();
                    let push = sr / dist.max(1e-6);
                    if dist < sr {
                        x = sx + vx * push;
                        y = sy + vy * push;
                        z = sz + vz * push;
                    }
                    y = y.max(ph.floor);
                }
                put(line, c, [x, y, z, w]);
            }
        });
}

/// Unit normals from the left-right and up-down neighbors (a missing
/// neighbor counts as the particle itself). Ghosts get a zero normal.
pub fn normals(lay: &Layout, pos: &[f32]) -> Vec<f32> {
    let bc = lay.buf_cols();
    let mut out = vec![0.0f32; pos.len()];
    out.par_chunks_mut(4 * bc)
        .enumerate()
        .for_each(|(r, line)| {
            for c in 0..bc {
                let p = get(pos, r * bc + c);
                let pick = |dr: i32, dc: i32| {
                    let (q, valid) = neighbor(lay, pos, r, c, dr, dc);
                    if valid {
                        q
                    } else {
                        p
                    }
                };
                let (l, rt, u, d) = (pick(0, -1), pick(0, 1), pick(-1, 0), pick(1, 0));
                let tx = [rt[0] - l[0], rt[1] - l[1], rt[2] - l[2]];
                let ty = [d[0] - u[0], d[1] - u[1], d[2] - u[2]];
                let n = [
                    tx[1] * ty[2] - tx[2] * ty[1],
                    tx[2] * ty[0] - tx[0] * ty[2],
                    tx[0] * ty[1] - tx[1] * ty[0],
                ];
                let len = dot(n, n).sqrt().max(1e-9);
                put(line, c, [n[0] / len, n[1] / len, n[2] / len, 0.0]);
            }
        });
    out
}

/// Project and light every particle: screen x, screen y, depth, and the
/// color as a packed 0xRRGGBB (exact in an f32).
pub fn shade(lay: &Layout, pos: &[f32], nrm: &[f32], cam: &Camera) -> Vec<f32> {
    let bc = lay.buf_cols();
    let mut out = vec![0.0f32; pos.len()];
    let rot = cam.rotation();
    let light = cam.light;
    let (cx, cy) = (cam.width as f32 * 0.5, cam.height as f32 * 0.5);
    out.par_chunks_mut(4 * bc)
        .enumerate()
        .for_each(|(r, line)| {
            for c in 0..bc {
                let i = r * bc + c;
                let [x, y, z, _] = get(pos, i);
                let [nx, ny, nz, _] = get(nrm, i);
                let d = [x - cam.target[0], y - cam.target[1], z - cam.target[2]];
                let q = [
                    rot[0] * d[0] + rot[1] * d[1] + rot[2] * d[2],
                    rot[3] * d[0] + rot[4] * d[1] + rot[5] * d[2],
                    rot[6] * d[0] + rot[7] * d[1] + rot[8] * d[2] + cam.dist,
                ];
                let sx = cx + cam.focal * q[0] / q[2];
                let sy = cy - cam.focal * q[1] / q[2];
                let n = [
                    rot[0] * nx + rot[1] * ny + rot[2] * nz,
                    rot[3] * nx + rot[4] * ny + rot[5] * nz,
                    rot[6] * nx + rot[7] * ny + rot[8] * nz,
                ];
                let vl = dot(q, q).sqrt().max(1e-6);
                let v = [-q[0] / vl, -q[1] / vl, -q[2] / vl];
                let facing = dot(n, v);
                let n = if facing < 0.0 {
                    [-n[0], -n[1], -n[2]]
                } else {
                    n
                };
                let diff = dot(n, light).max(0.0);
                let h = [light[0] + v[0], light[1] + v[1], light[2] + v[2]];
                let hl = dot(h, h).sqrt().max(1e-6);
                let spec = (dot(n, h) / hl).max(0.0).powf(32.0);
                let tint = if facing < 0.0 { 0.7 } else { 1.0 };
                let base = base_color(
                    r.wrapping_sub(crate::gpu::TILE),
                    c.wrapping_sub(crate::gpu::TILE),
                );
                let mut rgb = 0.0f32;
                for (k, scale) in [65536.0f32, 256.0, 1.0].into_iter().enumerate() {
                    let v = (base[k] * tint * (0.2 + 0.8 * diff) + 0.35 * spec).clamp(0.0, 1.0);
                    rgb = rgb + (v * 255.0 + 0.5).floor() * scale;
                }
                put(line, c, [sx, sy, q[2], rgb]);
            }
        });
    out
}

/// Mean stretch of the structural links, as a fraction of the rest length.
pub fn stretch(lay: &Layout, pos: &[f32], spacing: f32) -> f32 {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for r in 0..lay.rows {
        for c in 0..lay.cols {
            let p = get(pos, lay.index(r, c));
            for (dr, dc) in [(0usize, 1usize), (1, 0)] {
                if r + dr < lay.rows && c + dc < lay.cols {
                    let q = get(pos, lay.index(r + dr, c + dc));
                    let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                    sum += ((dot(d, d).sqrt() - spacing) / spacing).abs() as f64;
                    n += 1;
                }
            }
        }
    }
    (sum / n.max(1) as f64) as f32
}

/// Host-side state of the CPU simulation, with the GPU's ping-pong
/// buffers so no pass allocates.
pub struct Cloth {
    pub pos: Vec<f32>,
    pub prev: Vec<f32>,
    pub nrm: Vec<f32>,
    pub anchor: Vec<f32>,
    scratch: Vec<f32>,
}

impl Cloth {
    pub fn new(lay: Layout, pins: Pins, spacing: f32) -> Self {
        let pos = initial(lay, pins, spacing);
        let nrm = normals(&lay, &pos);
        Cloth {
            scratch: pos.clone(),
            prev: pos.clone(),
            pos,
            nrm,
            anchor: anchors(lay, pins, spacing),
        }
    }

    /// The same sequence as `Pipeline::run`: substeps of integrate plus
    /// `iterations` x 12 batch passes, then normals. Parameters (time
    /// included) are fixed for the whole frame, as they are on the GPU.
    pub fn frame(&mut self, lay: &Layout, ph: &Physics, substeps: usize, iterations: usize) {
        for _ in 0..substeps {
            integrate(lay, &self.pos, &self.prev, &self.nrm, ph, &mut self.scratch);
            // The integrated positions become `pos`, the old ones `prev`.
            std::mem::swap(&mut self.prev, &mut self.scratch);
            std::mem::swap(&mut self.pos, &mut self.prev);
            for _ in 0..iterations {
                for batch in Batch::iteration() {
                    pair(lay, &self.pos, &self.anchor, ph, batch, &mut self.scratch);
                    std::mem::swap(&mut self.pos, &mut self.scratch);
                }
            }
        }
        self.nrm = normals(lay, &self.pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stretch must settle, not grow, with every constraint family on.
    #[test]
    fn stable() {
        let lay = Layout::new(32, 32);
        let mut ph = Physics::demo(lay);
        ph.sphere[3] = 0.0;
        for pins in [Pins::Top, Pins::Corners, Pins::Free] {
            let mut cloth = Cloth::new(lay, pins, ph.spacing);
            let mut line = String::new();
            let mut worst = 0.0f32;
            for f in 0..120 {
                cloth.frame(&lay, &ph, 4, 2);
                ph.time += ph.dt * 4.0;
                let s = stretch(&lay, &cloth.pos, ph.spacing);
                if f >= 60 {
                    worst = worst.max(s);
                }
                if f % 15 == 14 {
                    line += &format!(" {:.2}", s * 100.0);
                }
            }
            println!("{:<10} stretch %:{line}", pins.name());
            assert!(worst < 0.08, "{}: stretch {worst}", pins.name());
        }
    }
}
