//! CPU rasterizer for the shaded particle grid.
//!
//! The GPU projects and lights every particle (`kernels::shade`); turning
//! the grid of quads into pixels is a scatter (each triangle touches an
//! unpredictable set of pixels), which the tile model has no primitive for.
//! So the last step runs here: two triangles per grid cell, Gouraud
//! shaded, with a depth buffer, on rayon over horizontal bands.

use rayon::prelude::*;

use crate::world::{Camera, Layout};

pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u32>,
    depth: Vec<f32>,
    band: usize,
}

/// A screen-space triangle: vertices with depth and packed color.
struct Tri {
    x: [f32; 3],
    y: [f32; 3],
    z: [f32; 3],
    c: [u32; 3],
}

impl Frame {
    pub fn new(width: usize, height: usize) -> Self {
        let bands = rayon::current_num_threads() * 4;
        Frame {
            width,
            height,
            pixels: vec![0; width * height],
            depth: vec![f32::INFINITY; width * height],
            band: height.div_ceil(bands).max(1),
        }
    }

    /// Background gradient, empty depth.
    pub fn clear(&mut self) {
        let (w, h) = (self.width, self.height as f32);
        for (y, line) in self.pixels.chunks_exact_mut(w).enumerate() {
            let t = y as f32 / h;
            let r = (0x16 as f32 + 0x14 as f32 * t) as u32;
            let g = (0x18 as f32 + 0x14 as f32 * t) as u32;
            let b = (0x20 as f32 + 0x18 as f32 * t) as u32;
            line.fill((r << 16) | (g << 8) | b);
        }
        self.depth.fill(f32::INFINITY);
    }

    /// Draw the shaded particle grid (`screen`: x, y, depth, color per
    /// buffer particle).
    pub fn draw_cloth(&mut self, lay: &Layout, screen: &[f32]) {
        let v = |r: usize, c: usize| {
            let i = lay.index(r, c) * 4;
            (
                screen[i],
                screen[i + 1],
                screen[i + 2],
                screen[i + 3] as u32,
            )
        };
        let mut tris = Vec::with_capacity(2 * lay.rows * lay.cols);
        for r in 0..lay.rows - 1 {
            for c in 0..lay.cols - 1 {
                let (a, b, d, e) = (v(r, c), v(r, c + 1), v(r + 1, c), v(r + 1, c + 1));
                for (p, q, s) in [(a, b, d), (b, e, d)] {
                    tris.push(Tri {
                        x: [p.0, q.0, s.0],
                        y: [p.1, q.1, s.1],
                        z: [p.2, q.2, s.2],
                        c: [p.3, q.3, s.3],
                    });
                }
            }
        }
        self.draw_tris(&tris);
    }

    fn draw_tris(&mut self, tris: &[Tri]) {
        // Bucket triangles by the bands they touch, then fill bands in
        // parallel: every pixel is owned by one band, so no races.
        let (w, band) = (self.width, self.band);
        let n_bands = self.height.div_ceil(band);
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_bands];
        for (i, t) in tris.iter().enumerate() {
            let y0 =
                t.y.iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min)
                    .floor()
                    .max(0.0) as usize;
            let y1 =
                t.y.iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
                    .ceil()
                    .max(0.0) as usize;
            if y0 >= self.height || y1 == 0 {
                continue;
            }
            for bucket in &mut buckets[y0 / band..=(y1.min(self.height - 1)) / band] {
                bucket.push(i);
            }
        }
        self.pixels
            .par_chunks_mut(band * w)
            .zip(self.depth.par_chunks_mut(band * w))
            .enumerate()
            .for_each(|(b, (pixels, depth))| {
                let y_start = b * band;
                let rows = pixels.len() / w;
                for &i in &buckets[b] {
                    fill(&tris[i], pixels, depth, w, y_start, rows);
                }
            });
    }

    /// A lit sphere at a world position, depth tested against the cloth.
    pub fn draw_sphere(&mut self, cam: &Camera, center: [f32; 3], radius: f32, color: [f32; 3]) {
        let q = cam.camera_space(center);
        if q[2] <= radius {
            return;
        }
        let (cx, cy) = cam.project(q);
        let rr = cam.focal * radius / q[2];
        let (w, h) = (self.width as i32, self.height as i32);
        let (x0, x1) = (
            ((cx - rr).floor() as i32).max(0),
            ((cx + rr).ceil() as i32).min(w - 1),
        );
        let (y0, y1) = (
            ((cy - rr).floor() as i32).max(0),
            ((cy + rr).ceil() as i32).min(h - 1),
        );
        let l = cam.light;
        for py in y0..=y1 {
            for px in x0..=x1 {
                let nx = (px as f32 + 0.5 - cx) / rr;
                let ny = -(py as f32 + 0.5 - cy) / rr;
                let d2 = nx * nx + ny * ny;
                if d2 > 1.0 {
                    continue;
                }
                let nz = -(1.0 - d2).sqrt();
                let z = q[2] + radius * nz;
                let i = py as usize * self.width + px as usize;
                if z >= self.depth[i] {
                    continue;
                }
                let diff = (nx * l[0] + ny * l[1] + nz * l[2]).max(0.0);
                // Half vector with the view direction (0, 0, -1).
                let (hx, hy, hz) = (l[0], l[1], l[2] - 1.0);
                let hl = (hx * hx + hy * hy + hz * hz).sqrt();
                let spec = ((nx * hx + ny * hy + nz * hz) / hl).max(0.0).powf(48.0);
                let shade = |c: f32| {
                    ((c * (0.15 + 0.85 * diff) + 0.5 * spec).clamp(0.0, 1.0) * 255.0) as u32
                };
                self.pixels[i] = (shade(color[0]) << 16) | (shade(color[1]) << 8) | shade(color[2]);
                self.depth[i] = z;
            }
        }
    }
}

/// Fill one triangle into a band of rows starting at `y_start`.
fn fill(t: &Tri, pixels: &mut [u32], depth: &mut [f32], w: usize, y_start: usize, rows: usize) {
    let area = (t.x[1] - t.x[0]) * (t.y[2] - t.y[0]) - (t.x[2] - t.x[0]) * (t.y[1] - t.y[0]);
    if area.abs() < 1e-6 || !area.is_finite() {
        return;
    }
    let inv = 1.0 / area;
    let xmin =
        t.x.iter()
            .cloned()
            .fold(f32::INFINITY, f32::min)
            .floor()
            .max(0.0) as usize;
    let xmax = (t.x.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil() as usize).min(w - 1);
    let ymin =
        (t.y.iter()
            .cloned()
            .fold(f32::INFINITY, f32::min)
            .floor()
            .max(0.0) as usize)
            .max(y_start);
    let ymax = (t.y.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil() as usize)
        .min(y_start + rows - 1);
    if xmin > xmax || ymin > ymax {
        return;
    }
    let chan = |k: usize, shift: u32| {
        [
            ((t.c[0] >> shift) & 255) as f32,
            ((t.c[1] >> shift) & 255) as f32,
            ((t.c[2] >> shift) & 255) as f32,
        ][k]
    };
    for py in ymin..=ymax {
        let sy = py as f32 + 0.5;
        for px in xmin..=xmax {
            let sx = px as f32 + 0.5;
            let w0 = ((t.x[1] - sx) * (t.y[2] - sy) - (t.x[2] - sx) * (t.y[1] - sy)) * inv;
            let w1 = ((t.x[2] - sx) * (t.y[0] - sy) - (t.x[0] - sx) * (t.y[2] - sy)) * inv;
            let w2 = 1.0 - w0 - w1;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            let z = w0 * t.z[0] + w1 * t.z[1] + w2 * t.z[2];
            let i = (py - y_start) * w + px;
            if z < depth[i] {
                depth[i] = z;
                let mix = |shift: u32| {
                    (w0 * chan(0, shift) + w1 * chan(1, shift) + w2 * chan(2, shift))
                        .clamp(0.0, 255.0) as u32
                };
                pixels[i] = (mix(16) << 16) | (mix(8) << 8) | mix(0);
            }
        }
    }
}
