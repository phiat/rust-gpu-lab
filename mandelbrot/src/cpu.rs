//! CPU reference renderer. Same f32 math as the GPU kernels, written the
//! ordinary way: a per-pixel loop that breaks as soon as the point escapes.

use rayon::prelude::*;

use crate::{View, BAILOUT_SQ};

/// Smooth escape count for one point, or -1 if it never escaped.
#[inline]
pub fn escape(cr: f32, ci: f32, max_iter: u32) -> f32 {
    let (mut zr, mut zi) = (0.0f32, 0.0f32);
    let mut n = 0u32;
    while n < max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if zr2 + zi2 > BAILOUT_SQ {
            break;
        }
        zi = (zr + zr) * zi + ci;
        zr = zr2 - zi2 + cr;
        n += 1;
    }
    let mag2 = zr * zr + zi * zi;
    if mag2 > BAILOUT_SQ {
        n as f32 + 1.0 - (0.5 * mag2.ln()).log2()
    } else {
        -1.0
    }
}

fn render_row(view: &View, row: usize, out: &mut [f32]) {
    let (x0, y0, step) = view.pixel_mapping();
    let ci = y0 - row as f32 * step;
    for (col, px) in out.iter_mut().enumerate() {
        let cr = col as f32 * step + x0;
        *px = escape(cr, ci, view.max_iter);
    }
}

pub fn render_serial(view: &View) -> Vec<f32> {
    let mut pixels = vec![0.0f32; view.width * view.height];
    for (row, line) in pixels.chunks_mut(view.width).enumerate() {
        render_row(view, row, line);
    }
    pixels
}

pub fn render_rayon(view: &View) -> Vec<f32> {
    let mut pixels = vec![0.0f32; view.width * view.height];
    pixels
        .par_chunks_mut(view.width)
        .enumerate()
        .for_each(|(row, line)| render_row(view, row, line));
    pixels
}
