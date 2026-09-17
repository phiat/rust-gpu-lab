//! CPU reference: the same integer math and "valid" convolutions as the GPU
//! chain, so results must match bit for bit.

use rayon::prelude::*;

use crate::Dims;

pub fn run(dims: &Dims, padded_rgba: &[u8]) -> Vec<u8> {
    let (mut h, mut w) = (dims.padded_rows(), dims.padded_cols());

    let mut img = vec![0u8; h * w];
    img.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
        for (c, px) in row.iter_mut().enumerate() {
            let p = &padded_rgba[(r * w + c) * 4..][..4];
            *px = ((77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32 + 128) >> 8) as u8;
        }
    });

    for _ in 0..dims.blur_passes {
        let mut mid = vec![0u16; h * (w - 4)];
        mid.par_chunks_mut(w - 4).enumerate().for_each(|(r, row)| {
            let x = &img[r * w..][..w];
            for (c, px) in row.iter_mut().enumerate() {
                let s = x[c] as u32
                    + x[c + 4] as u32
                    + 4 * (x[c + 1] as u32 + x[c + 3] as u32)
                    + 6 * x[c + 2] as u32;
                *px = s as u16;
            }
        });
        let w2 = w - 4;
        let mut out = vec![0u8; (h - 4) * w2];
        out.par_chunks_mut(w2).enumerate().for_each(|(r, row)| {
            let x = |dr: usize, c: usize| mid[(r + dr) * w2 + c] as u32;
            for (c, px) in row.iter_mut().enumerate() {
                let s = x(0, c) + x(4, c) + 4 * (x(1, c) + x(3, c)) + 6 * x(2, c);
                *px = ((s + 128) >> 8) as u8;
            }
        });
        (img, h, w) = (out, h - 4, w2);
    }

    let mut edges = vec![0u8; (h - 2) * (w - 2)];
    edges
        .par_chunks_mut(w - 2)
        .enumerate()
        .for_each(|(r, row)| {
            let x = |dr: usize, dc: usize, c: usize| img[(r + dr) * w + c + dc] as i32;
            for (c, px) in row.iter_mut().enumerate() {
                let gx = (x(0, 2, c) + 2 * x(1, 2, c) + x(2, 2, c))
                    - (x(0, 0, c) + 2 * x(1, 0, c) + x(2, 0, c));
                let gy = (x(2, 0, c) + 2 * x(2, 1, c) + x(2, 2, c))
                    - (x(0, 0, c) + 2 * x(0, 1, c) + x(0, 2, c));
                *px = (gx.abs() + gy.abs()).min(255) as u8;
            }
        });
    edges
}
