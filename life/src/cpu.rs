//! CPU reference: the same toroidal Life, one row per rayon task.

use rayon::prelude::*;

pub fn step(cells: &[u8], next: &mut [u8], rows: usize, cols: usize) {
    next.par_chunks_mut(cols).enumerate().for_each(|(r, out)| {
        let up = &cells[((r + rows - 1) % rows) * cols..][..cols];
        let mid = &cells[r * cols..][..cols];
        let down = &cells[((r + 1) % rows) * cols..][..cols];
        for (c, px) in out.iter_mut().enumerate() {
            let (l, rt) = ((c + cols - 1) % cols, (c + 1) % cols);
            let sum =
                up[l] + up[c] + up[rt] + mid[l] + mid[c] + mid[rt] + down[l] + down[c] + down[rt];
            *px = (sum == 3 || (sum == 4 && mid[c] == 1)) as u8;
        }
    });
}

/// Random soup with roughly `density` of cells alive (xorshift64*, seeded).
pub fn random_soup(rows: usize, cols: usize, density: f64, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    let threshold = (density.clamp(0.0, 1.0) * u32::MAX as f64) as u32;
    (0..rows * cols)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let r = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32;
            (r < threshold) as u8
        })
        .collect()
}
