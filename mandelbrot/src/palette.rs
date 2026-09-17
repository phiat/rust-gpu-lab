//! Smooth escape counts to RGB, done on the CPU.

use image::{Rgb, RgbImage};

/// Cosine gradient (a + b * cos(2π(c·t + d))), cycled along the escape count.
fn gradient(t: f32) -> [u8; 3] {
    const A: [f32; 3] = [0.5, 0.5, 0.5];
    const B: [f32; 3] = [0.5, 0.5, 0.5];
    const C: [f32; 3] = [1.0, 1.0, 1.0];
    const D: [f32; 3] = [0.00, 0.10, 0.20];
    let mut rgb = [0u8; 3];
    for i in 0..3 {
        let v = A[i] + B[i] * (std::f32::consts::TAU * (C[i] * t + D[i])).cos();
        rgb[i] = (v.clamp(0.0, 1.0) * 255.0) as u8;
    }
    rgb
}

pub fn to_image(pixels: &[f32], width: usize, height: usize) -> RgbImage {
    RgbImage::from_fn(width as u32, height as u32, |x, y| {
        let nu = pixels[y as usize * width + x as usize];
        if nu < 0.0 {
            Rgb([0, 0, 0])
        } else {
            // sqrt spreads the low counts (most of the image) across more colors.
            Rgb(gradient(nu.max(0.0).sqrt() * 0.12))
        }
    })
}
