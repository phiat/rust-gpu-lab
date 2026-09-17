mod cpu;
mod gpu;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use cutile::prelude::*;
use image::{GrayImage, RgbImage};

use gpu::Pipeline;

#[derive(Parser)]
#[command(about = "Image filter chain on the GPU: grayscale -> Gaussian blur -> Sobel edges")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Filter one image and write every stage as a PNG.
    Run {
        #[command(flatten)]
        src: Source,
        #[arg(long, default_value = "filters-out")]
        out_dir: PathBuf,
    },
    /// Process a stream of frames: CPU vs GPU eager vs GPU CUDA graph.
    Bench {
        #[command(flatten)]
        src: Source,
        #[arg(long, default_value_t = 30)]
        frames: usize,
    },
}

#[derive(Args)]
struct Source {
    /// PNG or JPEG to filter. Without it, a synthetic noisy test image is generated.
    #[arg(long)]
    input: Option<PathBuf>,
    /// Size of the synthetic image.
    #[arg(long, default_value_t = 3840)]
    width: usize,
    #[arg(long, default_value_t = 2160)]
    height: usize,
    /// 5x5 Gaussian passes before Sobel; each widens the blur.
    #[arg(long, default_value_t = 2)]
    blur_passes: usize,
    #[arg(long, default_value_t = 32)]
    tile: usize,
}

/// Image size and chain depth. Every stencil stage shrinks the image by
/// twice its radius, so the source is padded by the total radius up front.
#[derive(Clone, Copy)]
pub struct Dims {
    pub rows: usize,
    pub cols: usize,
    pub blur_passes: usize,
}

impl Dims {
    /// 2 per 5x5 blur pass, plus 1 for Sobel.
    pub fn radius(&self) -> usize {
        2 * self.blur_passes + 1
    }

    pub fn padded_rows(&self) -> usize {
        self.rows + 2 * self.radius()
    }

    pub fn padded_cols(&self) -> usize {
        self.cols + 2 * self.radius()
    }

    /// Pad an RGB image by repeating its edge pixels (clamp to edge), as
    /// RGBA with alpha 255.
    fn pad_rgba(&self, rgb: &[u8]) -> Vec<u8> {
        let (pad, pc) = (self.radius(), self.padded_cols());
        let mut out = vec![255u8; self.padded_rows() * pc * 4];
        for r in 0..self.padded_rows() {
            let sr = r.saturating_sub(pad).min(self.rows - 1);
            for c in 0..pc {
                let sc = c.saturating_sub(pad).min(self.cols - 1);
                let (dst, src) = ((r * pc + c) * 4, (sr * self.cols + sc) * 3);
                out[dst..dst + 3].copy_from_slice(&rgb[src..src + 3]);
            }
        }
        out
    }
}

fn load_source(src: &Source) -> Result<(Dims, RgbImage, bool), Box<dyn std::error::Error>> {
    let (img, synthetic) = match &src.input {
        Some(path) => (image::open(path)?.to_rgb8(), false),
        None => (
            synthetic_image(src.width as u32, src.height as u32, 7),
            true,
        ),
    };
    let dims = Dims {
        rows: img.height() as usize,
        cols: img.width() as usize,
        blur_passes: src.blur_passes,
    };
    Ok((dims, img, synthetic))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().cmd {
        Cmd::Run { src, out_dir } => run(&src, &out_dir),
        Cmd::Bench { src, frames } => bench(&src, frames),
    }
}

fn run(src: &Source, out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let (dims, img, synthetic) = load_source(src)?;
    let padded = dims.pad_rgba(img.as_raw());
    std::fs::create_dir_all(out_dir)?;
    if synthetic {
        img.save(out_dir.join("input.png"))?;
    }

    let stream = Device::new(0)?.new_stream()?;
    let mut p = Pipeline::new(&stream, dims, src.tile)?;
    p.upload(padded.clone())?;
    let t = Instant::now();
    p.run_eager()?;
    let first = t.elapsed();
    let t = Instant::now();
    p.run_eager()?;
    let warm = t.elapsed();

    let edges = p.download_edges()?;
    let (gray, blurred) = p.download_stages()?;
    let (w, h) = (dims.cols as u32, dims.rows as u32);
    GrayImage::from_raw(w, h, gray)
        .unwrap()
        .save(out_dir.join("gray.png"))?;
    GrayImage::from_raw(w, h, blurred)
        .unwrap()
        .save(out_dir.join("blurred.png"))?;
    GrayImage::from_raw(w, h, edges.clone())
        .unwrap()
        .save(out_dir.join("edges.png"))?;

    println!(
        "{w}x{h}, {} blur passes, tile {}: first run {} (JIT), warm {}",
        dims.blur_passes,
        src.tile,
        ms(first),
        ms(warm)
    );
    println!("matches cpu: {}", edges == cpu::run(&dims, &padded));
    println!("wrote {}/{{gray,blurred,edges}}.png", out_dir.display());
    Ok(())
}

fn bench(src: &Source, frames: usize) -> Result<(), Box<dyn std::error::Error>> {
    let (dims, img, _) = load_source(src)?;
    let padded = dims.pad_rgba(img.as_raw());
    println!(
        "{}x{} ({:.1} MP), {} blur passes, tile {}, {} frames",
        dims.cols,
        dims.rows,
        (dims.rows * dims.cols) as f64 / 1e6,
        dims.blur_passes,
        src.tile,
        frames
    );

    let t = Instant::now();
    let mut reference = Vec::new();
    for _ in 0..frames {
        reference = cpu::run(&dims, &padded);
    }
    let cpu_frame = t.elapsed() / frames as u32;
    println!(
        "cpu rayon ({:>2} threads)   filter {:>10}",
        rayon::current_num_threads(),
        ms(cpu_frame)
    );

    let stream = Device::new(0)?.new_stream()?;
    let mut p = Pipeline::new(&stream, dims, src.tile)?;
    p.upload(padded.clone())?;
    p.run_eager()?; // JIT compile every stage

    // Eager: each stage is launched and synchronized in turn.
    let mut times = Timings::default();
    let mut edges = Vec::new();
    for _ in 0..frames {
        let t = Instant::now();
        p.upload(padded.clone())?;
        times.upload += t.elapsed();
        let t = Instant::now();
        p.run_eager()?;
        times.filter += t.elapsed();
        let t = Instant::now();
        edges = p.download_edges()?;
        times.download += t.elapsed();
    }
    times.report("gpu eager", frames, cpu_frame);
    println!("    matches cpu: {}", edges == reference);

    // Graph: the whole chain replayed with one launch per frame.
    let graph = p.capture()?;
    let mut times = Timings::default();
    for _ in 0..frames {
        let t = Instant::now();
        p.upload(padded.clone())?;
        times.upload += t.elapsed();
        let t = Instant::now();
        graph.launch().sync_on(p.stream())?;
        times.filter += t.elapsed();
        let t = Instant::now();
        edges = p.download_edges()?;
        times.download += t.elapsed();
    }
    times.report("gpu graph", frames, cpu_frame);
    println!("    matches cpu: {}", edges == reference);
    Ok(())
}

#[derive(Default)]
struct Timings {
    upload: Duration,
    filter: Duration,
    download: Duration,
}

impl Timings {
    fn report(&self, label: &str, frames: usize, cpu_frame: Duration) {
        let n = frames as u32;
        let (up, filt, down) = (self.upload / n, self.filter / n, self.download / n);
        let total = up + filt + down;
        println!(
            "{label:<24} filter {:>10} ({:.1}x cpu)   + upload {} + download {} = {:.0} fps end to end",
            ms(filt),
            cpu_frame.as_secs_f64() / filt.as_secs_f64(),
            ms(up),
            ms(down),
            1.0 / total.as_secs_f64()
        );
    }
}

fn ms(d: Duration) -> String {
    format!("{:.2} ms", d.as_secs_f64() * 1e3)
}

/// Gradient background, random circles and boxes, plus per-pixel noise.
/// The noise is there on purpose: Sobel on the raw image lights it all up,
/// and blurring first is what removes it.
fn synthetic_image(width: u32, height: u32, seed: u64) -> RgbImage {
    let mut rng = seed | 1;
    let mut next = move || {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    };
    let (w, h) = (width as i64, height as i64);
    let mut img = RgbImage::from_fn(width, height, |x, y| {
        let (fx, fy) = (x as f32 / width as f32, y as f32 / height as f32);
        image::Rgb([
            (30.0 + 150.0 * fx) as u8,
            (40.0 + 120.0 * fy) as u8,
            (110.0 - 60.0 * fx) as u8,
        ])
    });

    let min_dim = w.min(h);
    for i in 0..80 {
        let size = min_dim / 30 + (next() as i64 % (min_dim / 6));
        let (cx, cy) = (next() as i64 % w, next() as i64 % h);
        let color = [next() as u8, next() as u8, next() as u8];
        let circle = i % 2 == 0;
        for y in (cy - size).max(0)..(cy + size).min(h) {
            for x in (cx - size).max(0)..(cx + size).min(w) {
                let (dx, dy) = (x - cx, y - cy);
                if !circle || dx * dx + dy * dy <= size * size {
                    img.put_pixel(x as u32, y as u32, image::Rgb(color));
                }
            }
        }
    }
    for px in img.pixels_mut() {
        for ch in px.0.iter_mut() {
            let noise = (next() % 49) as i32 - 24;
            *ch = (*ch as i32 + noise).clamp(0, 255) as u8;
        }
    }
    img
}
