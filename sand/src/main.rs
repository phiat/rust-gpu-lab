mod cpu;
mod gpu;
mod world;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use cutile::prelude::*;
use image::RgbImage;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};

use gpu::Pipeline;
use world::*;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Open a window and paint elements.
    ///
    /// Left drag paints, right drag erases, scroll sets the brush size.
    /// 1-8 pick sand, water, oil, wood, wall, fire, smoke, ember. Space pauses,
    /// E toggles the sand and water taps, C clears, R resets the scene,
    /// P saves a screenshot, Esc quits.
    Run {
        #[arg(long, default_value_t = 640)]
        width: usize,
        #[arg(long, default_value_t = 352)]
        height: usize,
        /// Window pixels per cell: 1, 2 or 4.
        #[arg(long, default_value_t = 2)]
        scale: usize,
        /// Margolus passes per frame (even). Each pair of passes is one
        /// full update.
        #[arg(long, default_value_t = 4)]
        passes: usize,
    },
    /// Run the demo scene for a while and save a PNG.
    Render {
        #[arg(long, default_value_t = 640)]
        width: usize,
        #[arg(long, default_value_t = 352)]
        height: usize,
        #[arg(long, default_value_t = 4)]
        passes: usize,
        #[arg(long, default_value_t = 240)]
        frames: u32,
        #[arg(long, default_value = "sand.png")]
        out: PathBuf,
    },
    /// Time a frame on the CPU and the GPU (eager and graph).
    Bench {
        #[arg(long, default_value_t = 640)]
        width: usize,
        #[arg(long, default_value_t = 352)]
        height: usize,
        #[arg(long, default_value_t = 4)]
        passes: usize,
        #[arg(long, default_value_t = 300)]
        frames: u32,
    },
    /// Check the GPU against the CPU, cell for cell, over many frames.
    Check {
        #[arg(long, default_value_t = 640)]
        width: usize,
        #[arg(long, default_value_t = 352)]
        height: usize,
        #[arg(long, default_value_t = 4)]
        passes: usize,
        #[arg(long, default_value_t = 60)]
        frames: u32,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tilekit::enable_jit_cache()?;
    match Cli::parse().cmd {
        Cmd::Run {
            width,
            height,
            scale,
            passes,
        } => run(Layout::new(height, width), scale, passes),
        Cmd::Render {
            width,
            height,
            passes,
            frames,
            out,
        } => render(Layout::new(height, width), passes, frames, &out),
        Cmd::Bench {
            width,
            height,
            passes,
            frames,
        } => bench(Layout::new(height, width), passes, frames),
        Cmd::Check {
            width,
            height,
            passes,
            frames,
            seed,
        } => check(Layout::new(height, width), passes, frames, seed),
    }
}

const BRUSH_KEYS: [(Key, i32); 8] = [
    (Key::Key1, SAND),
    (Key::Key2, WATER),
    (Key::Key3, OIL),
    (Key::Key4, WOOD),
    (Key::Key5, WALL),
    (Key::Key6, FIRE),
    (Key::Key7, SMOKE),
    (Key::Key8, EMBER),
];

fn run(lay: Layout, scale: usize, passes: usize) -> Result<(), Box<dyn std::error::Error>> {
    let scale = match scale {
        1 => Scale::X1,
        2 => Scale::X2,
        4 => Scale::X4,
        _ => return Err("--scale must be 1, 2 or 4".into()),
    };
    let stream = Device::new(0)?.new_stream()?;
    let mut pipeline = Pipeline::new(&stream, lay, passes)?;
    pipeline.upload_world(0, &Canvas::demo(lay).into_world())?;
    pipeline.set_frame(0)?;
    println!("compiling kernels (slow the first time, cached on disk after)...");
    pipeline.run_eager(0)?;
    // That eager frame moved the world to buffer 1, so start at parity 1.
    let mut frame: u32 = 1;
    let graphs = pipeline.capture()?;

    let mut window = Window::new(
        "sand",
        lay.cols,
        lay.rows,
        WindowOptions {
            scale,
            ..WindowOptions::default()
        },
    )?;
    window.set_target_fps(0);

    let mut paint = Canvas::blank(lay);
    let (mut element, mut brush) = (SAND, 4.0f32);
    let (mut paused, mut taps) = (false, true);
    let mut paint_dirty = true;
    let mut last_mouse: Option<(f32, f32)> = None;
    let mut stats = Stats::default();
    let mut last_title = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for key in window.get_keys_pressed(KeyRepeat::No) {
            if let Some((_, e)) = BRUSH_KEYS.iter().find(|(k, _)| *k == key) {
                element = *e;
            }
            match key {
                Key::Space => paused = !paused,
                Key::E => taps = !taps,
                Key::C => {
                    pipeline.upload_world(frame as usize % 2, &lay.empty_buffer())?;
                }
                Key::R => {
                    pipeline.upload_world(frame as usize % 2, &Canvas::demo(lay).into_world())?;
                }
                Key::P => {
                    let path = format!("sand-{frame}.png");
                    to_image(pipeline.download_frame()?, lay).save(&path)?;
                    println!("\nsaved {path}");
                }
                _ => {}
            }
        }
        if let Some((_, scroll)) = window.get_scroll_wheel() {
            brush = (brush * (1.0 + scroll.signum() * 0.15)).clamp(1.0, 40.0);
        }
        let mouse = window.get_mouse_pos(MouseMode::Discard);
        let (left, right) = (
            window.get_mouse_down(MouseButton::Left),
            window.get_mouse_down(MouseButton::Right),
        );
        if let Some(pos) = mouse {
            let value = if right {
                Some(EMPTY)
            } else if left {
                Some(element)
            } else {
                None
            };
            if let Some(value) = value {
                paint.stroke(last_mouse.unwrap_or(pos), pos, brush, value);
            }
        }
        last_mouse = if left || right { mouse } else { None };
        if taps && !paused {
            let (w, h) = (lay.cols as f32, lay.rows as f32);
            paint.disc(w * 0.25, h * 0.05, 0.5, SAND);
            paint.disc(w * 0.75, h * 0.05, 0.5, WATER);
        }

        // Upload the paint layer whenever it has something in it, and once
        // more after it's emptied so the GPU copy is cleared too.
        let t = Instant::now();
        let dirty = !paint.is_empty();
        if dirty || paint_dirty {
            pipeline.paint_mut().copy_from_slice(&paint.cells);
            pipeline.upload_paint()?;
            paint.clear();
        }
        paint_dirty = dirty;
        pipeline.set_frame(frame as i32)?;
        stats.upload += t.elapsed();

        let t = Instant::now();
        if !paused || dirty {
            graphs[frame as usize % 2]
                .launch()
                .sync_on(pipeline.stream())?;
            frame += 1;
        }
        stats.gpu += t.elapsed();
        let t = Instant::now();
        let pixels = pipeline.download_frame()?;
        stats.download += t.elapsed();
        if let Some((mx, my)) = mouse {
            outline(pixels, lay, mx, my, brush);
        }
        window.update_with_buffer(pixels, lay.cols, lay.rows)?;
        stats.frames += 1;

        if last_title.elapsed() >= Duration::from_secs(1) {
            let n = stats.frames.max(1);
            let title = format!(
                "sand {}x{} | {:.0} fps | upload {:.2} + gpu {:.2} + download {:.2} ms | {} passes | brush: {} {:.0}{}",
                lay.cols,
                lay.rows,
                stats.frames as f64 / last_title.elapsed().as_secs_f64(),
                ms(stats.upload / n),
                ms(stats.gpu / n),
                ms(stats.download / n),
                passes,
                NAMES[element as usize],
                brush,
                if paused { " | paused" } else { "" }
            );
            window.set_title(&title);
            print!("\r{title}   ");
            std::io::Write::flush(&mut std::io::stdout())?;
            stats = Stats::default();
            last_title = Instant::now();
        }
    }
    println!();
    Ok(())
}

#[derive(Default)]
struct Stats {
    frames: u32,
    upload: Duration,
    gpu: Duration,
    download: Duration,
}

/// Draw a thin circle where the brush is.
fn outline(pixels: &mut [u32], lay: Layout, cx: f32, cy: f32, radius: f32) {
    let n = (radius * 6.0).max(16.0) as usize;
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        let (x, y) = (
            (cx + radius * a.cos()) as i32,
            (cy + radius * a.sin()) as i32,
        );
        if x >= 0 && y >= 0 && (x as usize) < lay.cols && (y as usize) < lay.rows {
            pixels[y as usize * lay.cols + x as usize] = 0x00c0c0c0;
        }
    }
}

/// Run the demo scene with the taps on for `frames` frames.
fn render(
    lay: Layout,
    passes: usize,
    frames: u32,
    out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = Device::new(0)?.new_stream()?;
    let mut pipeline = Pipeline::new(&stream, lay, passes)?;
    pipeline.upload_world(0, &Canvas::demo(lay).into_world())?;
    pipeline.set_frame(0)?;
    let t = Instant::now();
    pipeline.run_eager(0)?;
    println!("first frame (JIT) {:.0} ms", ms(t.elapsed()));
    let graphs = pipeline.capture()?;

    let mut paint = Canvas::blank(lay);
    let (w, h) = (lay.cols as f32, lay.rows as f32);
    // Light the hut late enough that it is still burning at the end.
    let ignite = frames - frames / 6;
    let t = Instant::now();
    for frame in 1..=frames {
        paint.disc(w * 0.25, h * 0.05, 0.5, SAND);
        paint.disc(w * 0.75, h * 0.05, 0.5, WATER);
        if frame == ignite {
            paint.disc(w * 0.66, h * 0.9, 3.0, FIRE);
        }
        pipeline.paint_mut().copy_from_slice(&paint.cells);
        pipeline.upload_paint()?;
        paint.clear();
        pipeline.set_frame(frame as i32)?;
        graphs[frame as usize % 2]
            .launch()
            .sync_on(pipeline.stream())?;
    }
    let per_frame = t.elapsed() / frames.max(1);
    to_image(pipeline.download_frame()?, lay).save(out)?;
    println!(
        "{}x{}, {} passes/frame: {:.3} ms/frame including uploads, {} frames -> {}",
        lay.cols,
        lay.rows,
        passes,
        ms(per_frame),
        frames,
        out.display()
    );
    Ok(())
}

fn bench(lay: Layout, passes: usize, frames: u32) -> Result<(), Box<dyn std::error::Error>> {
    let world = Canvas::random(lay, 1).into_world();
    println!(
        "{}x{} world ({}x{} buffer), {} passes per frame",
        lay.cols,
        lay.rows,
        lay.buf_cols(),
        lay.buf_rows(),
        passes
    );

    let cpu_frames = 20;
    let t = Instant::now();
    let mut cells = world.clone();
    for f in 0..cpu_frames {
        for pass in 0..passes as i32 {
            cells = cpu::step(&lay, &cells, pass, f * 64 + pass);
        }
    }
    let cpu_frame = t.elapsed() / cpu_frames as u32;
    println!(
        "cpu rayon ({} threads)      {:.3} ms/frame",
        rayon::current_num_threads(),
        ms(cpu_frame)
    );

    let stream = Device::new(0)?.new_stream()?;
    let mut p = Pipeline::new(&stream, lay, passes)?;
    p.upload_world(0, &world)?;
    p.set_frame(0)?;
    let t = Instant::now();
    p.run_eager(0)?;
    println!("gpu first frame (JIT) {:.0} ms", ms(t.elapsed()));
    let time = |label: &str, d: Duration| {
        println!(
            "{label:<28} {:.3} ms/frame  {:>6.0} fps  {:.1}x cpu",
            ms(d / frames),
            frames as f64 / d.as_secs_f64(),
            cpu_frame.as_secs_f64() / (d / frames).as_secs_f64()
        );
    };

    let t = Instant::now();
    for f in 1..=frames {
        p.set_frame(f as i32)?;
        p.run_eager(f as usize % 2)?;
    }
    time("gpu eager", t.elapsed());

    let graphs = p.capture()?;
    let t = Instant::now();
    for f in 1..=frames {
        p.set_frame(f as i32)?;
        graphs[f as usize % 2].launch().sync_on(p.stream())?;
    }
    time("gpu graph", t.elapsed());

    let t = Instant::now();
    for _ in 0..frames {
        p.upload_paint()?;
    }
    let up = t.elapsed() / frames;
    let t = Instant::now();
    for _ in 0..frames {
        p.download_frame()?;
    }
    println!(
        "per frame: paint upload {:.3} ms, frame download {:.3} ms",
        ms(up),
        ms(t.elapsed() / frames)
    );
    Ok(())
}

fn check(
    lay: Layout,
    passes: usize,
    frames: u32,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let world = Canvas::random(lay, seed).into_world();
    let stream = Device::new(0)?.new_stream()?;
    let mut p = Pipeline::new(&stream, lay, passes)?;
    {
        let bc = lay.buf_cols() as i32;
        let gpu = p.hash_grid()?;
        let bad = gpu
            .iter()
            .enumerate()
            .filter(|(i, &h)| h != cpu::hash((*i as i32 / bc) * 4096 + *i as i32 % bc))
            .count();
        println!("hash matches cpu: {} ({bad} cells differ)", bad == 0);
        for i in [0usize, 1, 2, 3, 1000] {
            let x = (i as i32 / bc) * 4096 + i as i32 % bc;
            println!("  x={x}: gpu {:#x} cpu {:#x}", gpu[i], cpu::hash(x));
        }
    }
    p.upload_world(0, &world)?;
    let graphs = {
        // Warm up on a throwaway copy, then restore the world.
        p.set_frame(0)?;
        p.run_eager(0)?;
        let g = p.capture()?;
        p.upload_world(0, &world)?;
        g
    };

    // Same frames on the CPU: a paint stroke every 10th frame, taps on.
    let mut cpu_world = world.clone();
    let mut paint = Canvas::blank(lay);
    let (w, h) = (lay.cols as f32, lay.rows as f32);
    let mut first_bad = None;
    let mut counts = [0usize; 9];
    for frame in 0..frames {
        paint.disc(w * 0.25, h * 0.05, 0.5, SAND);
        paint.disc(w * 0.75, h * 0.05, 0.5, WATER);
        if frame % 10 == 5 {
            let k = 1 + (frame / 10) as i32 % 8;
            paint.disc(w * 0.5, h * 0.5, 6.0, k);
        }
        p.paint_mut().copy_from_slice(&paint.cells);
        p.upload_paint()?;
        p.set_frame(frame as i32)?;
        let parity = frame as usize % 2;
        if frame % 2 == 0 {
            p.run_eager(parity)?;
        } else {
            graphs[parity].launch().sync_on(p.stream())?;
        }

        cpu_world = cpu::paint(&cpu_world, &paint.cells);
        for pass in 0..passes as i32 {
            cpu_world = cpu::step(&lay, &cpu_world, pass, frame as i32 * 64 + pass);
        }
        paint.clear();

        let gpu_world = p.download_world(parity)?;
        if gpu_world != cpu_world.as_slice() && first_bad.is_none() {
            let bad = gpu_world
                .iter()
                .zip(&cpu_world)
                .filter(|(a, b)| a != b)
                .count();
            first_bad = Some((frame, bad));
            let bc = lay.buf_cols();
            for (i, (a, b)) in gpu_world
                .iter()
                .zip(&cpu_world)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .take(12)
            {
                println!(
                    "  ({}, {}): gpu {:#x} cpu {:#x}  src {:#x}",
                    i / bc,
                    i % bc,
                    a,
                    b,
                    world[i]
                );
            }
        }
        for &c in &cpu::world(&lay, &cpu_world) {
            counts[kind(c) as usize] += 1;
        }
    }
    match first_bad {
        None => println!(
            "gpu matches cpu on every cell of {frames} frames ({} passes each)",
            passes
        ),
        Some((frame, bad)) => println!("MISMATCH at frame {frame}: {bad} cells differ"),
    }
    let total: usize = counts.iter().sum();
    print!("cells seen:");
    for (k, n) in counts.iter().enumerate() {
        print!(" {} {:.1}%", NAMES[k], *n as f64 * 100.0 / total as f64);
    }
    println!();
    Ok(())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn to_image(pixels: &[u32], lay: Layout) -> RgbImage {
    let raw = pixels
        .iter()
        .flat_map(|p| [(p >> 16) as u8, (p >> 8) as u8, *p as u8])
        .collect();
    RgbImage::from_raw(lay.cols as u32, lay.rows as u32, raw).unwrap()
}
