mod cpu;
mod gpu;
mod scene;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use cutile::prelude::*;
use image::RgbImage;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};

use gpu::{Pipeline, Quality, PARAMS, TILE};
use scene::{pack, Canvas, LIGHT, WALL};

#[derive(Parser)]
#[command(about = "2D lighting from a jump-flooded distance field, as cuTile tile kernels")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Open a window and paint lights and walls.
    ///
    /// Left drag: paint light (Shift: erase). Right drag: paint wall.
    /// Scroll: brush size. 1-6: light color. V: view (lit, radiance,
    /// distance, voronoi). Q: quality. O: orbiting light. Space: pause it.
    /// C: clear. R: reset scene. P: screenshot. Esc: quit.
    Run {
        #[arg(long, default_value_t = 640)]
        width: usize,
        #[arg(long, default_value_t = 352)]
        height: usize,
        /// Window pixels per world pixel: 1, 2 or 4.
        #[arg(long, default_value_t = 2)]
        scale: usize,
        /// 1 = low, 2 = medium, 3 = high.
        #[arg(long, default_value_t = 3)]
        quality: usize,
    },
    /// Render the demo scene to a PNG, averaging many frames.
    Render {
        #[arg(long, default_value_t = 640)]
        width: usize,
        #[arg(long, default_value_t = 352)]
        height: usize,
        #[arg(long, default_value_t = 2)]
        quality: usize,
        #[arg(long, default_value_t = 64)]
        frames: u32,
        #[arg(long, value_enum, default_value_t = View::Lit)]
        view: View,
        #[arg(long, default_value = "light2d.png")]
        out: PathBuf,
    },
    /// Time the pipeline stages on GPU and CPU, and check the lighting
    /// against the CPU renderer.
    Bench {
        #[arg(long, default_value_t = 640)]
        width: usize,
        #[arg(long, default_value_t = 352)]
        height: usize,
        #[arg(long, default_value_t = 2)]
        quality: usize,
        #[arg(long, default_value_t = 300)]
        frames: u32,
    },
    /// Check the GPU jump flood against the CPU and an exact distance transform.
    Check {
        #[arg(long, default_value_t = 640)]
        width: usize,
        #[arg(long, default_value_t = 352)]
        height: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
}

/// World size in pixels, rounded up to whole tiles. GPU buffers add one
/// ghost tile before the first row and column.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub rows: usize,
    pub cols: usize,
}

impl Layout {
    pub fn new(height: usize, width: usize) -> Self {
        Layout {
            rows: height.div_ceil(TILE).max(1) * TILE,
            cols: width.div_ceil(TILE).max(1) * TILE,
        }
    }

    pub fn buf_rows(&self) -> usize {
        self.rows + TILE
    }

    pub fn buf_cols(&self) -> usize {
        self.cols + TILE
    }

    /// Jump sizes: the largest power of two below the buffer size, halving
    /// down to 1, then one more pass at 1 ("JFA+1"), which fixes most of
    /// the pixels plain JFA gets wrong.
    pub fn jfa_offsets(&self) -> Vec<usize> {
        let mut k = self.buf_rows().max(self.buf_cols()).next_power_of_two() / 2;
        let mut out = Vec::new();
        while k >= 1 {
            out.push(k);
            k /= 2;
        }
        out.push(1);
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum View {
    Lit,
    Radiance,
    Distance,
    Voronoi,
}

impl View {
    fn next(self) -> View {
        match self {
            View::Lit => View::Radiance,
            View::Radiance => View::Distance,
            View::Distance => View::Voronoi,
            View::Voronoi => View::Lit,
        }
    }
}

const QUALITY: [Quality; 3] = [
    Quality {
        rays: 8,
        steps: 32,
        check_every: 8,
    },
    Quality {
        rays: 16,
        steps: 48,
        check_every: 8,
    },
    Quality {
        rays: 32,
        steps: 64,
        check_every: 16,
    },
];

fn quality(level: usize) -> Quality {
    QUALITY[level.clamp(1, 3) - 1]
}

/// Per-frame values for the parameter buffer (see `kernels::radiance`).
struct Settings {
    frame: u32,
    /// Weight of the newest frame in the running average.
    blend: f32,
    intensity: f32,
    view: View,
    exposure: f32,
    ambient: f32,
}

impl Settings {
    fn new(view: View) -> Self {
        Settings {
            frame: 0,
            blend: 0.25,
            intensity: 3.0,
            view,
            exposure: 1.6,
            ambient: 0.03,
        }
    }

    fn params(&self, q: Quality) -> [f32; PARAMS] {
        let mut p = [0f32; PARAMS];
        // Wrap the frame counter so the float stays exact.
        p[0] = (self.frame % 4096) as f32;
        p[1] = self.blend;
        p[2] = self.intensity;
        p[3] = self.view as i32 as f32;
        p[4] = q.rays as f32;
        p[5] = self.exposure;
        p[6] = self.ambient;
        p
    }
}

const PALETTE: [[u8; 3]; 6] = [
    [255, 190, 110],
    [255, 80, 180],
    [90, 200, 255],
    [120, 255, 140],
    [255, 250, 240],
    [140, 110, 255],
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tilekit::enable_jit_cache()?;
    match Cli::parse().cmd {
        Cmd::Run {
            width,
            height,
            scale,
            quality: q,
        } => run(Layout::new(height, width), scale, quality(q)),
        Cmd::Render {
            width,
            height,
            quality: q,
            frames,
            view,
            out,
        } => render(Layout::new(height, width), quality(q), frames, view, &out),
        Cmd::Bench {
            width,
            height,
            quality: q,
            frames,
        } => bench(Layout::new(height, width), quality(q), frames),
        Cmd::Check {
            width,
            height,
            seed,
        } => check(Layout::new(height, width), seed),
    }
}

fn run(lay: Layout, scale: usize, q: Quality) -> Result<(), Box<dyn std::error::Error>> {
    let scale = match scale {
        1 => Scale::X1,
        2 => Scale::X2,
        4 => Scale::X4,
        _ => return Err("--scale must be 1, 2 or 4".into()),
    };
    let stream = Device::new(0)?.new_stream()?;
    let mut pipeline = Pipeline::new(&stream, lay, q)?;
    let mut settings = Settings::new(View::Lit);
    let mut canvas = Canvas::demo(lay);
    let mut stage = canvas.clone();
    pipeline.upload_scene(&stage.cells)?;
    pipeline.set_params(&settings.params(q))?;
    println!("compiling kernels (slow the first time, cached on disk after)...");
    pipeline.run_eager(0)?;
    let mut graphs = pipeline.capture()?;

    let mut window = Window::new(
        "light2d",
        lay.cols,
        lay.rows,
        WindowOptions {
            scale,
            ..WindowOptions::default()
        },
    )?;
    window.set_target_fps(0);

    let (mut color, mut brush, mut level) = (
        0usize,
        4.0f32,
        QUALITY.iter().position(|x| x.rays == q.rays).unwrap_or(1),
    );
    let (mut orbit, mut paused, mut orbit_t) = (true, false, 0.0f32);
    let mut last_mouse: Option<(f32, f32)> = None;
    let mut last = Instant::now();
    let mut stats = Stats::default();
    let mut last_title = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let dt = last.elapsed().as_secs_f32();
        last = Instant::now();

        for key in window.get_keys_pressed(KeyRepeat::No) {
            match key {
                Key::Key1 | Key::Key2 | Key::Key3 | Key::Key4 | Key::Key5 | Key::Key6 => {
                    color = key as usize - Key::Key1 as usize;
                }
                Key::V => settings.view = settings.view.next(),
                Key::O => orbit = !orbit,
                Key::Space => paused = !paused,
                Key::C => canvas.clear(),
                Key::R => canvas = Canvas::demo(lay),
                Key::Q => {
                    level = (level + 1) % QUALITY.len();
                    pipeline.quality = QUALITY[level];
                    graphs = pipeline.capture()?;
                }
                Key::P => {
                    let path = format!("light2d-{}.png", settings.frame);
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
                Some(pack(WALL, [70, 70, 80]))
            } else if left && window.is_key_down(Key::LeftShift) {
                Some(0)
            } else if left {
                Some(pack(LIGHT, PALETTE[color]))
            } else {
                None
            };
            if let Some(value) = value {
                canvas.stroke(last_mouse.unwrap_or(pos), pos, brush, value);
            }
        }
        last_mouse = if left || right { mouse } else { None };

        // The painted canvas plus the orbiting light, uploaded every frame.
        stage.cells.copy_from_slice(&canvas.cells);
        if !paused {
            orbit_t += dt;
        }
        if orbit {
            let (w, h) = (lay.cols as f32, lay.rows as f32);
            let x = w * 0.5 + w * 0.3 * (orbit_t * 0.7).cos();
            let y = h * 0.5 + h * 0.3 * (orbit_t * 0.9).sin();
            stage.disc(x, y, 5.0, pack(LIGHT, [255, 240, 210]));
        }

        let t = Instant::now();
        pipeline.upload_scene(&stage.cells)?;
        pipeline.set_params(&settings.params(pipeline.quality))?;
        stats.upload += t.elapsed();
        let t = Instant::now();
        graphs[settings.frame as usize % 2]
            .launch()
            .sync_on(pipeline.stream())?;
        stats.gpu += t.elapsed();
        let t = Instant::now();
        let pixels = pipeline.download_frame()?;
        stats.download += t.elapsed();
        settings.frame += 1;
        // Brush outline.
        if let Some((mx, my)) = mouse {
            outline(pixels, lay, mx, my, brush);
        }
        window.update_with_buffer(pixels, lay.cols, lay.rows)?;
        stats.frames += 1;

        if last_title.elapsed() >= Duration::from_secs(1) {
            let n = stats.frames.max(1);
            let pq = pipeline.quality;
            let title = format!(
                "light2d {}x{} {:?} | {:.0} fps | upload {:.2} + gpu {:.2} + download {:.2} ms | {} rays x {} steps | brush {:.0}",
                lay.cols,
                lay.rows,
                settings.view,
                stats.frames as f64 / last_title.elapsed().as_secs_f64(),
                ms(stats.upload / n),
                ms(stats.gpu / n),
                ms(stats.download / n),
                pq.rays,
                pq.steps,
                brush
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

fn render(
    lay: Layout,
    q: Quality,
    frames: u32,
    view: View,
    out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = Device::new(0)?.new_stream()?;
    let mut pipeline = Pipeline::new(&stream, lay, q)?;
    pipeline.upload_scene(&Canvas::demo(lay).cells)?;
    let mut settings = Settings::new(view);
    pipeline.set_params(&settings.params(q))?;
    let t = Instant::now();
    pipeline.run_eager(0)?;
    println!("first frame (JIT) {:.0} ms", ms(t.elapsed()));
    let graphs = pipeline.capture()?;

    // Progressive average: frame i gets weight 1 / (i + 1).
    let t = Instant::now();
    for i in 0..frames.max(1) {
        settings.frame = i;
        settings.blend = 1.0 / (i + 1) as f32;
        pipeline.set_params(&settings.params(q))?;
        graphs[i as usize % 2].launch().sync_on(pipeline.stream())?;
    }
    let per_frame = t.elapsed() / frames.max(1);
    to_image(pipeline.download_frame()?, lay).save(out)?;
    println!(
        "{}x{}, {} rays x {} steps: {:.2} ms/frame (flood + light + compose), {} frames averaged -> {}",
        lay.cols,
        lay.rows,
        q.rays,
        q.steps,
        ms(per_frame),
        frames,
        out.display()
    );
    Ok(())
}

fn bench(lay: Layout, q: Quality, frames: u32) -> Result<(), Box<dyn std::error::Error>> {
    let canvas = Canvas::demo(lay);
    let mut settings = Settings::new(View::Lit);
    settings.blend = 1.0;
    let params = settings.params(q);
    let passes = lay.jfa_offsets().len();
    println!(
        "{}x{} world, {} jump flood passes, {} rays x {} steps (check every {})",
        lay.cols, lay.rows, passes, q.rays, q.steps, q.check_every
    );

    let t = Instant::now();
    let seeds = cpu::jump_flood(&lay, &canvas.cells);
    let dist = cpu::distance(&lay, &seeds);
    let cpu_flood = t.elapsed();
    let t = Instant::now();
    let cpu_light = cpu::radiance(&lay, &dist, &canvas.cells, &params, q);
    let cpu_rad = t.elapsed();
    println!(
        "cpu rayon ({} threads): flood {:.1} ms, radiance {:.1} ms",
        rayon::current_num_threads(),
        ms(cpu_flood),
        ms(cpu_rad)
    );

    let stream = Device::new(0)?.new_stream()?;
    let mut p = Pipeline::new(&stream, lay, q)?;
    p.upload_scene(&canvas.cells)?;
    p.set_params(&params)?;
    p.run_eager(0)?; // JIT
    println!("gpu dist matches cpu: {}", p.download_dist()? == dist);
    let gpu_light = p.download_light(0)?;
    let (mut max, mut sum, mut over) = (0f32, 0f64, 0usize);
    for (i, c) in cpu_light.iter().enumerate() {
        let mut worst = 0f32;
        for ch in 0..3 {
            let d = (gpu_light[i * 4 + ch] - c[ch]).abs();
            sum += d as f64;
            worst = worst.max(d);
        }
        max = max.max(worst);
        over += (worst > 1e-3) as usize;
    }
    let mean: f64 = cpu_light
        .iter()
        .map(|c| (c[0] + c[1] + c[2]) as f64)
        .sum::<f64>()
        / (cpu_light.len() * 3) as f64;
    println!(
        "gpu radiance vs cpu (mean radiance {:.4}): mean abs diff {:.2e}, max {:.2e}, {:.3}% of pixels off by more than 1e-3",
        mean,
        sum / (cpu_light.len() * 3) as f64,
        max,
        over as f64 * 100.0 / cpu_light.len() as f64
    );

    for i in 0..50 {
        p.run_eager(i % 2)?;
    }
    // Flood-only runs compare with the CPU flood; whole frames with the CPU
    // flood + radiance (the CPU side skips compose).
    let time = |label: &str, total: Duration, cpu: Duration| {
        let per = total / frames;
        println!(
            "{label:<28} {:>7.3} ms/frame {:>6.0} fps {:>7.1}x cpu",
            ms(per),
            1.0 / per.as_secs_f64(),
            cpu.as_secs_f64() / per.as_secs_f64()
        );
    };
    let cpu_frame = cpu_flood + cpu_rad;
    let t = Instant::now();
    for i in 0..frames {
        p.run_eager(i as usize % 2)?;
    }
    time("gpu eager, whole frame", t.elapsed(), cpu_frame);

    let flood = p.capture_flood()?;
    let t = Instant::now();
    for _ in 0..frames {
        flood.launch().sync_on(p.stream())?;
    }
    let flood_time = t.elapsed();
    time("gpu graph, flood only", flood_time, cpu_flood);

    let graphs = p.capture()?;
    let t = Instant::now();
    for i in 0..frames {
        graphs[i as usize % 2].launch().sync_on(p.stream())?;
    }
    let frame_time = t.elapsed();
    time("gpu graph, whole frame", frame_time, cpu_frame);
    println!(
        "    so radiance + compose take {:.3} ms",
        ms(frame_time.saturating_sub(flood_time) / frames)
    );

    p.quality = Quality {
        check_every: q.steps,
        ..q
    };
    let graphs = p.capture()?;
    let t = Instant::now();
    for i in 0..frames {
        graphs[i as usize % 2].launch().sync_on(p.stream())?;
    }
    time("gpu graph, no early exit", t.elapsed(), cpu_frame);

    let t = Instant::now();
    for _ in 0..frames {
        p.upload_scene(&canvas.cells)?;
    }
    let up = t.elapsed() / frames;
    let t = Instant::now();
    for _ in 0..frames {
        p.download_frame()?;
    }
    println!(
        "per frame: scene upload {:.3} ms, frame download {:.3} ms",
        ms(up),
        ms(t.elapsed() / frames)
    );
    Ok(())
}

fn check(lay: Layout, seed: u64) -> Result<(), Box<dyn std::error::Error>> {
    let canvas = Canvas::random(lay, seed);
    println!(
        "{}x{} world ({}x{} buffer), jumps {:?}",
        lay.cols,
        lay.rows,
        lay.buf_cols(),
        lay.buf_rows(),
        lay.jfa_offsets()
    );

    let stream = Device::new(0)?.new_stream()?;
    let mut p = Pipeline::new(&stream, lay, QUALITY[1])?;
    p.upload_scene(&canvas.cells)?;
    let t = Instant::now();
    p.flood_eager()?;
    println!("gpu first flood (JIT) {:.0} ms", ms(t.elapsed()));
    let t = Instant::now();
    p.flood_eager()?;
    println!("gpu eager flood {:.3} ms", ms(t.elapsed()));
    let seeds = p.download_seeds()?;

    let t = Instant::now();
    let cpu_seeds = cpu::jump_flood(&lay, &canvas.cells);
    println!("cpu jump flood {:.1} ms", ms(t.elapsed()));
    println!("seeds match cpu: {}", seeds == cpu_seeds);
    let cpu_dist = cpu::distance(&lay, &cpu_seeds);
    println!("distance matches cpu: {}", p.download_dist()? == cpu_dist);
    let cpu_nearest = cpu::nearest(&cpu_seeds, &canvas.cells, lay.buf_cols());
    println!(
        "nearest matches cpu: {}",
        p.download_nearest()? == cpu_nearest
    );

    // A captured graph re-reads the scene buffer on every launch.
    let graph = p.capture_flood()?;
    let other = Canvas::random(lay, seed + 1);
    p.upload_scene(&other.cells)?;
    graph.launch().sync_on(p.stream())?;
    println!(
        "graph on a new scene matches cpu: {}",
        p.download_seeds()? == cpu::jump_flood(&lay, &other.cells)
    );

    // How close is JFA to the true nearest distance? The CPU flood matches
    // the GPU bit for bit, so it can also show plain JFA without the extra
    // pass at k = 1.
    let exact = cpu::exact_dist2(&lay, &canvas.cells);
    let mut plain = cpu::seed_init(&lay, &canvas.cells);
    let offsets = lay.jfa_offsets();
    for &k in &offsets[..offsets.len() - 1] {
        plain = cpu::jfa_step(&lay, &plain, k);
    }
    for (label, result) in [("jfa", &plain), ("jfa+1", &seeds)] {
        let cols = lay.buf_cols();
        let (mut wrong, mut worst, mut total) = (0usize, 0f64, 0usize);
        for r in TILE..lay.buf_rows() {
            for c in TILE..cols {
                let i = r * cols + c;
                let d = cpu::dist2(result[i], r as i32, c as i32) as f64;
                total += 1;
                if d != exact[i] {
                    wrong += 1;
                    worst = worst.max(d.sqrt() - exact[i].sqrt());
                }
            }
        }
        println!(
            "{label:<6} vs exact distance: {} of {} pixels off ({:.4}%), worst by {:.2} px",
            wrong,
            total,
            wrong as f64 * 100.0 / total as f64,
            worst
        );
    }
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
