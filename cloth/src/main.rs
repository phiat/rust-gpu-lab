mod cpu;
mod gpu;
mod raster;
mod world;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use cutile::prelude::*;
use image::RgbImage;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};

use gpu::Pipeline;
use raster::Frame;
use world::*;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone, Copy)]
struct Size {
    /// Particles across.
    #[arg(long, default_value_t = 256)]
    cols: usize,
    /// Particles down.
    #[arg(long, default_value_t = 160)]
    rows: usize,
    /// Verlet steps per frame.
    #[arg(long, default_value_t = 4)]
    substeps: usize,
    /// Solver iterations per substep (12 constraint batches each).
    #[arg(long, default_value_t = 2)]
    iterations: usize,
}

#[derive(Subcommand)]
enum Cmd {
    /// Open a window: the cloth blows in the wind around a sphere.
    ///
    /// The sphere follows the mouse; right drag orbits the camera, scroll
    /// changes the wind. 1-4 pin the top row / a few points / the corners /
    /// nothing (and reset). W toggles wind, Space pauses, R resets, P saves
    /// a screenshot, Esc quits.
    Run {
        #[command(flatten)]
        size: Size,
        #[arg(long, default_value_t = 960)]
        width: usize,
        #[arg(long, default_value_t = 640)]
        height: usize,
    },
    /// Run the demo for a while and save a PNG.
    Render {
        #[command(flatten)]
        size: Size,
        #[arg(long, default_value_t = 960)]
        width: usize,
        #[arg(long, default_value_t = 640)]
        height: usize,
        #[arg(long, default_value_t = 240)]
        frames: u32,
        #[arg(long, default_value = "cloth.png")]
        out: PathBuf,
    },
    /// Time a frame on the CPU and the GPU (eager and graph).
    Bench {
        #[command(flatten)]
        size: Size,
        #[arg(long, default_value_t = 300)]
        frames: u32,
    },
    /// Compare the GPU with the CPU, particle for particle, over many frames.
    Check {
        #[command(flatten)]
        size: Size,
        #[arg(long, default_value_t = 30)]
        frames: u32,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tilekit::enable_jit_cache()?;
    match Cli::parse().cmd {
        Cmd::Run {
            size,
            width,
            height,
        } => run(size, width, height),
        Cmd::Render {
            size,
            width,
            height,
            frames,
            out,
        } => render(size, width, height, frames, &out),
        Cmd::Bench { size, frames } => bench(size, frames),
        Cmd::Check { size, frames } => check(size, frames),
    }
}

/// Everything a scene needs: layout, physics, camera and the pin pattern.
struct Scene {
    lay: Layout,
    ph: Physics,
    cam: Camera,
    pins: Pins,
}

impl Scene {
    fn demo(size: Size, width: usize, height: usize) -> Self {
        let lay = Layout::new(size.rows, size.cols);
        let ph = Physics::demo(lay);
        Scene {
            lay,
            ph,
            cam: Camera::demo(lay, width, height),
            pins: Pins::Top,
        }
    }

    fn params(&self) -> [f32; PARAMS] {
        let mut p = [0.0f32; PARAMS];
        self.ph.write(&mut p);
        self.cam.write(&mut p);
        p
    }

    fn apply(&self, pipeline: &mut Pipeline) -> Result<(), Error> {
        pipeline.params_mut().copy_from_slice(&self.params());
        pipeline.upload_params()
    }

    fn cloth(&self) -> Vec<f32> {
        initial(self.lay, self.pins, self.ph.spacing)
    }

    fn anchors(&self) -> Vec<f32> {
        anchors(self.lay, self.pins, self.ph.spacing)
    }

    /// Advance the clock by one frame.
    fn tick(&mut self, size: Size) {
        self.ph.time += self.ph.dt * size.substeps as f32;
    }
}

fn pipeline(
    stream: &std::sync::Arc<cuda_core::Stream>,
    scene: &Scene,
    size: Size,
) -> Result<Pipeline, Error> {
    let mut p = Pipeline::new(stream, scene.lay, size.substeps, size.iterations)?;
    p.upload_cloth(0, &scene.cloth(), &scene.anchors())?;
    scene.apply(&mut p)?;
    Ok(p)
}

const PIN_KEYS: [(Key, Pins); 4] = [
    (Key::Key1, Pins::Top),
    (Key::Key2, Pins::Sparse),
    (Key::Key3, Pins::Corners),
    (Key::Key4, Pins::Free),
];

fn run(size: Size, width: usize, height: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut scene = Scene::demo(size, width, height);
    let stream = Device::new(0)?.new_stream()?;
    let mut p = pipeline(&stream, &scene, size)?;
    println!("compiling kernels (slow the first time, cached on disk after)...");
    let t = Instant::now();
    p.run_eager(0)?;
    println!("first frame {:.0} ms", ms(t.elapsed()));
    let graphs = p.capture()?;
    // Start over so the warm-up frame isn't part of the scene.
    p.upload_cloth(0, &scene.cloth(), &scene.anchors())?;
    let mut parity = 0;

    let mut window = Window::new("cloth", width, height, WindowOptions::default())?;
    window.set_target_fps(0);
    let mut frame = Frame::new(width, height);
    let mut paused = false;
    let mut wind_on = true;
    let wind = scene.ph.wind;
    let mut last_mouse: Option<(f32, f32)> = None;
    let mut stats = Stats::default();
    let mut last_title = Instant::now();
    let mut shot = 0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut reset = false;
        for key in window.get_keys_pressed(KeyRepeat::No) {
            if let Some((_, pins)) = PIN_KEYS.iter().find(|(k, _)| *k == key) {
                scene.pins = *pins;
                reset = true;
            }
            match key {
                Key::Space => paused = !paused,
                Key::W => wind_on = !wind_on,
                Key::R => reset = true,
                Key::P => {
                    shot += 1;
                    let path = format!("cloth-{shot}.png");
                    to_image(&frame.pixels, width, height).save(&path)?;
                    println!("\nsaved {path}");
                }
                _ => {}
            }
        }
        if reset {
            p.upload_cloth(parity, &scene.cloth(), &scene.anchors())?;
        }
        if let Some((_, scroll)) = window.get_scroll_wheel() {
            let f = if scroll > 0.0 { 1.15 } else { 1.0 / 1.15 };
            scene.ph.wind = [wind[0] * f, wind[1] * f, wind[2] * f];
        }
        let mouse = window.get_mouse_pos(MouseMode::Discard);
        if let Some((mx, my)) = mouse {
            if window.get_mouse_down(MouseButton::Right) {
                if let Some((lx, ly)) = last_mouse {
                    scene.cam.yaw += (mx - lx) * 0.006;
                    scene.cam.pitch = (scene.cam.pitch + (my - ly) * 0.006).clamp(-1.2, 1.2);
                }
            } else {
                // Keep the sphere at its depth, under the mouse.
                let depth = scene.cam.camera_space([
                    scene.ph.sphere[0],
                    scene.ph.sphere[1],
                    scene.ph.sphere[2],
                ])[2];
                let q = [
                    (mx - width as f32 * 0.5) / scene.cam.focal * depth,
                    -(my - height as f32 * 0.5) / scene.cam.focal * depth,
                    depth,
                ];
                let w = scene.cam.unproject(q);
                scene.ph.sphere[..3].copy_from_slice(&w);
            }
        }
        last_mouse = mouse;

        let t = Instant::now();
        let saved_wind = scene.ph.wind;
        if !wind_on {
            scene.ph.wind = [0.0; 3];
        }
        scene.apply(&mut p)?;
        scene.ph.wind = saved_wind;
        stats.upload += t.elapsed();

        let t = Instant::now();
        if !paused {
            graphs[parity].launch().sync_on(p.stream())?;
            parity = p.end_parity(parity);
            scene.tick(size);
        }
        stats.gpu += t.elapsed();

        let t = Instant::now();
        let screen = p.download_screen()?;
        stats.download += t.elapsed();
        let t = Instant::now();
        frame.clear();
        draw_sphere(&mut frame, &scene);
        frame.draw_cloth(&scene.lay, screen);
        stats.raster += t.elapsed();
        window.update_with_buffer(&frame.pixels, width, height)?;
        stats.frames += 1;

        if last_title.elapsed() >= Duration::from_secs(1) {
            let n = stats.frames.max(1);
            let title = format!(
                "cloth {}x{} | {:.0} fps | gpu {:.2} + download {:.2} + raster {:.2} ms | {} substeps x {} it | pins: {} | wind {:.0}{}{}",
                scene.lay.cols,
                scene.lay.rows,
                stats.frames as f64 / last_title.elapsed().as_secs_f64(),
                ms(stats.gpu / n),
                ms(stats.download / n),
                ms(stats.raster / n),
                size.substeps,
                size.iterations,
                scene.pins.name(),
                scene.ph.wind[2],
                if wind_on { "" } else { " (off)" },
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

/// The collider, drawn a little smaller than it is so the cloth's
/// triangles, whose vertices sit exactly on its surface, cover it.
fn draw_sphere(frame: &mut Frame, scene: &Scene) {
    let s = scene.ph.sphere;
    frame.draw_sphere(
        &scene.cam,
        [s[0], s[1], s[2]],
        s[3] * 0.96,
        [0.55, 0.6, 0.66],
    );
}

#[derive(Default)]
struct Stats {
    frames: u32,
    upload: Duration,
    gpu: Duration,
    download: Duration,
    raster: Duration,
}

fn render(
    size: Size,
    width: usize,
    height: usize,
    frames: u32,
    out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut scene = Scene::demo(size, width, height);
    let stream = Device::new(0)?.new_stream()?;
    let mut p = pipeline(&stream, &scene, size)?;
    let t = Instant::now();
    p.run_eager(0)?;
    println!("first frame (JIT) {:.0} ms", ms(t.elapsed()));
    let graphs = p.capture()?;
    p.upload_cloth(0, &scene.cloth(), &scene.anchors())?;
    let mut parity = 0;
    let t = Instant::now();
    for _ in 0..frames {
        scene.apply(&mut p)?;
        graphs[parity].launch().sync_on(p.stream())?;
        parity = p.end_parity(parity);
        scene.tick(size);
    }
    let per_frame = t.elapsed() / frames.max(1);
    let stretch = cpu::stretch(&scene.lay, p.download_pos(parity)?, scene.ph.spacing);
    let mut frame = Frame::new(width, height);
    frame.clear();
    draw_sphere(&mut frame, &scene);
    frame.draw_cloth(&scene.lay, p.download_screen()?);
    to_image(&frame.pixels, width, height).save(out)?;
    println!(
        "{}x{} particles, {} substeps x {} iterations: {:.3} ms/frame, mean stretch {:.2}%, {frames} frames -> {}",
        scene.lay.cols,
        scene.lay.rows,
        size.substeps,
        size.iterations,
        ms(per_frame),
        stretch * 100.0,
        out.display()
    );
    Ok(())
}

fn bench(size: Size, frames: u32) -> Result<(), Box<dyn std::error::Error>> {
    let (width, height) = (960, 640);
    let mut scene = Scene::demo(size, width, height);
    let lay = scene.lay;
    println!(
        "{}x{} particles ({}x{} buffer), {} substeps x {} iterations = {} passes per frame",
        lay.cols,
        lay.rows,
        lay.buf_cols(),
        lay.buf_rows(),
        size.substeps,
        size.iterations,
        size.substeps * (1 + 12 * size.iterations) + 2
    );

    // Passes are short, and on a hybrid CPU (8 P-cores + 16 E-cores here)
    // every join waits for the slowest core, so fewer threads can be
    // faster than all of them. Time both and keep the better.
    let cpu_frames = 10;
    let all = rayon::current_num_threads();
    let mut cpu_frame = Duration::MAX;
    for threads in [all, 8.min(all)] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        let mut cloth = cpu::Cloth::new(lay, scene.pins, scene.ph.spacing);
        let mut ph = scene.ph;
        let d = pool.install(|| {
            let t = Instant::now();
            for _ in 0..cpu_frames {
                cloth.frame(&lay, &ph, size.substeps, size.iterations);
                cpu::shade(&lay, &cloth.pos, &cloth.nrm, &scene.cam);
                ph.time += ph.dt * size.substeps as f32;
            }
            t.elapsed() / cpu_frames
        });
        println!("cpu rayon ({threads:>2} threads)     {:.3} ms/frame", ms(d));
        cpu_frame = cpu_frame.min(d);
    }

    let stream = Device::new(0)?.new_stream()?;
    let mut p = pipeline(&stream, &scene, size)?;
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

    let mut parity = 0;
    let t = Instant::now();
    for _ in 0..frames {
        scene.apply(&mut p)?;
        p.run_eager(parity)?;
        parity = p.end_parity(parity);
        scene.tick(size);
    }
    time("gpu eager", t.elapsed());

    let graphs = p.capture()?;
    let t = Instant::now();
    for _ in 0..frames {
        scene.apply(&mut p)?;
        graphs[parity].launch().sync_on(p.stream())?;
        parity = p.end_parity(parity);
        scene.tick(size);
    }
    time("gpu graph", t.elapsed());
    let stretch = cpu::stretch(&lay, p.download_pos(parity)?, scene.ph.spacing);

    let t = Instant::now();
    for _ in 0..frames {
        p.download_screen()?;
    }
    let down = t.elapsed() / frames;
    let mut frame = Frame::new(width, height);
    let screen = p.download_screen()?.to_vec();
    let t = Instant::now();
    for _ in 0..frames {
        frame.clear();
        frame.draw_cloth(&lay, &screen);
    }
    println!(
        "per frame: params upload + screen download {:.3} ms, cpu raster {:.3} ms ({}x{})",
        ms(down),
        ms(t.elapsed() / frames),
        width,
        height
    );
    println!(
        "mean stretch after {} frames: {:.2}%",
        2 * frames,
        stretch * 100.0
    );
    Ok(())
}

fn check(size: Size, frames: u32) -> Result<(), Box<dyn std::error::Error>> {
    let mut scene = Scene::demo(size, 960, 640);
    let lay = scene.lay;
    let stream = Device::new(0)?.new_stream()?;
    let mut p = pipeline(&stream, &scene, size)?;
    p.run_eager(0)?;
    let graphs = p.capture()?;
    p.upload_cloth(0, &scene.cloth(), &scene.anchors())?;
    let mut cloth = cpu::Cloth::new(lay, scene.pins, scene.ph.spacing);

    let mut parity = 0;
    let mut worst = 0.0f32;
    let mut exact_frames = 0;
    for f in 0..frames {
        scene.apply(&mut p)?;
        if f % 2 == 0 {
            p.run_eager(parity)?;
        } else {
            graphs[parity].launch().sync_on(p.stream())?;
        }
        parity = p.end_parity(parity);
        cloth.frame(&lay, &scene.ph, size.substeps, size.iterations);
        scene.tick(size);

        let gpu = p.download_pos(parity)?;
        let (mut max, mut sum, mut n, mut arg) = (0.0f32, 0.0f64, 0usize, (0, 0));
        for r in 0..lay.rows {
            for c in 0..lay.cols {
                let i = lay.index(r, c) * 4;
                for k in 0..3 {
                    let d = (gpu[i + k] - cloth.pos[i + k]).abs();
                    sum += d as f64;
                    n += 1;
                    if d > max {
                        max = d;
                        arg = (r, c);
                    }
                }
            }
        }
        if max == 0.0 {
            exact_frames += 1;
        }
        worst = worst.max(max);
        if f < 3 || f + 1 == frames || max > 1e-2 {
            println!(
                "frame {f:>3}: max |gpu - cpu| {max:.3e} at ({}, {}), mean {:.3e}{}",
                arg.0,
                arg.1,
                sum / n as f64,
                if max == 0.0 { " (exact)" } else { "" }
            );
        }
    }
    let screen_gpu = p.download_screen()?.to_vec();
    let screen_cpu = cpu::shade(&lay, &cloth.pos, &cloth.nrm, &scene.cam);
    let (mut max_px, mut color_diff) = (0.0f32, 0usize);
    for r in 0..lay.rows {
        for c in 0..lay.cols {
            let i = lay.index(r, c) * 4;
            max_px = max_px
                .max((screen_gpu[i] - screen_cpu[i]).abs())
                .max((screen_gpu[i + 1] - screen_cpu[i + 1]).abs());
            if screen_gpu[i + 3] != screen_cpu[i + 3] {
                color_diff += 1;
            }
        }
    }
    println!(
        "{frames} frames: worst position difference {worst:.3e} spacings, {exact_frames} frames bit-exact; last frame's projection differs by at most {max_px:.3e} px, {color_diff} of {} colors differ",
        lay.rows * lay.cols
    );
    println!(
        "mean stretch: gpu {:.3}%",
        cpu::stretch(&lay, p.download_pos(parity)?, scene.ph.spacing) * 100.0
    );
    Ok(())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn to_image(pixels: &[u32], width: usize, height: usize) -> RgbImage {
    let raw = pixels
        .iter()
        .flat_map(|p| [(p >> 16) as u8, (p >> 8) as u8, *p as u8])
        .collect();
    RgbImage::from_raw(width as u32, height as u32, raw).unwrap()
}
