mod cpu;
mod gpu;
mod scene;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use cutile::prelude::*;
use image::RgbImage;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};

use gpu::{Quality, Renderer};
use scene::{Camera, Frame, View};

#[derive(Parser)]
#[command(about = "Real-time SDF ray marcher written as a cuTile tile kernel")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Open a window and fly around the scene.
    ///
    /// Drag or arrow keys: orbit. Scroll or +/-: zoom. Space: pause
    /// animation. O: auto-orbit. V: cycle view (shaded, march steps, tile
    /// work). 1/2/3: quality. P: save a screenshot. Esc: quit.
    Run {
        #[arg(long, default_value_t = 1280)]
        width: usize,
        #[arg(long, default_value_t = 720)]
        height: usize,
        /// Window pixels per rendered pixel: 1, 2 or 4.
        #[arg(long, default_value_t = 1)]
        scale: usize,
        /// 1 = low, 2 = medium, 3 = high.
        #[arg(long, default_value_t = 2)]
        quality: usize,
    },
    /// Render one frame to a PNG and compare it with the CPU renderer.
    Render {
        #[command(flatten)]
        shot: Shot,
        #[arg(long, default_value = "raymarch.png")]
        out: PathBuf,
        /// Skip the CPU reference render.
        #[arg(long)]
        no_compare: bool,
    },
    /// Time CPU rayon, GPU eager, GPU CUDA graph, and GPU without early exit.
    Bench {
        #[command(flatten)]
        shot: Shot,
        #[arg(long, default_value_t = 200)]
        frames: usize,
        #[arg(long, default_value_t = 3)]
        cpu_frames: usize,
    },
}

#[derive(Args)]
struct Shot {
    #[arg(long, default_value_t = 1280)]
    width: usize,
    #[arg(long, default_value_t = 720)]
    height: usize,
    /// Animation time in seconds.
    #[arg(long, default_value_t = 0.0)]
    time: f32,
    /// Camera angle around the scene, degrees.
    #[arg(long, allow_hyphen_values = true)]
    yaw: Option<f32>,
    /// Camera angle above the horizon, degrees.
    #[arg(long, allow_hyphen_values = true)]
    pitch: Option<f32>,
    #[arg(long)]
    distance: Option<f32>,
    #[arg(long, value_enum, default_value_t = View::Shaded)]
    view: View,
    /// 1 = low, 2 = medium, 3 = high.
    #[arg(long, default_value_t = 2)]
    quality: usize,
}

impl Shot {
    fn frame(&self) -> Frame {
        let mut camera = Camera::default();
        if let Some(yaw) = self.yaw {
            camera.yaw = yaw.to_radians();
        }
        if let Some(pitch) = self.pitch {
            camera.pitch = pitch.to_radians();
        }
        if let Some(distance) = self.distance {
            camera.distance = distance;
        }
        Frame {
            width: self.width,
            height: self.height,
            camera,
            time: self.time,
            view: self.view,
            quality: preset(self.quality),
        }
    }
}

fn preset(level: usize) -> Quality {
    Quality::PRESETS[level.clamp(1, 3) - 1]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // tileiras takes ~30 s to compile this kernel, so keep the compiled
    // cubin on disk (~/.cache/cutile/kernels) across runs. Editing the
    // kernel changes its key and pays the compile again.
    cutile::jit_cache::enable(std::sync::Arc::new(
        cutile::jit_cache::FileSystemJitStore::default_location()?,
    ));
    match Cli::parse().cmd {
        Cmd::Run {
            width,
            height,
            scale,
            quality,
        } => run(width, height, scale, preset(quality)),
        Cmd::Render {
            shot,
            out,
            no_compare,
        } => render(&shot, &out, !no_compare),
        Cmd::Bench {
            shot,
            frames,
            cpu_frames,
        } => bench(&shot, frames, cpu_frames),
    }
}

fn run(
    width: usize,
    height: usize,
    scale: usize,
    quality: Quality,
) -> Result<(), Box<dyn std::error::Error>> {
    let scale = match scale {
        1 => Scale::X1,
        2 => Scale::X2,
        4 => Scale::X4,
        _ => return Err("--scale must be 1, 2 or 4".into()),
    };
    let mut frame = Frame {
        width,
        height,
        camera: Camera::default(),
        time: 0.0,
        view: View::Shaded,
        quality,
    };

    let stream = Device::new(0)?.new_stream()?;
    let mut renderer = Renderer::new(&stream, width, height, quality)?;
    renderer.set_params(&frame.params())?;
    println!("compiling the kernel (about 30 s the first time, cached on disk after)...");
    renderer.render_eager()?;
    let mut graph = renderer.capture()?;

    let mut window = Window::new(
        "raymarch",
        width,
        height,
        WindowOptions {
            scale,
            ..WindowOptions::default()
        },
    )?;
    window.set_target_fps(0);

    let (mut paused, mut auto_orbit) = (false, true);
    let mut last_mouse: Option<(f32, f32)> = None;
    let mut last = Instant::now();
    let mut stats = Stats::default();
    let mut last_title = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let dt = last.elapsed().as_secs_f32();
        last = Instant::now();

        // Input.
        for key in window.get_keys_pressed(KeyRepeat::No) {
            match key {
                Key::Space => paused = !paused,
                Key::O => auto_orbit = !auto_orbit,
                Key::V => frame.view = frame.view.next(),
                Key::Key1 | Key::Key2 | Key::Key3 => {
                    let level = match key {
                        Key::Key1 => 1,
                        Key::Key2 => 2,
                        _ => 3,
                    };
                    // Loop bounds are kernel scalars, baked into the graph.
                    frame.quality = preset(level);
                    renderer.quality = frame.quality;
                    graph = renderer.capture()?;
                }
                Key::P => {
                    let path = format!("raymarch-{:.2}.png", frame.time);
                    to_image(&renderer.download()?, width, height).save(&path)?;
                    println!("\nsaved {path}");
                }
                _ => {}
            }
        }
        let cam = &mut frame.camera;
        let mouse = window.get_mouse_pos(MouseMode::Pass);
        if window.get_mouse_down(MouseButton::Left) {
            if let (Some((x, y)), Some((lx, ly))) = (mouse, last_mouse) {
                cam.yaw -= (x - lx) * 0.006;
                cam.pitch += (y - ly) * 0.006;
                auto_orbit = false;
            }
        }
        last_mouse = mouse;
        let turn = 1.5 * dt;
        if window.is_key_down(Key::Left) {
            cam.yaw += turn;
        }
        if window.is_key_down(Key::Right) {
            cam.yaw -= turn;
        }
        if window.is_key_down(Key::Up) {
            cam.pitch += turn;
        }
        if window.is_key_down(Key::Down) {
            cam.pitch -= turn;
        }
        if let Some((_, scroll)) = window.get_scroll_wheel() {
            cam.distance *= 1.0 - scroll.signum() * 0.08;
        }
        if window.is_key_down(Key::Equal) || window.is_key_down(Key::NumPadPlus) {
            cam.distance *= 1.0 - dt;
        }
        if window.is_key_down(Key::Minus) || window.is_key_down(Key::NumPadMinus) {
            cam.distance *= 1.0 + dt;
        }
        if auto_orbit {
            cam.yaw += 0.2 * dt;
        }
        cam.pitch = cam.pitch.clamp(0.02, 1.45);
        // Stay inside the inner ring of columns (radius 6.5).
        cam.distance = cam.distance.clamp(1.5, 6.0);
        if !paused {
            frame.time += dt;
        }

        // Render: new parameters into the fixed buffer, replay the graph,
        // download the packed pixels straight into the window.
        let t = Instant::now();
        renderer.set_params(&frame.params())?;
        stats.params += t.elapsed();
        let t = Instant::now();
        graph.launch().sync_on(renderer.stream())?;
        stats.render += t.elapsed();
        let t = Instant::now();
        let pixels = renderer.download()?;
        stats.download += t.elapsed();
        window.update_with_buffer(&pixels, width, height)?;
        stats.frames += 1;

        if last_title.elapsed() >= Duration::from_secs(1) {
            let n = stats.frames.max(1);
            let title = format!(
                "raymarch {}x{} {:?} | {:.0} fps | params {:.2} + render {:.2} + download {:.2} ms | {} steps{}",
                width,
                height,
                frame.view,
                stats.frames as f64 / last_title.elapsed().as_secs_f64(),
                ms(stats.params / n),
                ms(stats.render / n),
                ms(stats.download / n),
                frame.quality.max_steps,
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
    params: Duration,
    render: Duration,
    download: Duration,
}

fn render(shot: &Shot, out: &Path, compare: bool) -> Result<(), Box<dyn std::error::Error>> {
    let frame = shot.frame();
    let params = frame.params();
    let stream = Device::new(0)?.new_stream()?;
    let mut r = Renderer::new(&stream, frame.width, frame.height, frame.quality)?;
    r.set_params(&params)?;
    let t = Instant::now();
    r.render_eager()?;
    let first = t.elapsed();
    let t = Instant::now();
    r.render_eager()?;
    let warm = t.elapsed();
    let pixels = r.download()?;
    println!(
        "{}x{} {:?}: first render {:.0} ms (JIT), warm {:.2} ms",
        frame.width,
        frame.height,
        frame.view,
        ms(first),
        ms(warm)
    );
    to_image(&pixels, frame.width, frame.height).save(out)?;
    println!("wrote {}", out.display());

    if compare && frame.view == View::Shaded {
        let t = Instant::now();
        let reference = cpu::render(&params, frame.width, frame.height, frame.quality);
        println!("cpu rayon {:.1} ms", ms(t.elapsed()));
        println!("vs cpu: {}", Diff::new(&pixels, &reference));
    }
    Ok(())
}

fn bench(shot: &Shot, frames: usize, cpu_frames: usize) -> Result<(), Box<dyn std::error::Error>> {
    let frame = shot.frame();
    let params = frame.params();
    let q = frame.quality;
    println!(
        "{}x{}, {} march steps (check every {}), {} shadow steps",
        frame.width, frame.height, q.max_steps, q.check_every, q.shadow_steps
    );

    let t = Instant::now();
    let mut reference = Vec::new();
    for _ in 0..cpu_frames {
        reference = cpu::render(&params, frame.width, frame.height, q);
    }
    let cpu_frame = t.elapsed() / cpu_frames.max(1) as u32;
    report(
        &format!("cpu rayon ({} threads)", rayon::current_num_threads()),
        cpu_frame,
        cpu_frame,
    );

    let stream = Device::new(0)?.new_stream()?;
    let mut r = Renderer::new(&stream, frame.width, frame.height, q)?;
    r.set_params(&params)?;
    r.render_eager()?; // JIT
                       // Let the GPU leave its idle power state before timing anything.
    for _ in 0..50 {
        r.render_eager()?;
    }

    let t = Instant::now();
    for _ in 0..frames {
        r.render_eager()?;
    }
    report("gpu eager", t.elapsed() / frames as u32, cpu_frame);
    println!("    vs cpu: {}", Diff::new(&r.download()?, &reference));

    let graph = r.capture()?;
    let t = Instant::now();
    for _ in 0..frames {
        graph.launch().sync_on(r.stream())?;
    }
    report("gpu graph", t.elapsed() / frames as u32, cpu_frame);

    // Same frame with a single check at the very end: every tile runs
    // max_steps, as if there were no early exit.
    r.quality = Quality {
        check_every: q.max_steps,
        ..q
    };
    let graph = r.capture()?;
    let t = Instant::now();
    for _ in 0..frames {
        graph.launch().sync_on(r.stream())?;
    }
    report(
        "gpu graph, no early exit",
        t.elapsed() / frames as u32,
        cpu_frame,
    );

    let t = Instant::now();
    for _ in 0..frames {
        r.set_params(&params)?;
    }
    let upload = t.elapsed() / frames as u32;
    let t = Instant::now();
    for _ in 0..frames {
        r.download()?;
    }
    let download = t.elapsed() / frames as u32;
    println!(
        "per frame: params upload {:.3} ms, download {:.2} ms",
        ms(upload),
        ms(download)
    );
    Ok(())
}

fn report(label: &str, per_frame: Duration, cpu_frame: Duration) {
    println!(
        "{label:<26} {:>8.2} ms/frame {:>7.0} fps {:>7.1}x cpu",
        ms(per_frame),
        1.0 / per_frame.as_secs_f64(),
        cpu_frame.as_secs_f64() / per_frame.as_secs_f64()
    );
}

/// Per-channel differences between two packed 0RGB images.
struct Diff {
    max: u32,
    mean: f64,
    over_8: f64,
}

impl Diff {
    fn new(a: &[u32], b: &[u32]) -> Self {
        let (mut max, mut sum, mut over) = (0u32, 0u64, 0usize);
        for (&x, &y) in a.iter().zip(b) {
            let mut worst = 0;
            for shift in [0, 8, 16] {
                let d = ((x >> shift) & 255).abs_diff((y >> shift) & 255);
                sum += d as u64;
                worst = worst.max(d);
            }
            max = max.max(worst);
            over += (worst > 8) as usize;
        }
        Diff {
            max,
            mean: sum as f64 / (a.len() * 3) as f64,
            over_8: over as f64 * 100.0 / a.len() as f64,
        }
    }
}

impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "mean channel diff {:.3}, max {}, {:.3}% of pixels off by more than 8",
            self.mean, self.max, self.over_8
        )
    }
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
