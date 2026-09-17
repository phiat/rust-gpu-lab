mod cpu;
mod gpu;
mod palette;

use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};

use gpu::{Gpu, Kernel};

/// Squared escape radius; must match `bailout` in the GPU kernels.
pub const BAILOUT_SQ: f32 = 256.0;

#[derive(Parser)]
#[command(about = "Mandelbrot renderer: cuTile GPU kernels vs a rayon CPU baseline")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Render a PNG.
    Render {
        #[command(flatten)]
        view: ViewArgs,
        #[arg(long, default_value = "mandelbrot.png")]
        out: String,
        /// Render on the CPU (rayon) instead of the GPU.
        #[arg(long)]
        cpu: bool,
    },
    /// Time the CPU and GPU renderers and check that they agree.
    Bench {
        #[command(flatten)]
        view: ViewArgs,
        /// Warm GPU runs to take the median of.
        #[arg(long, default_value_t = 10)]
        runs: usize,
        /// Skip the single-threaded CPU run (slow at high iteration counts).
        #[arg(long)]
        skip_serial: bool,
    },
}

#[derive(Args, Clone)]
struct ViewArgs {
    #[arg(long, default_value_t = 1920)]
    width: usize,
    #[arg(long, default_value_t = 1080)]
    height: usize,
    #[arg(long, default_value_t = 1000)]
    iters: u32,
    /// Center of the view as "re,im".
    #[arg(long, default_value = "-0.65,0.0", allow_hyphen_values = true)]
    center: String,
    /// Width of the view in the complex plane.
    #[arg(long, default_value_t = 3.2)]
    span: f64,
    /// Tile edge length in pixels. Each tile is one GPU tile program.
    /// 16-64 all perform about the same; 128+ is far slower to run and to
    /// JIT compile.
    #[arg(long, default_value_t = 64)]
    tile: usize,
    #[arg(long, value_enum, default_value_t = Kernel::EarlyExit)]
    kernel: Kernel,
    /// For --kernel early-exit: iterations between whole-tile "anything alive?" checks.
    #[arg(long, default_value_t = 16)]
    check_every: i32,
}

pub struct View {
    pub width: usize,
    pub height: usize,
    pub max_iter: u32,
    pub check_every: i32,
    center: (f64, f64),
    span: f64,
}

impl View {
    fn from_args(a: &ViewArgs) -> Self {
        let (re, im) = a
            .center
            .split_once(',')
            .expect("--center must look like re,im");
        View {
            width: a.width,
            height: a.height,
            max_iter: a.iters,
            check_every: a.check_every.max(1),
            center: (re.trim().parse().unwrap(), im.trim().parse().unwrap()),
            span: a.span,
        }
    }

    /// (x0, y0, step): pixel (row, col) maps to c = (x0 + col*step, y0 - row*step).
    /// Pixels are square; row 0 is the top of the image.
    pub fn pixel_mapping(&self) -> (f32, f32, f32) {
        let step = self.span / self.width as f64;
        let x0 = self.center.0 - self.span / 2.0 + step / 2.0;
        let y0 = self.center.1 + step * self.height as f64 / 2.0 - step / 2.0;
        (x0 as f32, y0 as f32, step as f32)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().cmd {
        Cmd::Render { view, out, cpu } => {
            let tile = view.tile;
            let kernel = view.kernel;
            let view = View::from_args(&view);
            let t = Instant::now();
            let pixels = if cpu {
                cpu::render_rayon(&view)
            } else {
                Gpu::new()?.render(&view, kernel, tile)?
            };
            let elapsed = t.elapsed();
            palette::to_image(&pixels, view.width, view.height).save(&out)?;
            println!(
                "{}x{} @ {} iters on {} in {}  ->  {out}",
                view.width,
                view.height,
                view.max_iter,
                if cpu {
                    "cpu".to_string()
                } else {
                    format!("gpu ({kernel:?})")
                },
                ms(elapsed)
            );
        }
        Cmd::Bench {
            view,
            runs,
            skip_serial,
        } => bench(&view, runs, skip_serial)?,
    }
    Ok(())
}

fn bench(
    args: &ViewArgs,
    runs: usize,
    skip_serial: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let view = View::from_args(args);
    let tile = args.tile;
    let grid = (view.height.div_ceil(tile), view.width.div_ceil(tile));
    println!(
        "{}x{} px, {} iters, tile {tile}x{tile} -> grid {}x{} = {} tile programs",
        view.width,
        view.height,
        view.max_iter,
        grid.0,
        grid.1,
        grid.0 * grid.1
    );

    let serial = if skip_serial {
        None
    } else {
        let t = Instant::now();
        let _ = cpu::render_serial(&view);
        let d = t.elapsed();
        println!("cpu 1 thread                {:>10}", ms(d));
        Some(d)
    };

    let t = Instant::now();
    let reference = cpu::render_rayon(&view);
    let rayon_time = t.elapsed();
    println!(
        "cpu rayon ({:>2} threads)      {:>10}{}",
        rayon::current_num_threads(),
        ms(rayon_time),
        serial
            .map(|s| format!("   {:.1}x vs 1 thread", ratio(s, rayon_time)))
            .unwrap_or_default()
    );

    let gpu = Gpu::new()?;
    for kernel in [Kernel::Fixed, Kernel::EarlyExit] {
        // First launch includes JIT compilation of this kernel variant.
        let t = Instant::now();
        let mut buf = gpu.alloc(&view)?;
        buf = gpu.render_on_device(buf, &view, kernel, tile)?;
        let first = t.elapsed();

        let mut times = Vec::with_capacity(runs);
        for _ in 0..runs {
            let t = Instant::now();
            buf = gpu.render_on_device(buf, &view, kernel, tile)?;
            times.push(t.elapsed());
        }
        times.sort();
        let median = times[times.len() / 2];

        let t = Instant::now();
        let pixels = gpu.download(buf)?;
        let download = t.elapsed();

        println!(
            "gpu {:<24}{:>10}   {:.1}x vs rayon   (first launch {}, download {})",
            format!("{kernel:?}"),
            ms(median),
            ratio(rayon_time, median),
            ms(first),
            ms(download)
        );
        compare(&reference, &pixels);
    }
    Ok(())
}

/// GPU and CPU run the same f32 recurrence, but the GPU compiler may fuse
/// or reorder float ops, so chaotic boundary pixels can differ slightly.
fn compare(cpu: &[f32], gpu: &[f32]) {
    let mut class_mismatch = 0usize;
    let mut off = 0usize;
    for (&c, &g) in cpu.iter().zip(gpu) {
        if (c < 0.0) != (g < 0.0) {
            class_mismatch += 1;
        } else if (c - g).abs() > 1e-2 {
            off += 1;
        }
    }
    let pct = |n: usize| 100.0 * n as f64 / cpu.len() as f64;
    println!(
        "    vs cpu: {:.4}% inside/outside disagree, {:.4}% escape counts differ by >0.01",
        pct(class_mismatch),
        pct(off)
    );
}

fn ms(d: Duration) -> String {
    format!("{:.2} ms", d.as_secs_f64() * 1e3)
}

fn ratio(a: Duration, b: Duration) -> f64 {
    a.as_secs_f64() / b.as_secs_f64()
}
