mod cpu;
mod gpu;

use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use cutile::prelude::*;
use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};

use gpu::World;

#[derive(Parser)]
#[command(about = "Conway's Game of Life on a torus: cuTile stencil kernel + CUDA graphs")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Open a window and run the simulation.
    /// Keys: Space pause, R reseed, Esc quit.
    Run {
        #[arg(long, default_value_t = 1280)]
        width: usize,
        #[arg(long, default_value_t = 720)]
        height: usize,
        /// Tile edge in cells. The world is rounded up to a multiple of it.
        #[arg(long, default_value_t = 32)]
        tile: usize,
        /// Window pixels per cell: 1, 2, 4 or 8.
        #[arg(long, default_value_t = 1)]
        scale: usize,
        /// Generations per frame, captured as one CUDA graph (rounded up to even).
        #[arg(long, default_value_t = 2)]
        gens_per_frame: usize,
        #[arg(long, default_value_t = 0.3)]
        density: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Compare CPU, GPU eager and GPU CUDA-graph throughput, and check that
    /// all three produce identical worlds.
    Bench {
        /// World edge in cells (rounded up to a multiple of --tile).
        #[arg(long, default_value_t = 4096)]
        size: usize,
        #[arg(long, default_value_t = 64)]
        tile: usize,
        #[arg(long, default_value_t = 100)]
        gens: usize,
        /// Generations per graph launch (rounded up to even).
        #[arg(long, default_value_t = 10)]
        graph_gens: usize,
        #[arg(long, default_value_t = 0.3)]
        density: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

/// World geometry. Cells live in a padded buffer with one ring of ghost
/// tiles, so logical cell (r, c) is at buffer [tile + r, tile + c].
#[derive(Clone, Copy)]
pub struct Layout {
    pub tile: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Layout {
    fn new(height: usize, width: usize, tile: usize) -> Self {
        Layout {
            tile,
            rows: height.div_ceil(tile).max(1) * tile,
            cols: width.div_ceil(tile).max(1) * tile,
        }
    }

    pub fn buf_rows(&self) -> usize {
        self.rows + 2 * self.tile
    }

    pub fn buf_cols(&self) -> usize {
        self.cols + 2 * self.tile
    }

    /// Embed `cells` in a padded buffer with ghost tiles filled by wrap-around.
    pub fn pad(&self, cells: &[u8]) -> Vec<u8> {
        let (br, bc, t) = (self.buf_rows(), self.buf_cols(), self.tile);
        let mut buf = vec![0u8; br * bc];
        for r in 0..br {
            let src_r = (r + self.rows - t % self.rows) % self.rows;
            for c in 0..bc {
                let src_c = (c + self.cols - t % self.cols) % self.cols;
                buf[r * bc + c] = cells[src_r * self.cols + src_c];
            }
        }
        buf
    }

    pub fn unpad(&self, buf: &[u8]) -> Vec<u8> {
        let (bc, t) = (self.buf_cols(), self.tile);
        let mut cells = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            let start = (t + r) * bc + t;
            cells.extend_from_slice(&buf[start..start + self.cols]);
        }
        cells
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().cmd {
        Cmd::Run {
            width,
            height,
            tile,
            scale,
            gens_per_frame,
            density,
            seed,
        } => run(
            Layout::new(height, width, tile),
            scale,
            gens_per_frame,
            density,
            seed,
        ),
        Cmd::Bench {
            size,
            tile,
            gens,
            graph_gens,
            density,
            seed,
        } => bench(
            Layout::new(size, size, tile),
            gens,
            graph_gens,
            density,
            seed,
        ),
    }
}

fn run(
    lay: Layout,
    scale: usize,
    gens_per_frame: usize,
    density: f64,
    mut seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = Device::new(0)?.new_stream()?;
    let mut world = World::new(
        &stream,
        lay,
        &cpu::random_soup(lay.rows, lay.cols, density, seed),
    )?;
    // Warm up (JIT compile) outside the graph, then restore the seed.
    world.step_eager()?;
    world.load(&cpu::random_soup(lay.rows, lay.cols, density, seed))?;

    let graph = world.capture(gens_per_frame)?;
    let gens_per_launch = gens_per_frame.div_ceil(2).max(1) * 2;

    let scale = match scale {
        1 => Scale::X1,
        2 => Scale::X2,
        4 => Scale::X4,
        8 => Scale::X8,
        _ => return Err("--scale must be 1, 2, 4 or 8".into()),
    };
    let mut window = Window::new(
        "life",
        lay.cols,
        lay.rows,
        WindowOptions {
            scale,
            ..WindowOptions::default()
        },
    )?;
    window.set_target_fps(60);

    // Per-cell brightness on the host so dying cells leave a short trail.
    let mut heat = vec![0u8; lay.rows * lay.cols];
    let mut frame = vec![0u32; lay.rows * lay.cols];
    let mut paused = false;
    let (mut gens, mut frames, mut gpu_time) = (0u64, 0u32, Duration::ZERO);
    let mut last_title = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            paused = !paused;
        }
        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            seed = seed.wrapping_add(1);
            world.load(&cpu::random_soup(lay.rows, lay.cols, density, seed))?;
        }
        if !paused {
            let t = Instant::now();
            graph.launch().sync_on(&stream)?;
            gpu_time += t.elapsed();
            gens += gens_per_launch as u64;
        }

        let cells = world.download()?;
        for ((h, px), &alive) in heat.iter_mut().zip(frame.iter_mut()).zip(&cells) {
            *h = if alive != 0 {
                255
            } else {
                h.saturating_sub(24)
            };
            *px = shade(*h);
        }
        window.update_with_buffer(&frame, lay.cols, lay.rows)?;
        frames += 1;

        if last_title.elapsed() >= Duration::from_secs(1) {
            let secs = last_title.elapsed().as_secs_f64();
            let per_launch = gpu_time.as_secs_f64() * 1e3 / frames.max(1) as f64;
            let stats = format!(
                "life {}x{} | {:.0} fps | {:.0} gens/s | graph launch {:.2} ms{}",
                lay.cols,
                lay.rows,
                frames as f64 / secs,
                gens as f64 / secs,
                per_launch,
                if paused { " | paused" } else { "" }
            );
            window.set_title(&stats);
            print!("\r{stats}   ");
            std::io::Write::flush(&mut std::io::stdout())?;
            (gens, frames, gpu_time) = (0, 0, Duration::ZERO);
            last_title = Instant::now();
        }
    }
    Ok(())
}

/// Dark blue background, cyan trail, near-white live cells.
fn shade(heat: u8) -> u32 {
    let t = heat as u32;
    let r = 8 + t * 200 / 255;
    let g = 12 + t * 240 / 255;
    let b = 28 + t * 227 / 255;
    (r << 16) | (g << 8) | b
}

fn bench(
    lay: Layout,
    gens: usize,
    graph_gens: usize,
    density: f64,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let graph_gens = graph_gens.div_ceil(2).max(1) * 2;
    let launches = gens.div_ceil(graph_gens).max(1);
    let gens = launches * graph_gens;
    let cells = lay.rows * lay.cols;
    println!(
        "{}x{} torus ({:.1}M cells), tile {}, {} generations",
        lay.cols,
        lay.rows,
        cells as f64 / 1e6,
        lay.tile,
        gens
    );
    let seed_cells = cpu::random_soup(lay.rows, lay.cols, density, seed);

    // CPU reference.
    let (mut a, mut b) = (seed_cells.clone(), vec![0u8; cells]);
    let t = Instant::now();
    for _ in 0..gens {
        cpu::step(&a, &mut b, lay.rows, lay.cols);
        std::mem::swap(&mut a, &mut b);
    }
    let cpu_time = t.elapsed();
    let reference = a;
    report(
        &format!("cpu rayon ({} threads)", rayon::current_num_threads()),
        cpu_time,
        gens,
        cells,
        None,
    );

    let stream = Device::new(0)?.new_stream()?;

    // GPU eager: one kernel launch + sync per generation.
    let mut world = World::new(&stream, lay, &seed_cells)?;
    let t = Instant::now();
    world.step_eager()?;
    let first = t.elapsed();
    let t = Instant::now();
    for _ in 1..gens {
        world.step_eager()?;
    }
    let eager_time = t.elapsed();
    report(
        "gpu eager",
        eager_time,
        gens - 1,
        cells,
        Some(cpu_time.as_secs_f64() / gens as f64),
    );
    println!(
        "    first step (JIT) {}, matches cpu: {}",
        ms(first),
        world.download()? == reference
    );

    // GPU CUDA graph: capture `graph_gens` generations once, replay.
    let mut world = World::new(&stream, lay, &seed_cells)?;
    let t = Instant::now();
    let graph = world.capture(graph_gens)?;
    let capture = t.elapsed();
    let t = Instant::now();
    for _ in 0..launches {
        graph.launch().sync_on(&stream)?;
    }
    let graph_time = t.elapsed();
    report(
        &format!("gpu graph ({graph_gens} gens/launch)"),
        graph_time,
        gens,
        cells,
        Some(cpu_time.as_secs_f64() / gens as f64),
    );
    println!(
        "    capture {}, matches cpu: {}",
        ms(capture),
        world.download()? == reference
    );
    Ok(())
}

fn report(label: &str, total: Duration, gens: usize, cells: usize, cpu_per_gen: Option<f64>) {
    let per_gen = total.as_secs_f64() / gens.max(1) as f64;
    println!(
        "{label:<28}{:>10} /gen  {:>9.0} gens/s  {:>8.2} B cells/s{}",
        ms(Duration::from_secs_f64(per_gen)),
        1.0 / per_gen,
        cells as f64 / per_gen / 1e9,
        cpu_per_gen
            .map(|c| format!("  {:.1}x vs cpu", c / per_gen))
            .unwrap_or_default()
    );
}

fn ms(d: Duration) -> String {
    format!("{:.3} ms", d.as_secs_f64() * 1e3)
}
