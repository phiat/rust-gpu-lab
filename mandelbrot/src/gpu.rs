//! GPU renderer: cuTile tile kernels plus the host code that launches them.

use std::sync::Arc;

use cuda_core::Stream;
use cutile::prelude::*;

use crate::View;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// Per-pixel c = (cr, ci) for the tile this program owns.
    ///
    /// There is no per-thread pixel index. A tile program knows its block
    /// id, so it derives the pixel coordinates of its whole BH x BW block:
    /// a 1-D row vector and a 1-D column vector, broadcast to 2-D.
    fn tile_coords<const BH: i32, const BW: i32>(
        s: Shape<{ [BH, BW] }>,
        x0: f32,
        y0: f32,
        step: f32,
    ) -> (Tile<f32, { [BH, BW] }>, Tile<f32, { [BH, BW] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row0: i32 = pid.0 * BH;
        let col0: i32 = pid.1 * BW;

        let rows: Tile<i32, { [BH] }> = iota(shape![BH]);
        let rows: Tile<i32, { [BH] }> = rows + row0.broadcast(shape![BH]);
        let rows: Tile<f32, { [BH] }> = convert_tile(rows);
        // Row 0 is the top of the image, so imaginary part decreases with row.
        let ci: Tile<f32, { [BH] }> = y0.broadcast(shape![BH]) - rows * step.broadcast(shape![BH]);

        let cols: Tile<i32, { [BW] }> = iota(shape![BW]);
        let cols: Tile<i32, { [BW] }> = cols + col0.broadcast(shape![BW]);
        let cols: Tile<f32, { [BW] }> = convert_tile(cols);
        let cr: Tile<f32, { [BW] }> = cols * step.broadcast(shape![BW]) + x0.broadcast(shape![BW]);

        let cr: Tile<f32, { [BH, BW] }> = cr.reshape(shape![1, BW]).broadcast(s);
        let ci: Tile<f32, { [BH, BW] }> = ci.reshape(shape![BH, 1]).broadcast(s);
        (cr, ci)
    }

    /// Smooth (fractional) escape count, or -1 for points that never escaped.
    fn smooth_count<const BH: i32, const BW: i32>(
        s: Shape<{ [BH, BW] }>,
        zr: Tile<f32, { [BH, BW] }>,
        zi: Tile<f32, { [BH, BW] }>,
        n: Tile<f32, { [BH, BW] }>,
    ) -> Tile<f32, { [BH, BW] }> {
        let mag2: Tile<f32, { [BH, BW] }> = zr * zr + zi * zi;
        let limit: Tile<f32, { [BH, BW] }> = constant(256.0f32, s);
        let escaped: Tile<bool, { [BH, BW] }> = gt_tile(mag2, limit);
        // nu = n + 1 - log2(ln|z|), with ln|z| = 0.5 * ln|z|^2.
        // Interior pixels produce NaN here; select discards them.
        let half: Tile<f32, { [BH, BW] }> = constant(0.5f32, s);
        let one: Tile<f32, { [BH, BW] }> = constant(1.0f32, s);
        let inside: Tile<f32, { [BH, BW] }> = constant(-1.0f32, s);
        let ln_mod: Tile<f32, { [BH, BW] }> = log(mag2) * half;
        let nu: Tile<f32, { [BH, BW] }> = n + one - log2(ln_mod);
        select(escaped, nu, inside)
    }

    // Launchers return every argument back as a tuple, and `.first()` on
    // that tuple only exists up to 6 elements, so the entry points keep
    // their scalar lists short (square pixels -> a single `step`).

    /// Fixed-iteration Mandelbrot.
    ///
    /// Every pixel in the tile runs all `max_iter` steps. Pixels that
    /// escape are frozen with `select` instead of branching away: a tile
    /// program computes on the whole block at once, so there is no
    /// per-pixel `if`.
    #[cutile::entry()]
    pub fn mandelbrot_fixed<const BH: i32, const BW: i32>(
        out: &mut Tensor<f32, { [BH, BW] }>,
        x0: f32,
        y0: f32,
        step: f32,
        max_iter: i32,
    ) {
        let s: Shape<{ [BH, BW] }> = out.shape();
        let (cr, ci) = tile_coords(s, x0, y0, step);
        let limit: Tile<f32, { [BH, BW] }> = constant(256.0f32, s);
        let one: Tile<f32, { [BH, BW] }> = constant(1.0f32, s);

        let mut zr: Tile<f32, { [BH, BW] }> = constant(0.0f32, s);
        let mut zi: Tile<f32, { [BH, BW] }> = constant(0.0f32, s);
        let mut n: Tile<f32, { [BH, BW] }> = constant(0.0f32, s);
        for _step in 0i32..max_iter {
            let zr2: Tile<f32, { [BH, BW] }> = zr * zr;
            let zi2: Tile<f32, { [BH, BW] }> = zi * zi;
            let alive: Tile<bool, { [BH, BW] }> = le_tile(zr2 + zi2, limit);
            let zi_next: Tile<f32, { [BH, BW] }> = (zr + zr) * zi + ci;
            let zr_next: Tile<f32, { [BH, BW] }> = zr2 - zi2 + cr;
            zr = select(alive, zr_next, zr);
            zi = select(alive, zi_next, zi);
            n = select(alive, n + one, n);
        }
        out.store(smooth_count(s, zr, zi, n));
    }

    /// Same math, but a tile stops as soon as every one of its pixels has
    /// escaped. Early exit is per *tile*, not per pixel: tiles fully
    /// outside the set finish in a few steps, tiles touching the set still
    /// run to `max_iter`.
    ///
    /// Loop shape matters here. A single `while` with the check inside
    /// its body ran ~2.7x slower than `mandelbrot_fixed` even when the check
    /// never fired, so the hot loop stays a counted `for` over
    /// `check_every` steps and only the outer loop can `break` (which
    /// `for` doesn't allow anyway).
    #[cutile::entry()]
    pub fn mandelbrot_early_exit<const BH: i32, const BW: i32>(
        out: &mut Tensor<f32, { [BH, BW] }>,
        x0: f32,
        y0: f32,
        step: f32,
        max_iter: i32,
        check_every: i32,
    ) {
        let s: Shape<{ [BH, BW] }> = out.shape();
        let (cr, ci) = tile_coords(s, x0, y0, step);
        let limit: Tile<f32, { [BH, BW] }> = constant(256.0f32, s);
        let one: Tile<f32, { [BH, BW] }> = constant(1.0f32, s);
        let zero: Tile<f32, { [BH, BW] }> = constant(0.0f32, s);

        let mut zr: Tile<f32, { [BH, BW] }> = zero;
        let mut zi: Tile<f32, { [BH, BW] }> = zero;
        let mut n: Tile<f32, { [BH, BW] }> = zero;
        let mut done: i32 = 0i32;
        while done < max_iter {
            let chunk: i32 = min(check_every, max_iter - done);
            for _step in 0i32..chunk {
                let zr2: Tile<f32, { [BH, BW] }> = zr * zr;
                let zi2: Tile<f32, { [BH, BW] }> = zi * zi;
                let alive: Tile<bool, { [BH, BW] }> = le_tile(zr2 + zi2, limit);
                let zi_next: Tile<f32, { [BH, BW] }> = (zr + zr) * zi + ci;
                let zr_next: Tile<f32, { [BH, BW] }> = zr2 - zi2 + cr;
                zr = select(alive, zr_next, zr);
                zi = select(alive, zi_next, zi);
                n = select(alive, n + one, n);
            }
            done = done + chunk;

            // Reduce the 2-D mask to one scalar: is anything still alive?
            let mag2: Tile<f32, { [BH, BW] }> = zr * zr + zi * zi;
            let alive: Tile<bool, { [BH, BW] }> = le_tile(mag2, limit);
            let alive_f: Tile<f32, { [BH, BW] }> = select(alive, one, zero);
            let row_any: Tile<f32, { [BH] }> = reduce_max(alive_f, 1i32);
            let any: Tile<f32, { [] }> = reduce_max(row_any, 0i32);
            let any: f32 = tile_to_scalar(any);
            if any < 0.5f32 {
                break;
            }
        }
        out.store(smooth_count(s, zr, zi, n));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Kernel {
    Fixed,
    EarlyExit,
}

pub struct Gpu {
    stream: Arc<Stream>,
}

impl Gpu {
    pub fn new() -> Result<Self, Error> {
        let device = Device::new(0)?;
        let stream = device.new_stream()?;
        Ok(Self { stream })
    }

    /// Render one frame and copy it back to the host.
    ///
    /// The whole pipeline (allocate -> partition -> kernel -> unpartition
    /// -> download) is built lazily and synced once at the end.
    pub fn render(&self, view: &View, kernel: Kernel, tile: usize) -> Result<Vec<f32>, Error> {
        let out = api::zeros::<f32>(&[view.height, view.width]).partition([tile, tile]);
        let (x0, y0, step) = view.pixel_mapping();
        let iters = view.max_iter as i32;
        let pixels = match kernel {
            Kernel::Fixed => kernels::mandelbrot_fixed(out, x0, y0, step, iters)
                .first()
                .unpartition()
                .to_host_vec()
                .sync_on(&self.stream)?,
            Kernel::EarlyExit => {
                kernels::mandelbrot_early_exit(out, x0, y0, step, iters, view.check_every)
                    .first()
                    .unpartition()
                    .to_host_vec()
                    .sync_on(&self.stream)?
            }
        };
        Ok(pixels)
    }

    /// Run the kernel into a reusable device buffer without downloading.
    /// Used by the benchmark to time GPU work on its own.
    pub fn render_on_device(
        &self,
        buf: Tensor<f32>,
        view: &View,
        kernel: Kernel,
        tile: usize,
    ) -> Result<Tensor<f32>, Error> {
        let (x0, y0, step) = view.pixel_mapping();
        let iters = view.max_iter as i32;
        let out = buf.partition([tile, tile]);
        let buf = match kernel {
            Kernel::Fixed => kernels::mandelbrot_fixed(out, x0, y0, step, iters)
                .first()
                .unpartition()
                .sync_on(&self.stream)?,
            Kernel::EarlyExit => {
                kernels::mandelbrot_early_exit(out, x0, y0, step, iters, view.check_every)
                    .first()
                    .unpartition()
                    .sync_on(&self.stream)?
            }
        };
        Ok(buf)
    }

    pub fn alloc(&self, view: &View) -> Result<Tensor<f32>, Error> {
        Ok(api::zeros::<f32>(&[view.height, view.width]).sync_on(&self.stream)?)
    }

    pub fn download(&self, buf: Tensor<f32>) -> Result<Vec<f32>, Error> {
        Ok(buf.to_host_vec().sync_on(&self.stream)?)
    }
}
