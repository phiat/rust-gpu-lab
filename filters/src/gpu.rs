//! GPU filter chain: grayscale -> separable 5x5 blur (N passes) -> Sobel.
//!
//! Every stencil stage is a "valid" convolution: output pixel (r, c) reads
//! input pixels (r + dr, c + dc) with 0 <= dr, dc <= 2R, so the output is
//! 2R smaller than its input. The host pads the source image once by the
//! total radius of the chain, and the final edge map comes out at the
//! original size. No read ever needs a negative offset, so each neighbor is
//! a plain host view that lines up with the output tile via `load_like`.

use std::sync::Arc;

use cuda_core::Stream;
use cutile::prelude::*;

use tilekit::{Eager, Pinned, Submit};

use crate::Dims;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    // JIT workaround (cutile d92c160): `+ - *` require both operands to have
    // the same tile type *token for token*, and the compiler doesn't agree
    // with itself on what that type is. Nested arithmetic, `constant` and
    // `exti` results keep the generic `{[B, B]}`; reductions and device
    // function results get the concrete `{[32, 32]}`. Each kernel below
    // keeps its arithmetic on one side: grayscale is concrete (reduce_sum +
    // `fill`), the stencils are generic (`exti` + `constant`).

    /// `v` in every cell of a B x B tile, with a concrete tile type.
    fn fill<const B: i32>(v: i32, s: Shape<{ [B, B] }>) -> Tile<i32, { [B, B] }> {
        broadcast_scalar(v, s)
    }

    /// RGBA -> luma with BT.601 integer weights: (77 R + 150 G + 29 B + 128) >> 8.
    ///
    /// The input is RGBA rather than RGB because Tile IR requires every tile
    /// dimension to be a power of two: a [B, B, 3] tile is rejected.
    #[cutile::entry()]
    pub fn grayscale<const B: i32>(
        out: &mut Tensor<u8, { [B, B] }>,
        rgba: &Tensor<u8, { [-1, -1, 4] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part = rgba.partition(shape![B, B, 4]);
        let px: Tile<u8, { [B, B, 4] }> = part.load([pid.0, pid.1, 0i32]);
        // Integer width changes use exti/trunci; convert_tile only handles
        // int <-> float and float <-> float.
        let px: Tile<i32, { [B, B, 4] }> = exti(px);

        // Weights as a [4] tile, broadcast over the pixels, then summed over
        // the channel axis.
        let w_r: Tile<i32, { [1] }> = constant(77i32, shape![1]);
        let w_g: Tile<i32, { [1] }> = constant(150i32, shape![1]);
        let w_b: Tile<i32, { [1] }> = constant(29i32, shape![1]);
        let w_a: Tile<i32, { [1] }> = constant(0i32, shape![1]);
        let w_rg: Tile<i32, { [2] }> = cat(w_r, w_g, 0i32);
        let w_ba: Tile<i32, { [2] }> = cat(w_b, w_a, 0i32);
        let w: Tile<i32, { [4] }> = cat(w_rg, w_ba, 0i32);
        let w: Tile<i32, { [B, B, 4] }> = w.reshape(shape![1, 1, 4]).broadcast(shape![B, B, 4]);
        let weighted: Tile<i32, { [B, B, 4] }> = px * w;
        let luma: Tile<i32, { [B, B] }> = reduce_sum(weighted, 2i32);

        // Constants take their shape from a computed tile. Built from
        // `shape![B, B]` or `out.shape()`, they keep the generic type
        // `{[B, B]}`, and the JIT then rejects `luma + half` as a type
        // mismatch against the concrete `{[64, 64]}`.
        let half: Tile<i32, { [B, B] }> = fill(128i32, out.shape());
        let eight: Tile<i32, { [B, B] }> = fill(8i32, out.shape());
        let luma: Tile<i32, { [B, B] }> = shri(luma + half, eight);
        let luma: Tile<u8, { [B, B] }> = trunci(luma, overflow::NoWrap);
        out.store(luma);
    }

    /// Horizontal pass of the 5-tap binomial [1 4 6 4 1], unnormalized
    /// (u8 -> u16, max 16 * 255). `c0..c4` are views of the input at column
    /// offsets 0..4, each exactly the shape of `out`.
    #[cutile::entry()]
    pub fn blur_h<const B: i32>(
        out: &mut Tensor<u16, { [B, B] }>,
        c0: &Tensor<u8, { [-1, -1] }>,
        c1: &Tensor<u8, { [-1, -1] }>,
        c2: &Tensor<u8, { [-1, -1] }>,
        c3: &Tensor<u8, { [-1, -1] }>,
        c4: &Tensor<u8, { [-1, -1] }>,
    ) {
        let t0: Tile<u8, { [B, B] }> = c0.load_like(out);
        let t1: Tile<u8, { [B, B] }> = c1.load_like(out);
        let t2: Tile<u8, { [B, B] }> = c2.load_like(out);
        let t3: Tile<u8, { [B, B] }> = c3.load_like(out);
        let t4: Tile<u8, { [B, B] }> = c4.load_like(out);
        let x0: Tile<i32, { [B, B] }> = exti(t0);
        let x1: Tile<i32, { [B, B] }> = exti(t1);
        let x2: Tile<i32, { [B, B] }> = exti(t2);
        let x3: Tile<i32, { [B, B] }> = exti(t3);
        let x4: Tile<i32, { [B, B] }> = exti(t4);

        let four: Tile<i32, { [B, B] }> = constant(4i32, shape![B, B]);
        let six: Tile<i32, { [B, B] }> = constant(6i32, shape![B, B]);
        let sum: Tile<i32, { [B, B] }> = x0 + x4 + (x1 + x3) * four + x2 * six;
        let sum: Tile<u16, { [B, B] }> = trunci(sum, overflow::NoWrap);
        out.store(sum);
    }

    /// Vertical pass (u16 -> u8), normalizing by the combined weight 256.
    /// `r0..r4` are views of the horizontal result at row offsets 0..4.
    #[cutile::entry()]
    pub fn blur_v<const B: i32>(
        out: &mut Tensor<u8, { [B, B] }>,
        r0: &Tensor<u16, { [-1, -1] }>,
        r1: &Tensor<u16, { [-1, -1] }>,
        r2: &Tensor<u16, { [-1, -1] }>,
        r3: &Tensor<u16, { [-1, -1] }>,
        r4: &Tensor<u16, { [-1, -1] }>,
    ) {
        let t0: Tile<u16, { [B, B] }> = r0.load_like(out);
        let t1: Tile<u16, { [B, B] }> = r1.load_like(out);
        let t2: Tile<u16, { [B, B] }> = r2.load_like(out);
        let t3: Tile<u16, { [B, B] }> = r3.load_like(out);
        let t4: Tile<u16, { [B, B] }> = r4.load_like(out);
        let x0: Tile<i32, { [B, B] }> = exti(t0);
        let x1: Tile<i32, { [B, B] }> = exti(t1);
        let x2: Tile<i32, { [B, B] }> = exti(t2);
        let x3: Tile<i32, { [B, B] }> = exti(t3);
        let x4: Tile<i32, { [B, B] }> = exti(t4);

        let four: Tile<i32, { [B, B] }> = constant(4i32, shape![B, B]);
        let six: Tile<i32, { [B, B] }> = constant(6i32, shape![B, B]);
        let half: Tile<i32, { [B, B] }> = constant(128i32, shape![B, B]);
        let eight: Tile<i32, { [B, B] }> = constant(8i32, shape![B, B]);
        let sum: Tile<i32, { [B, B] }> = x0 + x4 + (x1 + x3) * four + x2 * six;
        let avg: Tile<i32, { [B, B] }> = shri(sum + half, eight);
        let avg: Tile<u8, { [B, B] }> = trunci(avg, overflow::NoWrap);
        out.store(avg);
    }

    /// Sobel gradient magnitude, approximated as min(|gx| + |gy|, 255).
    /// Views are named by row (t/m/b: top, middle, bottom) and column
    /// (l/m/r); the center pixel has weight 0 in both kernels, so it isn't
    /// passed.
    #[cutile::entry()]
    pub fn sobel<const B: i32>(
        out: &mut Tensor<u8, { [B, B] }>,
        tl: &Tensor<u8, { [-1, -1] }>,
        tm: &Tensor<u8, { [-1, -1] }>,
        tr: &Tensor<u8, { [-1, -1] }>,
        ml: &Tensor<u8, { [-1, -1] }>,
        mr: &Tensor<u8, { [-1, -1] }>,
        bl: &Tensor<u8, { [-1, -1] }>,
        bm: &Tensor<u8, { [-1, -1] }>,
        br: &Tensor<u8, { [-1, -1] }>,
    ) {
        let t_tl: Tile<u8, { [B, B] }> = tl.load_like(out);
        let t_tm: Tile<u8, { [B, B] }> = tm.load_like(out);
        let t_tr: Tile<u8, { [B, B] }> = tr.load_like(out);
        let t_ml: Tile<u8, { [B, B] }> = ml.load_like(out);
        let t_mr: Tile<u8, { [B, B] }> = mr.load_like(out);
        let t_bl: Tile<u8, { [B, B] }> = bl.load_like(out);
        let t_bm: Tile<u8, { [B, B] }> = bm.load_like(out);
        let t_br: Tile<u8, { [B, B] }> = br.load_like(out);
        let x_tl: Tile<i32, { [B, B] }> = exti(t_tl);
        let x_tm: Tile<i32, { [B, B] }> = exti(t_tm);
        let x_tr: Tile<i32, { [B, B] }> = exti(t_tr);
        let x_ml: Tile<i32, { [B, B] }> = exti(t_ml);
        let x_mr: Tile<i32, { [B, B] }> = exti(t_mr);
        let x_bl: Tile<i32, { [B, B] }> = exti(t_bl);
        let x_bm: Tile<i32, { [B, B] }> = exti(t_bm);
        let x_br: Tile<i32, { [B, B] }> = exti(t_br);

        let two: Tile<i32, { [B, B] }> = constant(2i32, shape![B, B]);
        let max: Tile<i32, { [B, B] }> = constant(255i32, shape![B, B]);
        let gx: Tile<i32, { [B, B] }> = (x_tr + x_mr * two + x_br) - (x_tl + x_ml * two + x_bl);
        let gy: Tile<i32, { [B, B] }> = (x_bl + x_bm * two + x_br) - (x_tl + x_tm * two + x_tr);
        let mag: Tile<i32, { [B, B] }> = absi(gx) + absi(gy);
        let mag: Tile<i32, { [B, B] }> = min_tile(mag, max);
        let mag: Tile<u8, { [B, B] }> = trunci(mag, overflow::NoWrap);
        out.store(mag);
    }
}

/// Views of `src` at column offsets 0..=4, each `rows x (cols - 4)`.
fn col_views(src: &Tensor<u8>) -> Result<[TensorView<'_, u8>; 5], Error> {
    let (rows, cols) = (src.shape()[0] as usize, src.shape()[1] as usize);
    let v = |dc: usize| src.slice(&[0..rows, dc..dc + cols - 4]);
    Ok([v(0)?, v(1)?, v(2)?, v(3)?, v(4)?])
}

/// Views of `src` at row offsets 0..=4, each `(rows - 4) x cols`.
fn row_views(src: &Tensor<u16>) -> Result<[TensorView<'_, u16>; 5], Error> {
    let (rows, cols) = (src.shape()[0] as usize, src.shape()[1] as usize);
    let v = |dr: usize| src.slice(&[dr..dr + rows - 4, 0..cols]);
    Ok([v(0)?, v(1)?, v(2)?, v(3)?, v(4)?])
}

/// The eight non-center 3x3 views of `src`, each `(rows - 2) x (cols - 2)`.
fn sobel_views(src: &Tensor<u8>) -> Result<[TensorView<'_, u8>; 8], Error> {
    let (rows, cols) = (src.shape()[0] as usize, src.shape()[1] as usize);
    let v = |dr: usize, dc: usize| src.slice(&[dr..dr + rows - 2, dc..dc + cols - 2]);
    Ok([
        v(0, 0)?,
        v(0, 1)?,
        v(0, 2)?,
        v(1, 0)?,
        v(1, 2)?,
        v(2, 0)?,
        v(2, 1)?,
        v(2, 2)?,
    ])
}

/// Preallocated buffers for every stage, so the chain can be captured as a
/// CUDA graph and replayed per frame.
pub struct Pipeline {
    pub dims: Dims,
    tile: usize,
    stream: Arc<Stream>,
    /// Padded source, [rows + 2R, cols + 2R, 4] (RGBA).
    rgba: Tensor<u8>,
    gray: Tensor<u8>,
    /// Per blur pass: horizontal result (u16) and vertical result (u8).
    blur: Vec<(Tensor<u16>, Tensor<u8>)>,
    edges: Tensor<u8>,
    /// Pinned host copies of `rgba` and `edges`: frames go in and out
    /// through these with no allocation per frame.
    frame_in: Pinned<u8>,
    frame_out: Pinned<u8>,
}

impl Pipeline {
    pub fn new(stream: &Arc<Stream>, dims: Dims, tile: usize) -> Result<Self, Error> {
        let (mut h, mut w) = (dims.padded_rows(), dims.padded_cols());
        let rgba = api::zeros::<u8>(&[h, w, 4]).sync_on(stream)?;
        let gray = api::zeros::<u8>(&[h, w]).sync_on(stream)?;
        let mut blur = Vec::with_capacity(dims.blur_passes);
        for _ in 0..dims.blur_passes {
            let mid = api::zeros::<u16>(&[h, w - 4]).sync_on(stream)?;
            let out = api::zeros::<u8>(&[h - 4, w - 4]).sync_on(stream)?;
            blur.push((mid, out));
            (h, w) = (h - 4, w - 4);
        }
        let edges = api::zeros::<u8>(&[h - 2, w - 2]).sync_on(stream)?;
        debug_assert_eq!((h - 2, w - 2), (dims.rows, dims.cols));
        Ok(Pipeline {
            dims,
            tile,
            stream: stream.clone(),
            rgba,
            gray,
            blur,
            edges,
            frame_in: Pinned::new(stream, dims.padded_rows() * dims.padded_cols() * 4)?,
            frame_out: Pinned::new(stream, dims.rows * dims.cols)?,
        })
    }

    /// The padded RGBA frame to upload next. Write pixels straight into it
    /// (a decoder or camera would), then call `upload`.
    pub fn frame_in(&mut self) -> &mut [u8] {
        self.frame_in.as_mut_slice()
    }

    /// Copy `frame_in` into the input buffer. The buffer is reused, so a
    /// captured graph stays valid.
    pub fn upload(&mut self) -> Result<(), Error> {
        self.frame_in.upload(&mut self.rgba, &self.stream)
    }

    /// The slow way, for comparison: allocate a device tensor, copy from
    /// pageable memory, then copy device to device into the input buffer.
    pub fn upload_pageable(&mut self, padded_rgba: Vec<u8>) -> Result<(), Error> {
        let shape = [self.dims.padded_rows(), self.dims.padded_cols(), 4];
        let src = api::copy_host_vec_to_device(&Arc::new(padded_rgba))
            .sync_on(&self.stream)?
            .reshape(&shape)?;
        api::memcpy(&mut self.rgba, &src).sync_on(&self.stream)?;
        Ok(())
    }

    /// Submit every stage in order.
    pub fn run(&mut self, sub: &impl Submit) -> Result<(), Error> {
        let t = self.tile;
        sub.submit(kernels::grayscale(
            (&mut self.gray).partition([t, t]),
            &self.rgba,
        ))?;

        let mut src: &Tensor<u8> = &self.gray;
        for (mid, dst) in self.blur.iter_mut() {
            {
                let v = col_views(src)?;
                sub.submit(kernels::blur_h(
                    mid.partition([t, t]),
                    &v[0],
                    &v[1],
                    &v[2],
                    &v[3],
                    &v[4],
                ))?;
            }
            {
                let v = row_views(mid)?;
                sub.submit(kernels::blur_v(
                    dst.partition([t, t]),
                    &v[0],
                    &v[1],
                    &v[2],
                    &v[3],
                    &v[4],
                ))?;
            }
            src = dst;
        }

        let v = sobel_views(src)?;
        sub.submit(kernels::sobel(
            (&mut self.edges).partition([t, t]),
            &v[0],
            &v[1],
            &v[2],
            &v[3],
            &v[4],
            &v[5],
            &v[6],
            &v[7],
        ))?;
        Ok(())
    }

    pub fn run_eager(&mut self) -> Result<(), Error> {
        let stream = self.stream.clone();
        self.run(&Eager(&stream))
    }

    /// Record the whole chain as one CUDA graph.
    pub fn capture(&mut self) -> Result<CudaGraph<()>, Error> {
        let stream = self.stream.clone();
        Ok(CudaGraph::scope(&stream, |s| Ok(self.run(s)?))?)
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Copy the edge map into the pinned output buffer and borrow it.
    pub fn download_edges(&mut self) -> Result<&[u8], Error> {
        self.frame_out.download(&self.edges, &self.stream)?;
        Ok(self.frame_out.as_slice())
    }

    /// The slow way, for comparison: duplicate on the device (`to_host_vec`
    /// consumes its tensor), then copy into a new pageable `Vec`.
    pub fn download_edges_pageable(&self) -> Result<Vec<u8>, Error> {
        Ok(self.edges.dup().to_host_vec().sync_on(&self.stream)?)
    }

    /// Intermediate images cropped to the original size: (gray, blurred).
    pub fn download_stages(&self) -> Result<(Vec<u8>, Vec<u8>), Error> {
        let d = self.dims;
        let gray = self.gray.dup().to_host_vec().sync_on(&self.stream)?;
        let gray = crop(&gray, d.padded_cols(), d.radius(), d.rows, d.cols);
        let (blurred, pad) = match self.blur.last() {
            Some((_, out)) => (out.dup().to_host_vec().sync_on(&self.stream)?, 1),
            None => (
                self.gray.dup().to_host_vec().sync_on(&self.stream)?,
                d.radius(),
            ),
        };
        let blurred = crop(&blurred, d.cols + 2 * pad, pad, d.rows, d.cols);
        Ok((gray, blurred))
    }
}

fn crop(buf: &[u8], buf_cols: usize, pad: usize, rows: usize, cols: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let start = (r + pad) * buf_cols + pad;
        out.extend_from_slice(&buf[start..start + cols]);
    }
    out
}
