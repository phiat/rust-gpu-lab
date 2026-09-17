# filters

Demo #2 in tileworld: an image filter chain on the GPU. It converts to
grayscale, runs N passes of a separable 5x5 Gaussian blur, and finishes with
Sobel edge detection. Every stage is integer math, so the result is checked bit
for bit against a CPU version.

```
src/gpu.rs   kernels, Pipeline with preallocated stage buffers and pinned frame buffers
src/cpu.rs   rayon reference with identical integer math
src/main.rs  padding, synthetic test image, `run` (PNGs) and `bench` (frame stream)
```

## Run

```sh
cargo run --release -p filters -- run                      # 4K synthetic image -> filters-out/*.png
cargo run --release -p filters -- run --input photo.jpg    # any PNG/JPEG
cargo run --release -p filters -- run --blur-passes 0 --out-dir filters-out-noblur
cargo run --release -p filters -- bench                    # 30 frames: CPU vs GPU eager vs GPU graph
```

`run` writes `input.png` (synthetic source only), `gray.png`, `blurred.png`
and `edges.png`. The synthetic image has per-pixel noise on purpose. With
`--blur-passes 0`, Sobel lights up the noise everywhere. Two passes wipe it
out and leave the real edges.

## Results

3840x2160, 2 blur passes, so 6 kernels per frame. RTX 4070 Ti SUPER,
i9-14900KF (28 threads), WSL2. Times are per frame.

| path         | tile 16 | tile 32 | tile 64 |
|--------------|---------|---------|---------|
| CPU rayon    | 11.7 ms | 11.7 ms | 11.7 ms |
| GPU eager    | 0.84 ms | 0.76 ms | 0.94 ms |
| GPU graph    | 0.66 ms | 0.64 ms | 0.77 ms |

Filtering is the cheap part: moving the frame costs several times more. Per
frame, with the graph:

| transfers                         | upload  | download | end to end |
|-----------------------------------|---------|----------|------------|
| pageable `Vec`, new device tensor | 4.6 ms  | 0.8 ms   | ~170 fps   |
| pinned buffers, reused            | 3.0 ms  | 0.4 ms   | ~245 fps   |

## Transfers: pinned buffers

The first version uploaded with `api::copy_host_vec_to_device`. Per frame that
clones the 33 MB frame into a new `Vec`, allocates a device tensor, copies
from pageable memory (which the driver first stages through its own pinned
buffer), then copies device to device into the buffer the graph reads.
Downloads did the mirror image: `dup()` on the device, because
`to_host_vec()` consumes its tensor, then a copy into a new `Vec`.

`tilekit::Pinned` replaces both. It allocates page-locked host memory once
(`cuMemAllocHost`), which the GPU can read and write directly, and each
transfer is one `cuMemcpy*Async` between it and the tensor the kernels
already use. cuTile has no safe wrapper for this yet, so it goes through
`cuda-core` with `tensor.device_pointer()` in a few lines of `unsafe`.

Of the 3.0 ms upload, 1.3 ms is the benchmark copying the frame into the
pinned buffer; the transfer itself runs at about 20 GB/s. A decoder or camera
writing straight into `Pipeline::frame_in()` would skip that copy.

## How the stencils work: "valid" convolution

Game of Life needed ghost tiles because its state wraps around and has to stay
the same size step after step. A filter chain doesn't need that:

- The host pads the source once by the chain's total radius, repeating edge
  pixels: 2 per blur pass plus 1 for Sobel.
- Each stencil stage is a *valid* convolution. Output (r, c) reads input
  (r + dr, c + dc) with 0 <= dr, dc <= 2R, so its output is 2R smaller than its
  input.
- Every read has a non-negative offset. Each neighbor is a plain host `slice`
  view with exactly the output's shape, and `load_like(out)` lines it up with
  no block-index arithmetic at all.
- After the last stage, the edge map is exactly the original image size.

The separable blur uses the horizontal 5-tap binomial `[1 4 6 4 1]` into a
`u16` buffer, then the vertical pass normalizes by 256 back to `u8`. That's 5
views per pass instead of 25 for a full 5x5 kernel.

## Chaining: why not `.then()`

`.then(|out| next_op)` passes the previous stage's *owned* output to a closure.
A stencil stage needs *borrowed* views of that output, and views created inside
the closure can't outlive it. So the pipeline preallocates every stage buffer
and defines the chain once, against a small trait (now in `tilekit`, shared
with `light2d`):

```rust
pub trait Submit {
    fn submit<T: Send, N: GraphNode + DeviceOp<Output = T>>(&self, op: N) -> Result<(), Error>;
}
```

`Eager` implements `submit` as `op.sync_on(stream)`, and cuTile's graph `Scope`
implements it as `s.record(op)`. The same `Pipeline::run` therefore executes
eagerly or records a CUDA graph. Each new frame is copied into the fixed input
buffer, and one graph launch runs all six kernels.

## Gotchas hit while building it (cutile `d92c160`)

- **Tile dimensions must be powers of two.** A `[B, B, 3]` RGB tile fails in
  `tileiras` with "tile shape dimensions must have power of two length". The
  input is uploaded as RGBA (`[B, B, 4]`, with alpha weight 0) instead. The
  only visible error was "input does not correspond to Tile IR bytecode"; the
  real cause was further up in stderr.
- **`convert_tile` only does int <-> float and float <-> float.** For integer
  widths, use `exti` (signedness comes from the source type, so `u8`
  zero-extends) and `trunci`.
- **`trunci(x, overflow::None)` fails** when the bytecode is written ("missing
  attribute 'overflow'"). Use a real mode; `overflow::NoWrap` is correct here
  because every value is already in range.
- **Tile type mismatches in `+ - *`.** Those operators compare operand types
  token for token, and the JIT assigns two different types to the same tile.
  Nested arithmetic, `constant(...)` and `exti(...)` results keep the generic
  `{[B, B]}`. Reductions (`reduce_sum`) and device-function results get the
  concrete `{[32, 32]}`. Mixing them gives ``binary `Add` requires operands of
  the same type``. Ops like `shri`, `select` and `eq_tile` don't check. The
  workaround is to keep each kernel on one side: grayscale is concrete
  (`reduce_sum` plus a `fill` device function), and the stencils are generic
  (`exti` plus `constant(k, shape![B, B])`). A cleaner fix, found later in
  `raymarch`: drop the `const B` generic and write the shapes as literals
  (`{[32, 32]}`), so there is only one spelling.
- **Scalar `.broadcast(shape)` only resolves on entry parameters.** On a
  literal or a local `let`, the JIT inlines the trait's placeholder body and
  fails with "unrecognized macro `unreachable`". `Tile::shape()` fails the same
  way.
- Debug with `CUTILE_DUMP=ir,bytecode`, and read *all* of stderr: the useful
  `tileiras` message appears before the generic one.

## Ideas to try next

- Overlap transfers with compute: upload frame N+1 on a second stream while
  frame N is filtered.
- Upload 3 bytes per pixel instead of 4 (the alpha channel is padding).
- Compare f32 Sobel with `sqrt(gx² + gy²)` against the integer L1 version.
- Push a live video stream or webcam frames through the captured graph.
