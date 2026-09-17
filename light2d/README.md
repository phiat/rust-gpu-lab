# light2d

The second demo from the rendering brainstorm in tileworld: paint walls and
lights, and a 2D global illumination pass lights the scene with soft shadows,
live in a window.

It runs every frame as a CUDA graph of cuTile kernels:

1. `seed_init`: every occupied pixel becomes its own seed.
2. `jfa_step` x 11: **jump flooding**. Each pass looks at offsets k = 512,
   256, ... 1 (plus one more pass at 1) and keeps the nearest seed.
3. `distance` and `nearest`: the distance field, and the color of the nearest
   surface (a gather through the seed coordinates).
4. `radiance`: every pixel casts rays that **sphere-trace the distance field**
   (a gather per step) and add up the lights they land on, blended into a
   running average.
5. `compose`: shading, tone mapping and debug views, packed into 0RGB pixels.

```
src/gpu.rs    kernels + Pipeline (buffers, ping-pong, even/odd graphs)
src/cpu.rs    bit-exact CPU jump flood, exact distance transform, CPU radiance
src/scene.rs  packed scene pixels, brush strokes, demo and random scenes
src/main.rs   run (window) / render (PNG) / bench / check
```

## Run

```sh
cargo run --release -p light2d -- run                  # 640x352 world, 2x window
cargo run --release -p light2d -- run --width 1280 --height 704 --scale 1
cargo run --release -p light2d -- render --view distance --out dist.png
cargo run --release -p light2d -- check                # JFA vs CPU vs exact EDT
cargo run --release -p light2d -- bench --quality 3
```

Window controls: left drag paints light (Shift+drag erases), right drag paints
wall, scroll sets the brush size, `1`-`6` pick the light color, `V` cycles
views (lit, radiance, distance, Voronoi), `Q` cycles quality, `O` toggles the
orbiting light, Space pauses it, `C` clears, `R` resets the scene, `P` saves a
screenshot, Esc quits.

The first run compiles 15 kernels, 10 of them `jfa_step` variants (see
gotchas). Later runs load the compiled kernels from the disk cache but still
spend about 4 s building IR before the first frame.

## Results

RTX 4070 Ti SUPER, i9-14900KF (28 threads), WSL2, demo scene. Frame times are
for the whole pipeline replayed as one CUDA graph.

| world, rays x steps          | CPU flood | CPU radiance | GPU flood | GPU frame | GPU frame, eager | GPU frame, no early exit |
|------------------------------|-----------|--------------|-----------|-----------|------------------|--------------------------|
| 640x352, 16 x 48             | 27 ms     | 21 ms        | 0.11 ms   | 0.65 ms   | 1.14 ms          | 0.97 ms                  |
| 640x352, 32 x 64             | 28 ms     | 36 ms        | 0.12 ms   | 1.25 ms   | 1.78 ms          | 2.21 ms                  |
| 1280x704, 32 x 64 (12 passes)| 37 ms     | 161 ms       | 0.27 ms   | 3.40 ms   | 3.94 ms          | 6.50 ms                  |

The window at its defaults (640x352 world, 32 rays) runs at about 360 fps, or
460 fps at 16 rays. Besides the GPU frame, each window frame uploads the scene
(0.14 ms) and downloads the image (0.12 ms) through pinned host buffers
(`tilekit::Pinned`; see `filters/README.md`).

**Everything matches the CPU exactly.** Seeds, distances and nearest-surface
colors are integer or correctly rounded math, so they match bit for bit, both
eager and replayed from a graph on a new scene. The float radiance matched too:
zero difference on every pixel of the demo scene, `cos`/`sin` included.

**JFA accuracy** against an exact Euclidean distance transform, on random
discs plus 3000 isolated dots:

| world     | plain JFA                    | JFA+1                       |
|-----------|------------------------------|-----------------------------|
| 640x352   | 51 px wrong, worst by 0.92 px | 1 px wrong, worst by 0.27 px |
| 1920x1088 | 51 px wrong, worst by 0.94 px | 1 px wrong, worst by 0.12 px |

## What it teaches

**Long-range stencils are Life's offset trick at any distance.** A read at
offset d splits into d = q * 32 + s: a host view shifted by s (0 <= s < 32)
loaded at block index + q. For k >= 32, s is 0 and all nine views are the same
unshifted tensor, with block offsets of +-k/32. For k < 32, the views are
shifted by 32 - k (behind, q = -1) and by k (ahead, q = 0). One kernel takes the
nine views plus the two block offsets and handles every pass.

**Edges come from padding rules, not branches.** cuTile zero-pads partial
tiles, but an out-of-range block index is a runtime assertion. So the kernel:

- **Encodes seeds so zero means far away.** A seed packs `(row + 8192, col +
  8192)`, so 0 decodes to (-8192, -8192), farther than any real seed. Padding
  reads as "no seed" with no special case.
- **Clamps every block index** and uses a scalar `if` to swap in zeros for
  blocks outside the buffer.
- **Adds a leading ghost ring of one tile**, so block i - 1 exists for the
  first real tile. Only a leading ring is needed; reads past the far edge land
  in zero-padded partial tiles.

**Gathers need the unsafe escape hatch.** A ray marching through the distance
field samples wherever it currently is. No fixed view offset expresses that, so
`radiance` builds a pointer tile from `tensor.as_ptr()`, offsets it by
`row * cols + col`, and calls `load_ptr_tko` inside `unsafe`. The safety
condition is that every index is in bounds, which the kernel guarantees by
clamping before the gather. `nearest` uses the same helper to look up the scene
pixel each seed points at.

**CUDA graphs matter again.** A frame is 16 kernel launches, so replaying the
graph saves about 0.5 ms per frame: 1.14 ms eager vs 0.65 ms at 640x352. The
11 flood passes on their own take 0.11 ms.

**Temporal accumulation needs two graphs.** The running average reads last
frame's buffer and writes the other one. A graph bakes in which buffer is which,
so the pipeline captures an even-frame graph and an odd-frame graph and
alternates between them. The noise pattern is interleaved gradient noise,
scrolled by the golden ratio each frame, so a few frames of averaging resolve
it.

**Early exit still pays off.** Rays stop when they land on a surface or leave
the world. Each tile checks every 8 or 16 steps and stops when none of its rays
are still marching, which makes the whole frame 1.5-1.9x faster.

## Gotchas hit while building it (cutile `d92c160`)

- **Kernels are specialized on divisibility, not just generics.** The JIT cache
  key records the largest power-of-two divisor (capped at 16) of every integer
  scalar argument and of every tensor's shape, strides and base pointer. JFA
  passes differ in their block offsets (+-16 or more, 8, 4, 2, 1) or in their
  view shifts (16, 24/8, 28/4, 30/2, 31/1), so `jfa_step` compiles into 10
  variants. Passes in the same class share one: the two k=1 passes, and at
  1920x1088 the k=1024 and k=512 passes. Any world size gets the same 10 keys,
  because buffers are always whole tiles. Each quality preset has its own
  `radiance` key too (8, 16 and 32 rays), so pressing `Q` compiles a new one
  the first time. Every variant costs about 270-320 ms of IR building at each
  startup, even when the compiled kernel comes from the disk cache.
- **Helper names can collide with DSL ops.** A device function called `unpack`
  failed with "duplicate functions are not supported", because Tile IR already
  has `pack`/`unpack` ops.
- **`convert_tile` needs an annotated `let`.** As a function's tail expression
  it failed with "Failed to get type parameters for convert_tile".
- **A `&mut Tensor` passed as a read-only kernel input must be reborrowed as
  `&*src`**, or the launcher looks for `DeviceOp` on `&mut Tensor`.
- **Out-of-range partition loads assert at runtime; partial tiles zero-pad.**
  Hence the clamp plus scalar `if` in `jfa_step`.

## Ideas to try next

- Radiance cascades for noise-free 2D GI, instead of more rays.
- A bounce: let walls reflect the radiance arriving at them.
- Point lights with SDF soft shadows (one march per light instead of per ray).
- Half-resolution radiance with an edge-aware upsample for 4K worlds.
