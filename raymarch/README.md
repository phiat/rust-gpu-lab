# raymarch

The first demo from the rendering brainstorm in tileworld (after the original
plan's #1-#3): a real-time signed-distance-field ray marcher written as a
single cuTile tile kernel. Each 32x32 tile program sphere-traces its pixels,
then shades them with soft shadows, ambient occlusion, specular highlights and
fog, and packs the result into `0x00RRGGBB` for a minifb window. A rayon CPU
version of the same math checks the result.

```
src/gpu.rs    the kernel (SDFs, march, shading) + Renderer (param buffer, frame, graph)
src/cpu.rs    the same scene and shading, one pixel at a time
src/scene.rs  orbit camera, animation, 32-slot parameter block
src/main.rs   run (window) / render (PNG + CPU diff) / bench
```

## Run

```sh
cargo run --release -p raymarch -- run                  # window, 1280x720
cargo run --release -p raymarch -- run --width 1920 --height 1080 --quality 3
cargo run --release -p raymarch -- render --yaw 200 --time 2 --out shot.png
cargo run --release -p raymarch -- render --view tiles  # per-tile work heatmap
cargo run --release -p raymarch -- bench --width 1920 --height 1080 --quality 3
```

Window controls: drag or arrow keys to orbit, scroll or `+`/`-` to zoom, Space
to pause the animation, `O` to toggle auto-orbit, `V` to cycle views (shaded,
march steps per pixel, steps each tile ran), `1`/`2`/`3` for quality, `P` to
save a screenshot, Esc to quit.

The **first run takes about 30 s**: that is `tileiras` compiling the kernel.
`main` enables cuTile's disk cache (`~/.cache/cutile/kernels`), so later runs
start in about 1.5 s. Any edit to the kernel changes its cache key and pays the
compile again.

## Results

RTX 4070 Ti SUPER, i9-14900KF (28 threads), WSL2, 1920x1080, default camera.

| quality (march / shadow steps) | CPU rayon | GPU eager | GPU graph | GPU, no early exit |
|--------------------------------|-----------|-----------|-----------|--------------------|
| low (64 / 16)                  | 129 ms    | 2.34 ms   | 2.25 ms   | 2.77 ms            |
| medium (128 / 32)              | 175 ms    | 3.19 ms   | 3.17 ms   | 5.07 ms            |
| high (256 / 64)                | 256 ms    | 4.14 ms   | 4.10 ms   | 9.58 ms            |

That's 55-62x faster than rayon. The window at 1280x720, medium quality, runs at
about 310 fps: 1.65 ms rendering, 0.4 ms downloading and 0.12 ms uploading
parameters.

`render` and `bench` compare the GPU frame with the CPU renderer. The largest
per-channel difference is 1 level out of 255.

## What it teaches

**One kernel does everything.** Mandelbrot iterated one formula; here a tile
program runs a full shading pipeline: the camera ray, the march, normals from
4 scene samples, a 32-step shadow march, 5 occlusion samples, a material lookup
and a color. Everything is a `[32, 32]` tile, every "if" is a `select`, and
helper functions compose like normal Rust.

**Finished rays are frozen, and tiles stop when all of theirs are.** Each march
step does `active = active && d >= eps*t && t <= tmax`, then
`t = select(active, t + d, t)`. Every 16 steps the kernel reduces `active` to one
scalar and breaks if nothing is still marching. `--view tiles` shows the result:
tiles in front of the camera stop after 16 steps, sky tiles after about 32, and
tiles along the horizon, where rays skim the ground, use the whole budget. The
more steps you allow, the more early exit saves: 1.2x at 64 steps, 2.3x at 256.

**Scalar `if` skips work for whole tiles.** Sky-only tiles skip the normal,
shadow and occlusion code entirely (`if any_hit > 0.5 { ... }`, with tiles
modified inside the block). That's the only way to skip work. Within a tile
every pixel runs every op, so `select` computes both sides.

**No bounding volumes, so SDF cost is per-sample cost.** On a CPU or in a
shader you'd skip a costly shape when the sample is far from it. Here that
would be a `select`, and both branches run anyway. The first pillar rings used
`atan2` + `cos` + `sin` for polar repetition, and the 720p frame time went from
about 1.6 ms to 5.8 ms. Rewriting the same rings as mirror folds (`abs`, a
min/max swap, and two reflections through fixed lines) brought it back to 1.6 ms.

**Inputs live in a device buffer, not scalars.** Scalar kernel arguments are
baked into a captured CUDA graph. The camera, animation and view mode are in a
32-slot `f32` tensor instead. The kernel loads it as a `[32]` tile and pulls
values out with `extract`. Each frame `memcpy`s new values into the same
buffer, then replays the graph. Loop bounds (step counts) stay scalars, so
changing quality recaptures the graph.

A frame is one kernel launch, so the graph barely beats eager (3.17 vs
3.19 ms). Life, which launches a kernel per generation, gained up to 5x.

**Floats almost match the CPU.** The CPU version keeps the kernel's operation
order and lands within 1/255 everywhere. It isn't bit-exact (fused multiply-add
and `exp` can differ in the last bits), but sphere tracing didn't amplify that
into different hits.

## Gotchas hit while building it (cutile `d92c160`)

- **Literal tile shapes dodge the generic type bug.** In filters, mixing device
  function results (`{[32, 32]}`) with locals (`{[B, B]}`) broke `+ - *`. This
  kernel has no `const B` generic at all: every shape is written as `32`, behind
  aliases like `type F = Tile<f32, { [32, 32] }>`, and helper functions compose
  freely. The cost is that the host must partition with exactly 32x32 tiles.
- **Compile time grows with every inlined copy of the scene.** Device functions
  are inlined, so the scene SDF exists once per call site. With the normal
  written as 4 separate `scene` calls, `tileiras` took 48 s; folding those 4
  samples into a `for` loop cut it to 26 s. The final kernel takes about 31 s.
  Separately, IR construction costs about 1.4 s on every start, even with a disk
  cache hit.
- **The disk cache is opt-in.** Call `cutile::jit_cache::enable(...)`. Set
  `CUTILE_JIT_TIMING=1` to see stage timings and whether the compiled cubin came
  from disk or `tileiras`.
- **Kernel scalars aren't part of the cache key** (`generics=` is empty), so
  switching quality presets doesn't recompile.
- **A soft-shadow artifact from an underestimating SDF.** The first pillar
  layout was a repeated grid with a courtyard cut out,
  `max(column, 5 - length(xz))`. That is a valid lower bound, but near the cut it
  badly underestimates, and the soft-shadow estimate `k * h / t` turns that into
  wavy false penumbras. The fold-based rings are exact.
- `atan2(y, x)` matches Rust's `y.atan2(x)`.
- A warm frame measured right after a 30 s compile can take 30-90 ms, probably
  because the GPU is still leaving its idle power state. `bench` renders 50
  frames before timing anything.

## Ideas to try next

- Reflections: a second march from the hit point for the ring and ball.
- Anti-aliasing: 4 rays per pixel (the checkerboard shimmers in the distance),
  or temporal accumulation while the camera is still.
- A 16x16 kernel: finer early exit, but it needs a second copy of the kernel,
  since the shape is a literal.
- Pinned host memory for the download, now about 20% of the frame.
