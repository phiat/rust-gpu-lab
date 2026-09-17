# mandelbrot

Demo #1 in tileworld: an escape-time Mandelbrot renderer written as cuTile Rust
tile kernels, checked and benchmarked against a rayon CPU version.

```
src/gpu.rs      tile kernels (fixed-iteration, per-tile early exit) + launch code
src/cpu.rs      same f32 math as a plain per-pixel loop (serial + rayon)
src/palette.rs  escape counts -> PNG
src/main.rs     CLI: render / bench
```

## Run

Needs CUDA 13.2+ (`.cargo/config.toml` points at `/usr/local/cuda-13.3`).

```sh
cargo run --release -p mandelbrot -- render                    # -> mandelbrot.png
cargo run --release -p mandelbrot -- render --center=-0.7453,0.1127 --span 0.0065 --iters 3000 --out seahorse.png
cargo run --release -p mandelbrot -- bench                     # CPU vs both GPU kernels + pixel diff
cargo run --release -p mandelbrot -- bench --iters 10000 --skip-serial --tile 32
cargo run --release -p mandelbrot -- render --help
```

Every GPU render is compared against the CPU in `bench`. So far every run has
agreed on 100% of pixels.

## Results

RTX 4070 Ti SUPER, i9-14900KF (28 threads), WSL2, 1920x1080, 64x64 tiles.
GPU times are warm medians. Each new process first pays about 350-550 ms of JIT
compilation.

Default view, 1000 iterations:

| renderer        | time    |
|-----------------|---------|
| CPU, 1 thread   | 980 ms  |
| CPU, rayon      | 48.7 ms |
| GPU fixed       | 1.06 ms |
| GPU early-exit  | 0.69 ms |

10,000 iterations by view:

| view                            | rayon   | GPU fixed | GPU early-exit |
|---------------------------------|---------|-----------|----------------|
| default (mixed)                 | 435 ms  | 10.1 ms   | 6.1 ms         |
| entirely outside (`2.5,2.5`)    | 4.7 ms  | 10.2 ms   | 0.10 ms        |
| entirely inside (`-0.1,0`, 0.4) | 1650 ms | 10.6 ms   | 12.1 ms        |

Tile size (fixed kernel, 10k iterations, default view): 16 px 9.7 ms, 32 px
10.7 ms, 64 px 10.4 ms, **128 px 85 ms, 256 px 1577 ms**. The first launch also
grows with tile size: 1 s at 128 px, 12 s at 256 px.

## What it teaches

**Tile programs, not pixel threads.** Nothing gives a kernel a per-pixel index.
Each program gets its block id and computes coordinates for its whole tile:
`block_id * BH + iota(BH)` for rows and the same for columns, then broadcasts
those two 1-D tiles to 2-D.

**Mask with `select`, don't branch.** All pixels in a tile step together, and an
escaped pixel is frozen with `select(alive, next, current)`.

**Early exit works per tile, and loop shape matters.** The first early-exit
version used one `while` with the "anything alive?" check inside it. It ran
2.7x slower than the fixed kernel when nothing escaped, and was still that slow
when the check never fired. The fix was to keep the hot loop a counted `for`
over `check_every` steps inside an outer `while`, where the `break` goes.

**Image sizes don't have to divide evenly.** 1080 / 64 isn't whole. The partition
rounds the grid up and masks stores in the edge tiles.

## Gotchas hit while building it (cutile `d92c160`)

- crates.io only has 0.3.1, and the book documents `main`, so the workspace
  pins a git rev.
- A launch returns *every* argument as a tuple, and `.first()` only exists for
  tuples up to 6 elements. Adding a 7th argument makes `.first()` disappear
  with a trait-bound error, so keep scalar lists short. This is why pixels are
  square and use a single `step`.
- Nested calls need types: `gt_tile(x, constant(256.0f32, s))` fails at JIT time
  with "Return type required". Bind the constant to an annotated `let` first.
- A one-line device function `fn bailout<const BH, const BW>(s: Shape<..>) -> Tile<..>`
  failed at JIT time with ``binary `Le` requires operands of the same type
  ... `{[BH, BW]}` and `{[32, 32]}` ``: the const generics weren't substituted.
  Inlining the constant fixed it. `tile_coords`, which also takes a `Shape`,
  works fine, so the exact trigger is still unclear.
- `to_host_vec()` consumes the tensor, and the `Stream` type comes from
  `cuda-core`, which the cutile prelude doesn't re-export.
- Everything stays `f32`, so expect pixelation once `--span` drops to around
  `1e-4`. Moving to `f64` would be a good exercise.

## Ideas to try next

- Persist the JIT cache across runs (see cutile's `jit_disk_cache` example).
- Move the palette onto the GPU so it outputs RGB directly (a 3-D output tensor).
- Generate a zoom animation from a pre-allocated buffer, as a warm-up for
  demo #3 (CUDA graphs).
