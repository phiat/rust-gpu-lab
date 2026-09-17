# life

Demo #3 in tileworld: Conway's Game of Life on a torus. One cuTile stencil
kernel computes a generation, a captured CUDA graph replays many generations
per launch, and a window shows the result at 60 fps.

```
src/gpu.rs   stencil kernel, World (ping-pong buffers), eager step, graph capture
src/cpu.rs   rayon reference + seeded random soup
src/main.rs  Layout (ghost-tile padding), `run` window, `bench`
```

## Run

```sh
cargo run --release -p life -- run                      # 1280x736 world, 60 fps window
cargo run --release -p life -- run --width 640 --height 360 --tile 16 --scale 2
cargo run --release -p life -- run --gens-per-frame 20  # faster evolution
cargo run --release -p life -- bench                    # 4096^2: CPU vs GPU eager vs GPU graph
cargo run --release -p life -- bench --size 256 --tile 32 --gens 100
```

In the window: Space pauses, R reseeds, Esc quits. The title bar (and the
terminal) shows fps, generations/s, and the time per graph launch.

`bench` runs the same seed through all three paths and checks that each final
world matches the CPU cell for cell. Every run so far has matched.

## Results

RTX 4070 Ti SUPER, i9-14900KF (28 threads), WSL2. Times are per generation.

| world | tile | CPU rayon | GPU eager | GPU graph | graph vs eager |
|-------|------|-----------|-----------|-----------|----------------|
| 256²  | 32   | 1.633 ms  | 0.035 ms  | 0.007 ms  | 5.0x           |
| 1024² | 32   | 0.877 ms  | 0.046 ms  | 0.021 ms  | 2.2x           |
| 4096² | 64   | 4.117 ms  | 0.352 ms  | 0.332 ms  | 1.06x          |

"Eager" launches the kernel and syncs once per generation. "Graph" replays
10-20 generations per driver call.

CUDA graphs remove launch overhead, not kernel time. Their payoff shrinks as the
world grows. At 1024² and 4096² the GPU tops out around 50 B cells/s. Each cell
reads 9 bytes and writes 1, so that's about 500 GB/s, against 672 GB/s of GDDR6X
bandwidth on this card. That suggests the limit is memory bandwidth, not compute.

## How the stencil works

**Problem 1: a stencil reads outside its own tile.** A tile program writes its
B x B block but needs the ring of cells around it.

**Problem 2: views can't have negative offsets.** Host `slice` views start at
offset >= 0, and output partitions must cover a whole tensor. So you can't just
write into the interior view of a padded buffer.

**The trick:** split the offset. Offset d = q·B + s with 0 <= s < B, read as
"view offset by s, loaded at block i + q". Offset -1 becomes a view offset by
B-1, loaded at block i-1. Offset +1 becomes a view offset by 1, loaded at
block i. Nine host views (3 row offsets x 3 column offsets) cover the 3x3
neighborhood, and each one loads an aligned B x B tile.

**Ghost ring:** block i-1 doesn't exist for i = 0, so the buffer carries a
full ring of ghost tiles and logical cell (r, c) lives at [B + r, B + c]. A
ghost tile computes the interior tile it mirrors on the opposite side of the
torus, so the ring is correct after every step with no separate copy pass. The
host fills the ghosts only once, when it seeds the world.

**Rule without branches:** with S = the 3x3 sum including the cell,
`next = S == 3 || (S == 4 && alive)`, written as two `select`s.

## CUDA graph notes

- A graph bakes in its buffer pointers, so the eager `swap(front, back)` can't
  be used. The graph records ping-pong pairs (front -> back -> front), so the
  state always ends up in `front`, and generations per launch are even.
- Recording a graph doesn't advance the state. The benchmark confirms this:
  after `launches x gens` it matches the CPU after exactly that many
  generations.
- Nothing inside a graph may allocate. Reseeding therefore copies into the
  existing buffer (`api::memcpy`) instead of creating a new tensor, and the
  captured graph stays valid.
- Downloading a buffer that the graph keeps writing uses `.dup().to_host_vec()`,
  because `to_host_vec()` consumes the tensor.
- The first step runs eagerly to trigger JIT compilation before capture.

## Gotchas

- A 10-argument kernel works fine with `s.record(...)` and `sync_on`. Only the
  `.first()` tuple helper is limited to 6 elements.
- The window uses `minifb` with the `x11` feature only. WSLg provides the X
  server, and Wayland support would need extra dev packages.
- The world is rounded up to a multiple of `--tile` (720 -> 736 at tile 32).

## Ideas to try next

- Paint cells with the mouse by memcpy-ing a small host patch into `front`.
- Load RLE patterns (a Gosper glider gun wrapping around the torus).
- Store 8 cells per byte (bit-packing) to cut the memory traffic that now limits
  large worlds.
- Swap in a particle sim to exercise float kernels instead of `u8` stencils.
