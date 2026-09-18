# sand

Demo #6 in tileworld, and the first of the game-dev series: a falling sand
game on the GPU. Sand piles up, water levels out and drains off shelves, oil
floats on water, fire rises, wood catches and burns down to smoke, and water
puts embers out. Paint any of it with the mouse. Every cell rule is integer
math, so the GPU is checked cell for cell against the CPU.

```
src/world.rs  elements, the cell encoding, buffer layout, host-side painting
src/gpu.rs    the step kernel (one Margolus pass), paint and render kernels, Pipeline
src/cpu.rs    the same rules in plain Rust, on rayon
src/main.rs   clap CLI: run (window) / render (PNG) / bench / check
```

![sand: the demo scene after 300 frames, with the hut on fire](../docs/images/sand.png)

## Run

```sh
cargo run --release -p sand -- run                  # 640x352 window at 2x
cargo run --release -p sand -- run --width 960 --height 512 --scale 1
cargo run --release -p sand -- render --frames 300  # demo scene -> sand.png
cargo run --release -p sand -- bench                # CPU vs GPU eager vs GPU graph
cargo run --release -p sand -- check --frames 60    # exact CPU/GPU comparison
```

Window controls: left drag paints, right drag erases, scroll sets the brush
size. `1`-`8` pick sand, water, oil, wood, wall, fire, smoke, ember. `Space`
pauses, `E` toggles the sand and water taps, `C` clears, `R` resets the demo
scene, `P` saves a screenshot, `Esc` quits.

`--passes` is the number of Margolus passes per frame (default 4, must be
even). Each pass moves a cell at most one step, so more passes means faster
falling at the same frame rate.

## Results

RTX 4070 Ti SUPER, i9-14900KF (28 threads), WSL2. Per frame, 4 passes.

| world      | CPU rayon | GPU eager | GPU graph | speedup |
|------------|-----------|-----------|-----------|---------|
| 640x352    | 6.7 ms    | 0.356 ms  | 0.231 ms  | 29x     |
| 1920x1088  | 12.1 ms   | 1.28 ms   | 1.10 ms   | 11x     |

The graph is worth a third at the small size: a frame is one paint kernel,
four step kernels and one render kernel, so launch overhead is a large share
of 0.23 ms. Transfers per frame, through `tilekit::Pinned`: paint upload
0.08 ms and frame download 0.13 ms at 640x352; 0.46 ms and 1.0 ms at 1080p.
The window runs at about 600 fps at 640x352 (the CPU-side paint and blit
cost more than the simulation).

The first frame costs 2.2 s even with the JIT cache warm: the `step` kernel
is big (it spells out every block rule on nine views of the source), and its
first stage alone is 1.7 s. See "Compile time is run time" below.

`check` runs 60 frames at 640x352 with random blobs of every element and the
taps running, alternating eager and graph frames, and compares every world
cell with the CPU after each frame. The match is exact, there and at
1280x704.

## What it teaches

**Moving cells without write conflicts.** A cell that "wants to fall" has to
agree with the cell below about who ends up where, and on a GPU nobody can
lock anything. The classic answer is the Margolus neighborhood: partition the
grid into 2x2 blocks, update each block on its own (four cells, all rules in
plain code), and shift the partition by one cell each pass so blocks overlap
over time. Every cell is written by exactly one block, so there are no
conflicts and no atomics. In a tile kernel each cell still reads its 3x3
neighborhood from nine shifted views, picks out its three block mates with
masks that depend on the pass parity (`(row + pass) & 1`, `(col + pass) & 1`),
runs the block rule on all four cells, and keeps the result for its own
corner. All four cells of a block compute the same thing redundantly, and
that costs nothing next to the reads.

**Randomness as a hash.** There is no RNG on the device, so every block gets
one 32-bit word from `hash(bx ^ hash(by ^ hash(salt)))` (lowbias32) with
`salt = frame * 64 + pass`. The four bytes drive reactions and a second hash
supplies the bits that pick slide order and gas wandering. The CPU does the
same hashes, which is what makes the comparison exact.

**Rules that look right.** The block rule is short but each line fixes
something that looked wrong:

- Gravity swaps a cell with the one below when it is denser and neither is
  solid. Gases rise because empty space (density 2) is denser than smoke (0)
  and fire (1).
- Diagonal slides go left or right first at random; without that, piles lean.
- A cell that moved vertically in a pass may not also move sideways
  (`fell` flags). Without this rule water "trades" diffusively and piles up in
  45° cones instead of finding a level.
- Liquids carry a direction bit (`DIR`) and keep flowing that way until
  blocked, then turn around. That is what makes a pool level out and drain
  off a shelf in a stream instead of a random walk.
- Fire on its own rises and dies in a few passes, so it never had time to
  light wood. `EMBER` is burning wood: it stays put, throws fire into empty
  neighbors, burns out to smoke, and turns back to wood when water touches
  it.

**Passing per-launch integers without JIT variants.** cuTile specializes a
kernel on the power-of-two divisibility of every integer scalar argument.
Passing the pass index as a scalar compiled three variants of the 2.4 s
`step` kernel (for pass 0, odd passes, and passes divisible by 2 but not 4).
The pass index now lives in a small tensor, sliced per launch, and loaded
with a `[4]`-shaped tile: one variant, and the slice is also what a CUDA
graph needs (the graph replays the launch with the same view).

**Compile time is run time.** The first `step` kernel took 2.4 s to compile
and 0.5 ms per frame. Both numbers scale with the count of tile ops, and the
kernel had a lot of them: every `kind == X || kind == Y` predicate was a chain
of compares and selects, repeated for every neighbor. Replacing them with
bit-mask tables (`has(k, table) = (table >> k) & 1`), packing the density
table into one constant (`0x01993452 >> 4k & 15`), and hoisting `kind()` of
each cell out of the rules cut compile to 1.7 s and the frame to 0.21 ms.
Same rules, 2.4x faster, from writing them smaller.

## Gotchas hit while building it (cutile `d92c160`)

- **`muli(.., overflow::None)` fails to serialize** ("missing attribute
  'overflow' on op MulI"), and so does `shli`. The `*` operator emits
  `overflow=0` and wraps, which is exactly what a hash wants. Left shifts
  became multiplications by a power of two.
- **Signed constants**: `0x846c_a68b` does not fit an `i32`, and typing its
  two's complement by hand went wrong twice. The GPU was right both times;
  the CPU reference had the typo. Compute the constant (`0x846c_a68bu32 as
  i32`) instead of writing it.
- **Every kernel name becomes a type**: a kernel called `hash_map` generated
  a `HashMap` type that clashed with the `std` one the macro's own expansion
  uses. Renamed to `hash_grid`.
- **Block indices can't go negative**: views shifted by -1 assert when the
  block index is out of range, so the "up" and "left" views clamp with
  `max(i - 1, 0)` and the ghost tile ring makes that harmless (the ring is
  `WALL` and is never updated, so its content never matters).
- **`Stream::synchronize` is unsafe** and needs the device context bound
  to the calling thread first: `stream.device().bind_to_thread()?`.

## Ideas to try next

- More elements: acid, steam (water near fire rises, condenses at the
  ceiling), lava, plants that grow toward water, gunpowder.
- Temperature as a second field instead of "near fire" checks, so heat
  spreads through metal and water boils.
- Rigid bodies on top of the grid (Noita's trick: rasterize a body into the
  cells, simulate, read it back).
- Pressure for liquids so water rises in a U-bend.
- Persistent world larger than the window, streamed in tiles.
