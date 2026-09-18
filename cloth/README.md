# cloth

Demo #7 in tileworld: a cloth simulation on the GPU. A grid of particles
joined by distance constraints hangs from pins, blows in a gusty wind, and
drapes over a sphere you move with the mouse. The solver is position based
dynamics (Verlet integration, then constraint projection passes), and the
CPU reference runs the same operations in the same order.

```
src/world.rs   layout, particle encoding, pins, constraint batches, anchors, physics, camera
src/gpu.rs     integrate / pair / normals / shade kernels and the Pipeline
src/cpu.rs     the same math on rayon
src/raster.rs  CPU triangle rasterizer for the projected, lit particle grid
src/main.rs    clap CLI: run (window) / render (PNG) / bench / check
```

![cloth: the demo after 240 frames, draped over the sphere in the wind](../docs/images/cloth.png)

## Run

```sh
cargo run --release -p cloth -- run                       # 256x160 particles, 960x640 window
cargo run --release -p cloth -- run --cols 512 --rows 320
cargo run --release -p cloth -- render --frames 240        # demo -> cloth.png
cargo run --release -p cloth -- bench                      # CPU vs GPU eager vs GPU graph
cargo run --release -p cloth -- check --frames 30          # GPU vs CPU, particle for particle
```

Window controls: the sphere follows the mouse; right drag orbits the
camera; scroll changes the wind. `1`-`4` pin the top row, a few points of
it, the two corners, or nothing (each resets the cloth). `W` toggles the
wind, `Space` pauses, `R` resets, `P` saves a screenshot, `Esc` quits.

`--substeps` (default 4) is the number of Verlet steps per frame and
`--iterations` (default 2) the solver iterations per substep; each
iteration is 12 passes, one per constraint batch (see below).

## Results

RTX 4070 Ti SUPER, i9-14900KF (8 P-cores + 16 E-cores), WSL2. Per frame:
4 substeps x (1 integrate + 24 constraint passes), plus normals and shading,
102 launches.

| particles | CPU rayon         | GPU eager | GPU graph | speedup |
|-----------|-------------------|-----------|-----------|---------|
| 256x160   | 18.6 ms (8 thr.)  | 3.01 ms   | 0.91 ms   | 20x     |
| 512x320   | 59.9 ms (28 thr.) | 3.60 ms   | 1.34 ms   | 45x     |

This is the biggest win for CUDA graphs so far: 102 launches of small
kernels per frame, and the graph replays them 3.3x faster than eager
launches. The remaining per-frame costs are the CPU rasterizer (2.8 ms at
960x640 for 81k triangles) and the parameter upload plus screen download
(0.07 ms). The window runs at about 180 fps, rasterizer bound.

The CPU column has a twist. At 256x160 each pass is about 1 ms of single
threaded work, and rayon's default pool of 28 threads makes it *slower*
(166 ms per frame) than 8 threads (18.6 ms): every pass ends with a join
that waits for the slowest thread, and on a hybrid chip that is an E-core.
At 512x320 the passes are long enough for all 28 threads to pay off.
`bench` times both and reports the better.

`check` runs the GPU and CPU side by side (alternating eager and graph
frames) and compares every particle after each frame. Frame 0 matches to
3e-8 spacings, i.e. to float rounding; by frame 30 the difference has grown
to about 0.5 spacings. That is not a bug: a cloth in gusty wind against a
sphere is chaotic, and the rounding differences (the GPU fuses multiplies
and adds where the CPU doesn't) roughly double every frame.
Both stay at 0.3-3% mean stretch throughout, and the pictures are
indistinguishable.

## What it teaches

**A grid of objects, not pixels.** A particle is an `f32x4`: position and
inverse mass, in a `[rows, cols, 4]` tensor loaded as `[32, 32, 4]` tiles
with `extract` picking channels apart and `cat` putting them back. Pinned
particles have mass zero (inverse mass 0), and so does the ghost ring
around the grid, which means "never moves" falls out of the arithmetic
(`share = w / (w + w_neighbor)`) instead of needing a special case.

**Red/black, done right.** The roadmap said "an iterative solver with
red/black passes", and the first version did exactly that: color the
particles like a checkerboard, and in each pass move one color toward
satisfying all twelve of its links (four structural, four shear, four bend)
while the other color holds still. It exploded, even at 1% gravity. Only
the structural links join particles of different colors; shear and bend
links join particles of the *same* color, so both ends moved at once,
each against the other's stale position, and the error compounded pass
after pass. Pure Jacobi (everybody moves every pass, by half) did not
explode but converged too slowly to hold a curtain up.

The fix is to color the *constraints*, not the particles: split the links
into batches such that no particle has two links in a batch (vertical
links with an even top row, then an odd top row, then horizontal, then
each diagonal, then the two-apart bend links: 12 batches). One pass
projects one batch, and every particle has at most one partner, so both
ends can move by their share and the pair ends up exactly at rest length.
That is Gauss-Seidel over batches, which is what red/black is for a
one-link stencil. The same kernel serves all 12 batches: the batch's
direction, stride and parity come from a small table (see `sand` for why
not as integer arguments), and the two shifted views passed in are what
make one batch's partner "up or down" and another's "left or right".

**One kernel, whatever the direction.** A view shifted by -1 row and one
shifted by +2 columns have different shapes, and cuTile specializes a
kernel on the divisibility of every argument's shape; a kernel that takes
"minus" and "plus" views for six different directions would compile in
several variants. `views` therefore cuts every view to the same shape
(the buffer minus 31 rows and columns, starting inside the ghost ring) and
every kernel loads all of them at the same block `[i - 1, j - 1]`. The
ghost ring is what makes a view start at "row 32 - 2" without going
negative. One `pair` kernel, one compile (0.75 s first frame with the cache
warm).

**Long range attachments.** Gauss-Seidel moves information one link per
pass, so a 160-row curtain under gravity sagged by tens of percent no
matter how many iterations were affordable: the top row is pinned, and the
bottom rows fall until the correction reaches them. The cheap, standard
answer (Kim et al. 2012) is to give every particle its nearest pin as an
anchor and clamp it to within the flat-cloth distance of that anchor,
every pass. The anchors are a tensor computed on the host when the pins
change. With them, stretch stays under 1% for the pinned cases at 2
iterations per substep, and the free cloth (no pins, no anchors) simply
falls onto the floor and the sphere.

**Rasterizing on the CPU.** Projecting and lighting every particle is a
per-element job and runs on the GPU (`shade` writes screen x, y, depth and
a packed color for each particle). Turning the quads between particles
into pixels is a scatter: each triangle touches an unpredictable set of
pixels, and the tile model has no primitive for that. So the last step is
a small rasterizer on rayon, with triangles bucketed into horizontal bands
so every pixel is owned by one thread. Item 8 on the roadmap (boids) will
hit the same wall from the other side.

## Gotchas hit while building it (cutile `d92c160`)

- **Helper names collide with builtins.** A module function called
  `pack`, `load` or `dot` fails with "duplicate functions are not
  supported" because `cutile::core` already has those names. Renamed to
  `pack4`, `load4`, `dot3`.
- **`select` needs a mask of the tile's own shape.** Selecting between two
  `[32, 32, 4]` tiles with a `[32, 32]` mask is a type error; select each
  channel instead.
- **Block indices are scalars.** A kernel can't pick which block of a view
  to load from a tile value, so a batch can't decide its own view offsets;
  hence the same-shape views loaded at the same block for every direction
  (above).
- **Two `&mut Tensor` outputs are fine**: `integrate` writes the new
  positions and the new "previous" positions in one kernel.

## Ideas to try next

- Self collisions: a spatial hash on the GPU is a scatter, but a coarse
  grid density (splat particles into cells, gather in the next pass)
  would give a repulsion term.
- Tearing: drop a link when it stretches past a limit; needs a per-link
  flag tensor, and a mesh that can have holes.
- Bending as a dihedral constraint instead of a distance two apart.
- XPBD compliance, so stiffness stops depending on the iteration count.
- Soft bodies: the same solver on a 3D grid of particles with volume
  constraints.
