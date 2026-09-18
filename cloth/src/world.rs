//! The particle grid, the simulation parameters and the camera.
//!
//! A particle is four `f32`s: position x, y, z and inverse mass `w`. Pinned
//! particles have `w = 0`, and so do the ghost particles in the ring of one
//! tile around the grid, which exist so every neighbor load is in bounds.
//! Everything the kernels need per frame goes through one `f32` parameter
//! buffer (`PARAMS` slots), so no kernel takes an integer argument and the
//! JIT compiles each kernel once.

use crate::gpu::TILE;

/// Number of slots in the parameter buffer.
pub const PARAMS: usize = 64;

// Parameter slots. Physics first, then the camera.
pub const P_DT: usize = 0;
pub const P_GRAVITY: usize = 1;
pub const P_DAMPING: usize = 2;
pub const P_WIND: usize = 3; // 3 slots
pub const P_TIME: usize = 6;
pub const P_SPHERE: usize = 7; // x, y, z, radius
pub const P_FLOOR: usize = 11;
pub const P_STIFF: usize = 12; // structural, shear, bend
pub const P_SPACING: usize = 15;
pub const P_ROT: usize = 20; // 3x3, row major
pub const P_TARGET: usize = 29; // 3 slots
pub const P_DIST: usize = 32;
pub const P_FOCAL: usize = 33;
pub const P_CENTER: usize = 34; // screen center x, y
pub const P_LIGHT: usize = 36; // 3 slots, camera space, unit length

/// Grid size in particles, rounded up to whole tiles. Buffers add a ring of
/// one ghost tile on every side.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub rows: usize,
    pub cols: usize,
}

impl Layout {
    pub fn new(rows: usize, cols: usize) -> Self {
        Layout {
            rows: rows.div_ceil(TILE).max(1) * TILE,
            cols: cols.div_ceil(TILE).max(1) * TILE,
        }
    }

    pub fn buf_rows(&self) -> usize {
        self.rows + 2 * TILE
    }

    pub fn buf_cols(&self) -> usize {
        self.cols + 2 * TILE
    }

    /// Particles in a buffer, ghosts included.
    pub fn len(&self) -> usize {
        self.buf_rows() * self.buf_cols()
    }

    /// Buffer index of grid particle (r, c).
    pub fn index(&self, r: usize, c: usize) -> usize {
        (r + TILE) * self.buf_cols() + c + TILE
    }

    /// True for buffer positions inside the grid (not the ghost ring).
    pub fn is_world(&self, r: usize, c: usize) -> bool {
        (TILE..TILE + self.rows).contains(&r) && (TILE..TILE + self.cols).contains(&c)
    }
}

/// Which particles are held in place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pins {
    /// The whole top row: a curtain.
    Top,
    /// Every 16th particle of the top row: a curtain on rings.
    Sparse,
    /// The two top corners: a flag held by its ends.
    Corners,
    /// Nothing: the cloth falls onto the sphere and the floor.
    Free,
}

impl Pins {
    pub fn name(self) -> &'static str {
        match self {
            Pins::Top => "top row",
            Pins::Sparse => "sparse",
            Pins::Corners => "corners",
            Pins::Free => "free",
        }
    }

    /// Inverse mass of grid particle (r, c).
    pub fn weight(self, r: usize, c: usize, cols: usize) -> f32 {
        let pinned = r == 0
            && match self {
                Pins::Top => true,
                Pins::Sparse => c.is_multiple_of(16) || c == cols - 1,
                Pins::Corners => c == 0 || c == cols - 1,
                Pins::Free => false,
            };
        if pinned {
            0.0
        } else {
            1.0
        }
    }
}

/// A flat cloth in the x-y plane, centered on the origin, `spacing` units
/// between neighbors, top row at +y. Ghost particles are all zero.
pub fn initial(lay: Layout, pins: Pins, spacing: f32) -> Vec<f32> {
    let mut buf = vec![0.0f32; lay.len() * 4];
    let (w, h) = (lay.cols as f32 - 1.0, lay.rows as f32 - 1.0);
    for r in 0..lay.rows {
        for c in 0..lay.cols {
            let i = lay.index(r, c) * 4;
            buf[i] = (c as f32 - w * 0.5) * spacing;
            buf[i + 1] = (h * 0.5 - r as f32) * spacing;
            buf[i + 2] = 0.0;
            buf[i + 3] = pins.weight(r, c, lay.cols);
        }
    }
    buf
}

/// One batch of constraints, all solved in the same pass: every particle
/// is in at most one link of a batch, so both ends of a link can move by
/// their share without stepping on another link's result. This is
/// red/black for links: `kind` picks the link direction (0 vertical, 1
/// horizontal, 2 the `\` diagonal, 3 the `/` diagonal), `lg` the stride
/// (`1 << lg`: 1 for structural and shear, 2 for bend), and `b` which of
/// the two interleaved sets of links along that direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Batch {
    pub b: i32,
    pub kind: i32,
    pub lg: i32,
    /// Stiffness slot: 0 structural, 1 shear, 2 bend.
    pub k: usize,
}

impl Batch {
    /// The twelve batches of one solver iteration: structural, shear, bend.
    pub fn iteration() -> [Batch; 12] {
        let mut out = [Batch {
            b: 0,
            kind: 0,
            lg: 0,
            k: 0,
        }; 12];
        let mut i = 0;
        for (kind, lg, k) in [
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 1),
            (3, 0, 1),
            (0, 1, 2),
            (1, 1, 2),
        ] {
            for b in 0..2 {
                out[i] = Batch { b, kind, lg, k };
                i += 1;
            }
        }
        out
    }

    /// The pass table entry the kernel reads.
    pub fn entry(self) -> [i32; 4] {
        [self.b, self.kind, self.lg, self.k as i32]
    }

    /// Unit step of this batch's links.
    pub fn step(self) -> (i32, i32) {
        let s = 1 << self.lg;
        match self.kind {
            0 => (s, 0),
            1 => (0, s),
            2 => (1, 1),
            _ => (1, -1),
        }
    }

    /// Row and column offset of the partner of buffer particle (r, c).
    pub fn partner(self, r: usize, c: usize) -> (i32, i32) {
        let (sr, sc) = self.step();
        // Links along a direction pair up consecutive groups of `1 << lg`
        // coordinates; `b` shifts the pairing by one group.
        let coord = if self.kind == 1 { c } else { r } as i32;
        let plus = ((coord >> self.lg) + self.b) & 1 == 0;
        if plus {
            (sr, sc)
        } else {
            (-sr, -sc)
        }
    }

    pub fn rest(self, spacing: f32) -> f32 {
        if self.kind >= 2 {
            std::f32::consts::SQRT_2 * spacing
        } else {
            (1 << self.lg) as f32 * spacing
        }
    }
}

/// Long range attachments: each free particle's nearest pin and the
/// distance along the flat cloth to it (`w` slot), which it may never
/// exceed. Zero distance means no pin, no limit.
pub fn anchors(lay: Layout, pins: Pins, spacing: f32) -> Vec<f32> {
    let pos = initial(lay, pins, spacing);
    let pinned: Vec<(usize, usize)> = (0..lay.rows)
        .flat_map(|r| (0..lay.cols).map(move |c| (r, c)))
        .filter(|&(r, c)| pins.weight(r, c, lay.cols) == 0.0)
        .collect();
    let mut out = vec![0.0f32; lay.len() * 4];
    for r in 0..lay.rows {
        for c in 0..lay.cols {
            let nearest = pinned
                .iter()
                .map(|&(pr, pc)| {
                    let (dr, dc) = (pr as f32 - r as f32, pc as f32 - c as f32);
                    ((dr * dr + dc * dc).sqrt(), (pr, pc))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0));
            if let Some((d, (pr, pc))) = nearest {
                let i = lay.index(r, c) * 4;
                let a = lay.index(pr, pc) * 4;
                out[i..i + 3].copy_from_slice(&pos[a..a + 3]);
                out[i + 3] = d * spacing;
            }
        }
    }
    out
}

/// Simulation settings. Units: one grid spacing is about a centimeter,
/// time in seconds.
#[derive(Clone, Copy, Debug)]
pub struct Physics {
    pub dt: f32,
    pub gravity: f32,
    pub damping: f32,
    pub wind: [f32; 3],
    pub time: f32,
    /// x, y, z, radius.
    pub sphere: [f32; 4],
    pub floor: f32,
    /// Stiffness of structural, shear and bend constraints, 0..1.
    pub stiffness: [f32; 3],
    pub spacing: f32,
}

impl Physics {
    pub fn demo(lay: Layout) -> Self {
        let h = lay.rows as f32;
        Physics {
            dt: 1.0 / 240.0,
            gravity: -981.0,
            damping: 0.998,
            wind: [0.0, 0.0, 600.0],
            time: 0.0,
            sphere: [0.0, -h * 0.15, 46.0, 40.0],
            floor: -h * 0.5 - 24.0,
            stiffness: [1.0, 0.9, 0.6],
            spacing: 1.0,
        }
    }

    pub fn write(&self, p: &mut [f32]) {
        p[P_DT] = self.dt;
        p[P_GRAVITY] = self.gravity;
        p[P_DAMPING] = self.damping;
        p[P_WIND..P_WIND + 3].copy_from_slice(&self.wind);
        p[P_TIME] = self.time;
        p[P_SPHERE..P_SPHERE + 4].copy_from_slice(&self.sphere);
        p[P_FLOOR] = self.floor;
        p[P_STIFF..P_STIFF + 3].copy_from_slice(&self.stiffness);
        p[P_SPACING] = self.spacing;
    }
}

/// A camera orbiting a target: yaw around y, pitch around x, at `dist`.
/// Camera space looks down +z with y up; the screen is `width` x `height`
/// pixels with a pinhole of `focal` pixels.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub target: [f32; 3],
    pub focal: f32,
    pub width: usize,
    pub height: usize,
    /// Direction toward the light, in camera space, unit length.
    pub light: [f32; 3],
}

impl Camera {
    pub fn demo(lay: Layout, width: usize, height: usize) -> Self {
        let l = [-0.35f32, 0.55, -0.75];
        let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        Camera {
            yaw: -0.55,
            pitch: 0.25,
            dist: lay.cols.max(lay.rows) as f32 * 1.7,
            target: [0.0, -(lay.rows as f32) * 0.08, 10.0],
            focal: width as f32 * 0.95,
            width,
            height,
            light: [l[0] / n, l[1] / n, l[2] / n],
        }
    }

    /// World-to-camera rotation, row major. Camera space = R * (p - target)
    /// + (0, 0, dist).
    pub fn rotation(&self) -> [f32; 9] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        // Rx(pitch) * Ry(yaw).
        [
            cy,
            0.0,
            sy, //
            sp * sy,
            cp,
            -sp * cy, //
            -cp * sy,
            sp,
            cp * cy,
        ]
    }

    /// A camera-space point back to world space.
    pub fn unproject(&self, q: [f32; 3]) -> [f32; 3] {
        let r = self.rotation();
        let q = [q[0], q[1], q[2] - self.dist];
        // R is orthonormal, so the inverse is the transpose.
        [
            r[0] * q[0] + r[3] * q[1] + r[6] * q[2] + self.target[0],
            r[1] * q[0] + r[4] * q[1] + r[7] * q[2] + self.target[1],
            r[2] * q[0] + r[5] * q[1] + r[8] * q[2] + self.target[2],
        ]
    }

    /// World point to camera space (x, y, depth).
    pub fn camera_space(&self, p: [f32; 3]) -> [f32; 3] {
        let r = self.rotation();
        let d = [
            p[0] - self.target[0],
            p[1] - self.target[1],
            p[2] - self.target[2],
        ];
        [
            r[0] * d[0] + r[1] * d[1] + r[2] * d[2],
            r[3] * d[0] + r[4] * d[1] + r[5] * d[2],
            r[6] * d[0] + r[7] * d[1] + r[8] * d[2] + self.dist,
        ]
    }

    /// Camera-space point to screen pixels.
    pub fn project(&self, q: [f32; 3]) -> (f32, f32) {
        (
            self.width as f32 * 0.5 + self.focal * q[0] / q[2],
            self.height as f32 * 0.5 - self.focal * q[1] / q[2],
        )
    }

    pub fn write(&self, p: &mut [f32]) {
        p[P_ROT..P_ROT + 9].copy_from_slice(&self.rotation());
        p[P_TARGET..P_TARGET + 3].copy_from_slice(&self.target);
        p[P_DIST] = self.dist;
        p[P_FOCAL] = self.focal;
        p[P_CENTER] = self.width as f32 * 0.5;
        p[P_CENTER + 1] = self.height as f32 * 0.5;
        p[P_LIGHT..P_LIGHT + 3].copy_from_slice(&self.light);
    }
}

/// The two cloth colors, as a checkerboard of 16x16 particle squares.
pub fn base_color(r: usize, c: usize) -> [f32; 3] {
    if ((r / 16) + (c / 16)).is_multiple_of(2) {
        [0.82, 0.22, 0.18]
    } else {
        [0.92, 0.85, 0.72]
    }
}
