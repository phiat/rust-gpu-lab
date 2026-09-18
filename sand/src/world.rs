//! Elements, the cell encoding, the buffer layout, and host-side painting.
//!
//! A cell is an `i32`: the element in the low byte, a 4-bit shade in bits
//! 8-11 so a grain keeps its tint as it moves, and for liquids a direction
//! bit (`DIR`, bit 12: set means "flowing right"). The world is
//! surrounded by a ring of ghost tiles filled with `WALL`, so no rule ever
//! has to ask "is this the edge?".

use crate::gpu::TILE;

pub const EMPTY: i32 = 0;
pub const SAND: i32 = 1;
pub const WATER: i32 = 2;
pub const OIL: i32 = 3;
pub const WOOD: i32 = 4;
pub const WALL: i32 = 5;
pub const FIRE: i32 = 6;
pub const SMOKE: i32 = 7;
/// Wood that is on fire: stays put, throws off flames, burns out.
pub const EMBER: i32 = 8;

pub const NAMES: [&str; 9] = [
    "empty", "sand", "water", "oil", "wood", "wall", "fire", "smoke", "ember",
];

pub fn cell(kind: i32, shade: i32) -> i32 {
    kind | ((shade & 15) << 8)
}

/// Liquids flow in one direction until blocked, then turn around.
pub const DIR: i32 = 1 << 12;

pub fn kind(cell: i32) -> i32 {
    cell & 255
}

/// Set in a paint value to mean "replace the cell", so painting `EMPTY`
/// (erasing) is distinguishable from not painting.
pub const PAINT: i32 = 1 << 20;

/// World size in cells, rounded up to whole tiles. Buffers add a ring of
/// one ghost tile on every side.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub rows: usize,
    pub cols: usize,
}

impl Layout {
    pub fn new(height: usize, width: usize) -> Self {
        Layout {
            rows: height.div_ceil(TILE).max(1) * TILE,
            cols: width.div_ceil(TILE).max(1) * TILE,
        }
    }

    pub fn buf_rows(&self) -> usize {
        self.rows + 2 * TILE
    }

    pub fn buf_cols(&self) -> usize {
        self.cols + 2 * TILE
    }

    /// Buffer index of world cell (x, y).
    pub fn index(&self, x: usize, y: usize) -> usize {
        (y + TILE) * self.buf_cols() + x + TILE
    }

    /// True for buffer cells inside the world (not the ghost ring).
    pub fn is_world(&self, r: usize, c: usize) -> bool {
        (TILE..TILE + self.rows).contains(&r) && (TILE..TILE + self.cols).contains(&c)
    }

    /// A buffer with the ghost ring set to `WALL` and the world empty.
    pub fn empty_buffer(&self) -> Vec<i32> {
        let (br, bc) = (self.buf_rows(), self.buf_cols());
        let mut buf = vec![WALL; br * bc];
        for r in TILE..TILE + self.rows {
            buf[r * bc + TILE..r * bc + TILE + self.cols].fill(EMPTY);
        }
        buf
    }
}

/// Tiny deterministic RNG for shades and scenes.
#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as u32
    }

    /// A cell of `kind` with a random shade and direction.
    pub fn cell(&mut self, kind: i32) -> i32 {
        let bits = self.next() as i32;
        cell(kind, bits & 15) | (bits & DIR)
    }
}

/// Host-side buffer in the GPU layout: either a whole world, or a frame's
/// worth of paint (`PAINT | cell` where the brush touched, 0 elsewhere).
#[derive(Clone)]
pub struct Canvas {
    pub layout: Layout,
    pub cells: Vec<i32>,
    pub rng: Rng,
}

impl Canvas {
    pub fn blank(layout: Layout) -> Self {
        Canvas {
            layout,
            cells: vec![0; layout.buf_rows() * layout.buf_cols()],
            rng: Rng(0x9E37_79B9_7F4A_7C15),
        }
    }

    pub fn clear(&mut self) {
        self.cells.fill(0);
    }

    pub fn is_empty(&self) -> bool {
        self.cells.iter().all(|&c| c == 0)
    }

    fn set(&mut self, x: i32, y: i32, value: i32) {
        let lay = self.layout;
        if x >= 0 && y >= 0 && (x as usize) < lay.cols && (y as usize) < lay.rows {
            let i = lay.index(x as usize, y as usize);
            self.cells[i] = value;
        }
    }

    /// Paint a disc of `kind`, each cell with its own random shade.
    pub fn disc(&mut self, cx: f32, cy: f32, radius: f32, kind: i32) {
        let r = radius.ceil() as i32;
        let (x0, y0) = (cx.round() as i32, cy.round() as i32);
        for y in y0 - r..=y0 + r {
            for x in x0 - r..=x0 + r {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                if dx * dx + dy * dy <= radius * radius {
                    let v = self.rng.cell(kind);
                    self.set(x, y, PAINT | v);
                }
            }
        }
    }

    pub fn rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, kind: i32) {
        for y in y0..y1 {
            for x in x0..x1 {
                let v = self.rng.cell(kind);
                self.set(x, y, PAINT | v);
            }
        }
    }

    /// A thick line of discs, for brush strokes.
    pub fn stroke(&mut self, from: (f32, f32), to: (f32, f32), radius: f32, kind: i32) {
        let len = ((to.0 - from.0).powi(2) + (to.1 - from.1).powi(2)).sqrt();
        let n = (len / (radius * 0.5).max(0.5)).ceil().max(1.0) as i32;
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let p = (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t);
            self.disc(p.0, p.1, radius, kind);
        }
    }

    /// Paint values as plain cells (the `PAINT` flag stripped), for
    /// building a whole world on the host.
    pub fn into_world(mut self) -> Vec<i32> {
        let base = self.layout.empty_buffer();
        for (c, b) in self.cells.iter_mut().zip(&base) {
            *c = if *c & PAINT != 0 { *c & !PAINT } else { *b };
        }
        self.cells
    }

    /// Shelves, a wooden hut, a pool, a dune and a slick of oil.
    pub fn demo(layout: Layout) -> Self {
        let mut c = Canvas::blank(layout);
        let (w, h) = (layout.cols as i32, layout.rows as i32);
        // Shelves.
        c.rect(w / 8, h * 3 / 8, w * 3 / 8, h * 3 / 8 + 4, WALL);
        c.rect(w * 5 / 8, h / 2, w * 7 / 8, h / 2 + 4, WALL);
        // A wooden hut on the floor.
        let (hx, hy) = (w * 5 / 8, h - 1);
        c.rect(hx, hy - 40, hx + 4, hy, WOOD);
        c.rect(hx + 60, hy - 40, hx + 64, hy, WOOD);
        c.rect(hx - 4, hy - 44, hx + 68, hy - 40, WOOD);
        // A pool on the left shelf, a dune on the right, oil on the floor.
        c.rect(w / 8 + 4, h * 3 / 8 - 24, w * 3 / 8 - 4, h * 3 / 8, WATER);
        c.disc(w as f32 * 0.75, h as f32 * 0.5 - 20.0, 22.0, SAND);
        c.rect(w / 16, h - 12, w * 5 / 16, h, OIL);
        c
    }

    /// Random blobs of every element, for correctness checks.
    pub fn random(layout: Layout, seed: u64) -> Self {
        let mut c = Canvas::blank(layout);
        c.rng = Rng(seed | 1);
        let (w, h) = (layout.cols as u32, layout.rows as u32);
        for i in 0..120 {
            let kind = 1 + i % 7;
            let (x, y) = ((c.rng.next() % w) as f32, (c.rng.next() % h) as f32);
            let r = 2.0 + (c.rng.next() % 14) as f32;
            c.disc(x, y, r, kind);
        }
        c
    }
}
