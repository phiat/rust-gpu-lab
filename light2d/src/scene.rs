//! Host-side scene: a grid of packed pixels the user paints into.
//!
//! Each pixel is `kind << 24 | r << 16 | g << 8 | b`: kind 0 is empty space,
//! 1 a wall (blocks light, emits nothing), 2 a light (blocks and emits its
//! color). The grid is the GPU buffer layout, ghost ring included, so it
//! uploads as-is.

use crate::gpu::TILE;
use crate::Layout;

pub const WALL: i32 = 1;
pub const LIGHT: i32 = 2;

pub fn pack(kind: i32, rgb: [u8; 3]) -> i32 {
    (kind << 24) | ((rgb[0] as i32) << 16) | ((rgb[1] as i32) << 8) | rgb[2] as i32
}

#[derive(Clone)]
pub struct Canvas {
    pub layout: Layout,
    pub cells: Vec<i32>,
}

impl Canvas {
    pub fn new(layout: Layout) -> Self {
        Canvas {
            layout,
            cells: vec![0; layout.buf_rows() * layout.buf_cols()],
        }
    }

    pub fn clear(&mut self) {
        self.cells.fill(0);
    }

    /// Set a world pixel (the ghost ring is not paintable).
    fn set(&mut self, x: i32, y: i32, value: i32) {
        let lay = self.layout;
        if x >= 0 && y >= 0 && (x as usize) < lay.cols && (y as usize) < lay.rows {
            let (r, c) = (y as usize + TILE, x as usize + TILE);
            self.cells[r * lay.buf_cols() + c] = value;
        }
    }

    pub fn disc(&mut self, cx: f32, cy: f32, radius: f32, value: i32) {
        let r = radius.ceil() as i32;
        let (x0, y0) = (cx.round() as i32, cy.round() as i32);
        for y in y0 - r..=y0 + r {
            for x in x0 - r..=x0 + r {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                if dx * dx + dy * dy <= radius * radius {
                    self.set(x, y, value);
                }
            }
        }
    }

    pub fn rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, value: i32) {
        for y in y0..y1 {
            for x in x0..x1 {
                self.set(x, y, value);
            }
        }
    }

    /// A thick line of discs, for brush strokes.
    pub fn stroke(&mut self, from: (f32, f32), to: (f32, f32), radius: f32, value: i32) {
        let len = ((to.0 - from.0).powi(2) + (to.1 - from.1).powi(2)).sqrt();
        let n = (len / (radius * 0.5).max(0.5)).ceil().max(1.0) as i32;
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let p = (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t);
            self.disc(p.0, p.1, radius, value);
        }
    }

    /// Walls and lamps to start from.
    pub fn demo(layout: Layout) -> Self {
        let mut c = Canvas::new(layout);
        let (w, h) = (layout.cols as i32, layout.rows as i32);
        let wall = pack(WALL, [70, 70, 80]);
        // A room with doorways.
        c.rect(w / 5, h / 5, w * 4 / 5, h / 5 + 6, wall);
        c.rect(w / 5, h * 4 / 5 - 6, w * 2 / 5, h * 4 / 5, wall);
        c.rect(w * 3 / 5, h * 4 / 5 - 6, w * 4 / 5, h * 4 / 5, wall);
        c.rect(w / 5, h / 5, w / 5 + 6, h * 3 / 5, wall);
        c.rect(w * 4 / 5 - 6, h * 2 / 5, w * 4 / 5, h * 4 / 5, wall);
        // Pillars.
        for i in 0..4 {
            let x = w * (3 + 2 * i) / 12;
            c.disc(x as f32, h as f32 * 0.5, 7.0, wall);
        }
        // Lamps.
        c.disc(
            w as f32 * 0.1,
            h as f32 * 0.12,
            9.0,
            pack(LIGHT, [255, 190, 110]),
        );
        c.disc(
            w as f32 * 0.9,
            h as f32 * 0.9,
            7.0,
            pack(LIGHT, [90, 200, 255]),
        );
        c.rect(
            w * 2 / 5 + 20,
            h * 4 / 5 - 4,
            w * 3 / 5 - 20,
            h * 4 / 5,
            pack(LIGHT, [255, 80, 180]),
        );
        c
    }

    /// Random walls, lights and isolated dots, for correctness checks.
    pub fn random(layout: Layout, seed: u64) -> Self {
        let mut rng = seed | 1;
        let mut next = move || {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as u32
        };
        let mut c = Canvas::new(layout);
        let (w, h) = (layout.cols as u32, layout.rows as u32);
        for i in 0..40 {
            let kind = if i % 3 == 0 { LIGHT } else { WALL };
            let color = [next() as u8, next() as u8, next() as u8];
            let (x, y) = ((next() % w) as f32, (next() % h) as f32);
            c.disc(x, y, 2.0 + (next() % 12) as f32, pack(kind, color));
        }
        // Isolated dots make a Voronoi diagram, where plain JFA makes mistakes.
        for _ in 0..3000 {
            let (x, y) = ((next() % w) as i32, (next() % h) as i32);
            c.set(x, y, pack(WALL, [200, 200, 200]));
        }
        c
    }
}
