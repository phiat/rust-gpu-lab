//! Camera, animation and the per-frame parameter block shared by both
//! renderers.
//!
//! The GPU kernel reads its inputs from a 32-element `f32` device buffer
//! instead of scalar arguments. Scalars would be frozen into a captured CUDA
//! graph, but a buffer can be overwritten before every launch.

use crate::gpu::Quality;

pub const PARAMS: usize = 32;

/// Slots in the parameter block. The kernel reads them by number, so keep
/// the comments in `gpu::kernels::render` in sync with this list.
pub mod slot {
    pub const WIDTH: usize = 0;
    pub const HEIGHT: usize = 1;
    pub const EYE: usize = 2; // x, y, z
    pub const FORWARD: usize = 5; // x, y, z
    pub const RIGHT: usize = 8; // x, y, z
    pub const UP: usize = 11; // x, y, z
    pub const FOCAL: usize = 14;
    pub const TIME: usize = 15;
    pub const SUN: usize = 16; // x, y, z, unit vector toward the sun
    pub const SPIN_SIN: usize = 19;
    pub const SPIN_COS: usize = 20;
    pub const BOB: usize = 21;
    pub const VIEW: usize = 22;
    pub const MAX_STEPS: usize = 23;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum View {
    /// Lit scene.
    Shaded,
    /// Sphere-tracing steps each pixel took.
    Steps,
    /// Steps each 32x32 tile program actually ran (per-tile early exit).
    Tiles,
}

impl View {
    pub fn next(self) -> View {
        match self {
            View::Shaded => View::Steps,
            View::Steps => View::Tiles,
            View::Tiles => View::Shaded,
        }
    }
}

/// Orbit camera around a fixed target.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Radians around the vertical axis; 0 looks down -z.
    pub yaw: f32,
    /// Radians above the horizon.
    pub pitch: f32,
    pub distance: f32,
    /// Vertical field of view in degrees.
    pub fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            yaw: 0.6,
            pitch: 0.28,
            distance: 4.6,
            fov: 60.0,
        }
    }
}

const TARGET: [f32; 3] = [0.0, 0.7, 0.0];

impl Camera {
    pub fn eye(&self) -> [f32; 3] {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        [
            TARGET[0] + self.distance * cp * self.yaw.sin(),
            TARGET[1] + self.distance * sp,
            TARGET[2] + self.distance * cp * self.yaw.cos(),
        ]
    }

    /// Unit forward, right and up vectors.
    pub fn basis(&self) -> [[f32; 3]; 3] {
        let eye = self.eye();
        let forward = normalize(sub(TARGET, eye));
        let right = normalize(cross(forward, [0.0, 1.0, 0.0]));
        let up = cross(right, forward);
        [forward, right, up]
    }
}

pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub camera: Camera,
    pub time: f32,
    pub view: View,
    pub quality: Quality,
}

impl Frame {
    pub fn params(&self) -> [f32; PARAMS] {
        let mut p = [0f32; PARAMS];
        let [forward, right, up] = self.camera.basis();
        p[slot::WIDTH] = self.width as f32;
        p[slot::HEIGHT] = self.height as f32;
        p[slot::EYE..slot::EYE + 3].copy_from_slice(&self.camera.eye());
        p[slot::FORWARD..slot::FORWARD + 3].copy_from_slice(&forward);
        p[slot::RIGHT..slot::RIGHT + 3].copy_from_slice(&right);
        p[slot::UP..slot::UP + 3].copy_from_slice(&up);
        p[slot::FOCAL] = 1.0 / (self.camera.fov.to_radians() * 0.5).tan();
        p[slot::TIME] = self.time;
        p[slot::SUN..slot::SUN + 3].copy_from_slice(&normalize([-0.6, 0.55, 0.45]));
        // Trig for the animation happens once here, not per pixel per step.
        let spin = self.time * 0.7;
        p[slot::SPIN_SIN] = spin.sin();
        p[slot::SPIN_COS] = spin.cos();
        p[slot::BOB] = 1.2 + 0.35 * (self.time * 1.6).sin();
        p[slot::VIEW] = self.view as i32 as f32;
        p[slot::MAX_STEPS] = self.quality.max_steps as f32;
        p
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}
