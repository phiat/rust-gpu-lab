//! Host-side helpers shared by the demos: pinned transfer buffers, the
//! eager-or-graph `Submit` trait, and the JIT disk cache switch.

mod pinned;

use std::sync::Arc;

use cuda_core::Stream;
use cutile::prelude::*;

pub use pinned::Pinned;

/// Runs a device op either eagerly or as part of a graph capture, so a
/// pipeline is written once for both.
pub trait Submit {
    fn submit<T: Send, N: GraphNode + DeviceOp<Output = T>>(&self, op: N) -> Result<(), Error>;
}

/// Launch and synchronize each op as it's submitted.
pub struct Eager<'a>(pub &'a Arc<Stream>);

impl Submit for Eager<'_> {
    fn submit<T: Send, N: GraphNode + DeviceOp<Output = T>>(&self, op: N) -> Result<(), Error> {
        op.sync_on(self.0)?;
        Ok(())
    }
}

impl Submit for Scope {
    fn submit<T: Send, N: GraphNode + DeviceOp<Output = T>>(&self, op: N) -> Result<(), Error> {
        self.record(op)?;
        Ok(())
    }
}

/// Keep compiled kernels in `~/.cache/cutile/kernels` across runs.
///
/// The cache is opt-in. A hit skips `tileiras` (0.3-30 s per kernel here) but
/// not the IR build, which still costs 40-300 ms per kernel variant. Editing
/// a kernel changes its key and pays the compile again.
pub fn enable_jit_cache() -> Result<(), Box<dyn std::error::Error>> {
    cutile::jit_cache::enable(Arc::new(
        cutile::jit_cache::FileSystemJitStore::default_location()?,
    ));
    Ok(())
}

/// View packed `i32` pixels (what the kernels store) as the `u32` pixels a
/// window wants, without copying.
pub fn as_u32(pixels: &[i32]) -> &[u32] {
    // SAFETY: i32 and u32 have the same size and alignment, and every bit
    // pattern is valid for both.
    unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u32, pixels.len()) }
}
