//! Page-locked ("pinned") host memory for fast, allocation-free transfers.
//!
//! `api::copy_host_vec_to_device` allocates a new device tensor and copies
//! from pageable memory, which the driver first has to stage through its own
//! pinned buffer. A `Pinned` buffer is allocated once with `cuMemAllocHost`;
//! the GPU can DMA straight into or out of it, so a frame's upload is one
//! `cuMemcpyHtoDAsync` into the tensor the kernels already read.
//!
//! cuTile has no safe wrapper for this yet, so this goes through `cuda-core`
//! with the tensor's raw device pointer.

use std::sync::Arc;

use cuda_core::simt::memory::{free_host, malloc_host};
use cuda_core::{memcpy_dtoh_async, memcpy_htod_async, Stream};
use cutile::prelude::*;

pub struct Pinned<T: DType> {
    ptr: *mut T,
    len: usize,
}

impl<T: DType> Pinned<T> {
    /// A zeroed pinned buffer of `len` elements.
    pub fn new(stream: &Arc<Stream>, len: usize) -> Result<Self, Error> {
        assert!(len > 0);
        let bytes = len * std::mem::size_of::<T>();
        stream.device().bind_to_thread()?;
        // SAFETY: the context is current; the allocation is freed once, in `drop`.
        let ptr = unsafe { malloc_host(bytes)? } as *mut T;
        // SAFETY: `ptr` is valid for `bytes` bytes, and all-zero is a valid `T`
        // for every numeric dtype.
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes) };
        Ok(Pinned { ptr, len })
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `ptr` is valid for `len` initialized elements.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as above, and `&mut self` makes the access exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Copy this buffer into `dst` and wait for it.
    pub fn upload(&self, dst: &mut Tensor<T>, stream: &Arc<Stream>) -> Result<(), Error> {
        assert_eq!(dst.size(), self.len);
        assert!(dst.is_contiguous());
        stream.device().bind_to_thread()?;
        // SAFETY: both sides hold `len` elements; `&mut dst` means no kernel
        // is borrowing it; the context is current; and we synchronize before
        // returning, so the borrow outlives the copy.
        unsafe {
            memcpy_htod_async(
                dst.device_pointer().cu_deviceptr(),
                self.ptr,
                self.len,
                stream,
            )?;
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Copy `src` into this buffer and wait for it.
    pub fn download(&mut self, src: &Tensor<T>, stream: &Arc<Stream>) -> Result<(), Error> {
        assert_eq!(src.size(), self.len);
        assert!(src.is_contiguous());
        stream.device().bind_to_thread()?;
        // SAFETY: as in `upload`. The copy is queued on the stream the
        // kernels run on, so it sees their finished output.
        unsafe {
            memcpy_dtoh_async(
                self.ptr,
                src.device_pointer().cu_deviceptr(),
                self.len,
                stream,
            )?;
            stream.synchronize()?;
        }
        Ok(())
    }
}

impl<T: DType> Drop for Pinned<T> {
    fn drop(&mut self) {
        // SAFETY: allocated by `malloc_host`, freed once, no transfer in flight
        // (every transfer above synchronizes).
        let _ = unsafe { free_host(self.ptr as *mut std::ffi::c_void) };
    }
}
