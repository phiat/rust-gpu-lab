//! GPU Game of Life: a 3x3 stencil tile kernel on a torus, driven either
//! eagerly (one sync per generation) or by replaying a captured CUDA graph.

use std::sync::Arc;

use cuda_core::Stream;
use cutile::prelude::*;

use crate::Layout;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// One Game of Life generation for one tile.
    ///
    /// A stencil reads cells at offset -1, but a tensor view can only start
    /// at a non-negative offset. So split the offset: d = q*B + s with
    /// 0 <= s < B, and read offset d as "host view offset by s, loaded at
    /// block index i + q". Offset -1 is a view offset by B-1 loaded at block
    /// i-1; offset +1 is a view offset by 1 loaded at block i.
    ///
    /// Block i-1 doesn't exist for i = 0, which is why the world carries a
    /// ring of ghost tiles. A ghost tile computes the interior tile it
    /// mirrors on the far side of the torus, so the ring stays correct
    /// without a separate copy pass.
    ///
    /// The nine views are named by row offset (u/m/d: up, middle, down) and
    /// then column offset (l/m/r: left, middle, right).
    #[cutile::entry()]
    pub fn life_step<const B: i32>(
        out: &mut Tensor<u8, { [B, B] }>,
        ul: &Tensor<u8, { [-1, -1] }>,
        um: &Tensor<u8, { [-1, -1] }>,
        ur: &Tensor<u8, { [-1, -1] }>,
        ml: &Tensor<u8, { [-1, -1] }>,
        mm: &Tensor<u8, { [-1, -1] }>,
        mr: &Tensor<u8, { [-1, -1] }>,
        dl: &Tensor<u8, { [-1, -1] }>,
        dm: &Tensor<u8, { [-1, -1] }>,
        dr: &Tensor<u8, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        // `mm` is the unshifted buffer: (rows + 2) x (cols + 2) tiles.
        let mm_shape = mm.shape();
        let tile_rows: i32 = mm_shape[0] / B - 2i32;
        let tile_cols: i32 = mm_shape[1] / B - 2i32;

        // Ghost tiles alias the interior tile on the opposite edge.
        let i: i32 = if pid.0 == 0i32 { tile_rows } else { pid.0 };
        let i: i32 = if pid.0 == tile_rows + 1i32 { 1i32 } else { i };
        let j: i32 = if pid.1 == 0i32 { tile_cols } else { pid.1 };
        let j: i32 = if pid.1 == tile_cols + 1i32 { 1i32 } else { j };
        let up: i32 = i - 1i32;
        let left: i32 = j - 1i32;

        let p_ul = ul.partition(shape![B, B]);
        let p_um = um.partition(shape![B, B]);
        let p_ur = ur.partition(shape![B, B]);
        let p_ml = ml.partition(shape![B, B]);
        let p_mm = mm.partition(shape![B, B]);
        let p_mr = mr.partition(shape![B, B]);
        let p_dl = dl.partition(shape![B, B]);
        let p_dm = dm.partition(shape![B, B]);
        let p_dr = dr.partition(shape![B, B]);
        let t_ul: Tile<u8, { [B, B] }> = p_ul.load([up, left]);
        let t_um: Tile<u8, { [B, B] }> = p_um.load([up, j]);
        let t_ur: Tile<u8, { [B, B] }> = p_ur.load([up, j]);
        let t_ml: Tile<u8, { [B, B] }> = p_ml.load([i, left]);
        let t_mm: Tile<u8, { [B, B] }> = p_mm.load([i, j]);
        let t_mr: Tile<u8, { [B, B] }> = p_mr.load([i, j]);
        let t_dl: Tile<u8, { [B, B] }> = p_dl.load([i, left]);
        let t_dm: Tile<u8, { [B, B] }> = p_dm.load([i, j]);
        let t_dr: Tile<u8, { [B, B] }> = p_dr.load([i, j]);

        // With `sum` over the full 3x3 block (the cell included), a cell is
        // alive next generation iff sum == 3 (born, or survives with 2
        // neighbors) or sum == 4 and it is alive (survives with 3).
        let sum: Tile<u8, { [B, B] }> =
            t_ul + t_um + t_ur + t_ml + t_mm + t_mr + t_dl + t_dm + t_dr;
        let s: Shape<{ [B, B] }> = out.shape();
        let zero: Tile<u8, { [B, B] }> = constant(0u8, s);
        let one: Tile<u8, { [B, B] }> = constant(1u8, s);
        let three: Tile<u8, { [B, B] }> = constant(3u8, s);
        let four: Tile<u8, { [B, B] }> = constant(4u8, s);
        let is3: Tile<bool, { [B, B] }> = eq_tile(sum, three);
        let is4: Tile<bool, { [B, B] }> = eq_tile(sum, four);
        let survives: Tile<u8, { [B, B] }> = select(is4, t_mm, zero);
        out.store(select(is3, one, survives));
    }
}

/// The nine offset views of `src`, in `life_step` argument order.
fn neighborhood<'a>(src: &'a Tensor<u8>, lay: &Layout) -> Result<[TensorView<'a, u8>; 9], Error> {
    let (rows, cols) = (lay.buf_rows(), lay.buf_cols());
    // Offset -1 is (B-1 at block i-1), 0 is (0 at i), +1 is (1 at i).
    let offsets = [lay.tile - 1, 0, 1];
    let view = |r: usize, c: usize| src.slice(&[offsets[r]..rows, offsets[c]..cols]);
    Ok([
        view(0, 0)?,
        view(0, 1)?,
        view(0, 2)?,
        view(1, 0)?,
        view(1, 1)?,
        view(1, 2)?,
        view(2, 0)?,
        view(2, 1)?,
        view(2, 2)?,
    ])
}

/// Launch op writing one generation of `n`'s source into `dst`.
fn step_op<'a>(
    dst: &'a mut Tensor<u8>,
    n: &'a [TensorView<'a, u8>; 9],
    tile: usize,
) -> impl GraphNode + 'a {
    kernels::life_step(
        dst.partition([tile, tile]),
        &n[0],
        &n[1],
        &n[2],
        &n[3],
        &n[4],
        &n[5],
        &n[6],
        &n[7],
        &n[8],
    )
}

/// World state on the GPU: two padded buffers used ping-pong.
pub struct World {
    pub layout: Layout,
    stream: Arc<Stream>,
    /// Current generation.
    front: Tensor<u8>,
    back: Tensor<u8>,
}

impl World {
    pub fn new(stream: &Arc<Stream>, layout: Layout, cells: &[u8]) -> Result<Self, Error> {
        let shape = [layout.buf_rows(), layout.buf_cols()];
        let front = api::zeros::<u8>(&shape).sync_on(stream)?;
        let back = api::zeros::<u8>(&shape).sync_on(stream)?;
        let mut world = World {
            layout,
            stream: stream.clone(),
            front,
            back,
        };
        world.load(cells)?;
        Ok(world)
    }

    /// Replace the current generation with host `cells` (rows x cols).
    ///
    /// This copies into the existing buffer instead of allocating a new one,
    /// so a CUDA graph captured against it stays valid.
    pub fn load(&mut self, cells: &[u8]) -> Result<(), Error> {
        let padded = Arc::new(self.layout.pad(cells));
        let src = api::copy_host_vec_to_device(&padded)
            .sync_on(&self.stream)?
            .reshape(&[self.layout.buf_rows(), self.layout.buf_cols()])?;
        api::memcpy(&mut self.front, &src).sync_on(&self.stream)?;
        Ok(())
    }

    /// Advance one generation, synchronizing afterwards.
    pub fn step_eager(&mut self) -> Result<(), Error> {
        {
            let n = neighborhood(&self.front, &self.layout)?;
            step_op(&mut self.back, &n, self.layout.tile).sync_on(&self.stream)?;
        }
        std::mem::swap(&mut self.front, &mut self.back);
        Ok(())
    }

    /// Capture `gens` generations (rounded up to even) as one CUDA graph.
    ///
    /// Each graph step bakes in which buffer it reads and writes, so the
    /// eager `swap` can't be used. Instead the graph ping-pongs
    /// front -> back -> front and always leaves the result in `front`.
    /// Capture only records; the state doesn't advance until `launch`.
    pub fn capture(&mut self, gens: usize) -> Result<CudaGraph<()>, Error> {
        let pairs = gens.div_ceil(2).max(1);
        let lay = self.layout;
        let (front, back) = (&mut self.front, &mut self.back);
        let graph = CudaGraph::scope(&self.stream, |s| {
            for _ in 0..pairs {
                {
                    let n = neighborhood(front, &lay)?;
                    s.record(step_op(back, &n, lay.tile))?;
                }
                {
                    let n = neighborhood(back, &lay)?;
                    s.record(step_op(front, &n, lay.tile))?;
                }
            }
            Ok(())
        })?;
        Ok(graph)
    }

    /// Copy the current generation (without ghost tiles) to the host.
    pub fn download(&self) -> Result<Vec<u8>, Error> {
        let padded = self.front.dup().to_host_vec().sync_on(&self.stream)?;
        Ok(self.layout.unpad(&padded))
    }
}
