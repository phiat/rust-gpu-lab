//! CPU reference for one Margolus pass, with the kernel's exact integer
//! math, so the GPU can be checked cell for cell.
//!
//! The block rules live here in plain Rust, and `gpu.rs` spells the same
//! rules out on tiles. Keep the two in the same order.

// Mirrors the kernel expression for expression.
#![allow(clippy::assign_op_pattern)]

use rayon::prelude::*;

use crate::gpu::TILE;
use crate::world::*;

/// How readily a cell sinks. Solids never move; gases rise through empty
/// space because empty (2) is "heavier" than they are.
pub fn density(kind: i32) -> i32 {
    match kind {
        SMOKE => 0,
        FIRE => 1,
        EMPTY => 2,
        OIL => 3,
        WATER => 4,
        SAND => 5,
        _ => 9, // WOOD, WALL, EMBER
    }
}

pub fn solid(kind: i32) -> bool {
    kind == WOOD || kind == WALL || kind == EMBER
}

pub fn liquid(kind: i32) -> bool {
    kind == WATER || kind == OIL
}

/// Falls straight down and slides diagonally: powders and liquids.
pub fn falls(kind: i32) -> bool {
    kind == SAND || kind == WATER || kind == OIL
}

/// Rises, the mirror image of `falls`.
pub fn rises(kind: i32) -> bool {
    kind == FIRE || kind == SMOKE
}

/// May trade places sideways: fluids and empty space, but not powders.
pub fn spreads(kind: i32) -> bool {
    kind == EMPTY || kind == WATER || kind == OIL || kind == FIRE || kind == SMOKE
}

/// lowbias32 (Chris Wellons).
pub fn hash(x: i32) -> i32 {
    let x = x ^ (x >> 16);
    let x = x.wrapping_mul(0x7feb_352d);
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x846c_a68bu32 as i32);
    x ^ (x >> 16)
}

/// One random word per block per pass.
pub fn block_random(bx: i32, by: i32, salt: i32) -> i32 {
    hash(bx ^ hash(by ^ hash(salt)))
}

/// What one cell turns into, given the other three cells of its block and
/// a random byte. `byte >> 4` is the shade of anything new.
pub fn react(x: i32, o1: i32, o2: i32, o3: i32, byte: i32) -> i32 {
    let k = kind(x);
    let near = |kk: i32| kind(o1) == kk || kind(o2) == kk || kind(o3) == kk;
    let near_fire = near(FIRE);
    let near_ember = near(EMBER);
    let near_water = near(WATER);
    let shade = byte >> 4;
    let fire = cell(FIRE, shade);
    let smoke = cell(SMOKE, shade);
    if k == FIRE {
        if near_water || byte < 16 {
            smoke
        } else if byte < 20 {
            EMPTY
        } else {
            x
        }
    } else if k == EMPTY {
        if near_ember && byte < 48 {
            fire
        } else {
            x
        }
    } else if k == OIL {
        if (near_fire || near_ember) && byte < 128 {
            fire
        } else {
            x
        }
    } else if k == WOOD {
        if (near_fire || near_ember) && byte < 40 {
            cell(EMBER, shade)
        } else {
            x
        }
    } else if k == EMBER {
        if near_water {
            cell(WOOD, shade)
        } else if byte < 6 {
            smoke
        } else {
            x
        }
    } else if k == SMOKE {
        if byte < 4 {
            EMPTY
        } else {
            x
        }
    } else {
        x
    }
}

/// `top` sinks into `bottom`.
fn sinks(top: i32, bottom: i32) -> bool {
    let (kt, kb) = (kind(top), kind(bottom));
    !solid(kt) && !solid(kb) && density(kt) > density(kb)
}

/// `x` slides diagonally into `into` (a powder or liquid moving down, or a
/// gas moving up: `up` says which).
fn slides(x: i32, into: i32, up: bool) -> bool {
    let (kx, ki) = (kind(x), kind(into));
    let moves = if up { rises(kx) } else { falls(kx) };
    let lighter = if up {
        density(ki) > density(kx)
    } else {
        density(ki) < density(kx)
    };
    moves && !solid(ki) && lighter
}

/// Sideways movement for the pair `x` (left), `y` (right). A liquid that
/// didn't just fall keeps going in its `DIR` direction while it can and
/// turns around when it can't. Gases wander at random.
fn flow(x: i32, y: i32, x_fell: bool, y_fell: bool, bit: bool) -> (i32, i32) {
    let (kx, ky) = (kind(x), kind(y));
    let open = spreads(kx) && spreads(ky) && kx != ky;
    let x_right = liquid(kx) && x & DIR != 0 && !x_fell;
    let y_left = liquid(ky) && y & DIR == 0 && !y_fell;
    let gas_pair = !liquid(kx) && !liquid(ky);
    if open && (x_right || y_left || (gas_pair && bit)) {
        (y, x)
    } else {
        (
            if x_right { x ^ DIR } else { x },
            if y_left { y ^ DIR } else { y },
        )
    }
}

/// One pass on a 2x2 block: `[a, b]` over `[c, d]`. Reactions first, then
/// gravity, diagonal slides, and sideways spreading.
pub fn update(a: i32, b: i32, c: i32, d: i32, r: i32) -> [i32; 4] {
    let mut a = react(a, b, c, d, r & 255);
    let mut b = react(b, a, c, d, (r >> 8) & 255);
    let mut c = react(c, a, b, d, (r >> 16) & 255);
    let mut d = react(d, a, b, c, (r >> 24) & 255);
    let r2 = hash(r);

    // Gravity. `fell` remembers who moved vertically this pass.
    let mut fell = [false; 4];
    if sinks(a, c) {
        (a, c) = (c, a);
        (fell[0], fell[2]) = (true, true);
    }
    if sinks(b, d) {
        (b, d) = (d, b);
        (fell[1], fell[3]) = (true, true);
    }
    // Diagonals, in a random order so piles don't lean.
    let left_first = r2 & 1 == 0;
    for i in 0..2 {
        if (i == 0) == left_first {
            if slides(a, d, false) {
                (a, d) = (d, a);
                (fell[0], fell[3]) = (true, true);
            }
        } else if slides(b, c, false) {
            (b, c) = (c, b);
            (fell[1], fell[2]) = (true, true);
        }
    }
    for i in 0..2 {
        if (i == 0) == left_first {
            if slides(c, b, true) {
                (c, b) = (b, c);
                (fell[1], fell[2]) = (true, true);
            }
        } else if slides(d, a, true) {
            (d, a) = (a, d);
            (fell[0], fell[3]) = (true, true);
        }
    }
    // Sideways.
    let (a, b) = flow(a, b, fell[0], fell[1], (r2 >> 1) & 1 == 1);
    let (c, d) = flow(c, d, fell[2], fell[3], (r2 >> 2) & 1 == 1);
    [a, b, c, d]
}

/// One pass over the whole buffer. Blocks start at rows and columns with
/// the parity of `pass`; `salt` seeds the randomness. Ghost tiles keep
/// their cells.
pub fn step(lay: &Layout, src: &[i32], pass: i32, salt: i32) -> Vec<i32> {
    let cols = lay.buf_cols();
    let parity = (pass & 1) as usize;
    let mut out = src.to_vec();
    // Block rows start at row `parity`, so chunk the buffer from there:
    // chunk i is rows parity + 2i and parity + 2i + 1, one block row, which
    // is independent of every other block row.
    out[parity * cols..]
        .par_chunks_mut(2 * cols)
        .enumerate()
        .for_each(|(i, lines)| {
            if lines.len() < 2 * cols {
                return;
            }
            let r0 = parity + 2 * i;
            let mut c0 = parity;
            while c0 + 1 < cols {
                let idx = [
                    r0 * cols + c0,
                    r0 * cols + c0 + 1,
                    (r0 + 1) * cols + c0,
                    (r0 + 1) * cols + c0 + 1,
                ];
                let r = block_random((c0 / 2) as i32, (r0 / 2) as i32, salt);
                let new = update(src[idx[0]], src[idx[1]], src[idx[2]], src[idx[3]], r);
                for (n, &at) in idx.iter().enumerate() {
                    if lay.is_world(at / cols, at % cols) {
                        lines[at - r0 * cols] = new[n];
                    }
                }
                c0 += 2;
            }
        });
    out
}

/// Apply a frame of paint (`PAINT | cell` where painted).
pub fn paint(src: &[i32], paint: &[i32]) -> Vec<i32> {
    src.iter()
        .zip(paint)
        .map(|(&s, &p)| if p & PAINT != 0 { p & !PAINT } else { s })
        .collect()
}

/// Cells outside the ghost ring, for comparisons.
pub fn world(lay: &Layout, buf: &[i32]) -> Vec<i32> {
    let bc = lay.buf_cols();
    (TILE..TILE + lay.rows)
        .flat_map(|r| buf[r * bc + TILE..r * bc + TILE + lay.cols].iter().copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn show(lay: &Layout, buf: &[i32]) {
        let bc = lay.buf_cols();
        for r in TILE..TILE + lay.rows {
            let line: String = (TILE..TILE + lay.cols)
                .map(|c| match kind(buf[r * bc + c]) {
                    EMPTY => '.',
                    SAND => 's',
                    WATER => 'w',
                    WALL => '#',
                    _ => '?',
                })
                .collect();
            println!("{line}");
        }
    }

    #[test]
    fn water_levels() {
        let lay = Layout::new(32, 64);
        let mut c = Canvas::blank(lay);
        c.disc(32.0, 26.0, 10.0, SAND);
        c.disc(32.0, 8.0, 7.0, WATER);
        let mut buf = c.into_world();
        for pass in 0..200 {
            buf = step(&lay, &buf, pass, pass);
            if pass % 50 == 49 {
                println!("--- after {} passes", pass + 1);
                show(&lay, &buf);
            }
        }
    }
}
