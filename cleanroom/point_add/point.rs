//! Point addition (x, y) += (ox, oy) on y^2 = x^3 + 7 over F_p, with (x, y) quantum and the offset
//! classical. Requires x != ox (the harness never draws it) and a finite result.
//!
//!   1. x -= ox, y -= oy                       x = dx, y = dy
//!   2. y = y / x                              y = lambda = dy/dx      (in-place division)
//!   3. x += 3 ox                              x = x1 + 2 ox
//!   4. x -= lambda^2                          x = ox - x3
//!   5. y = y * x                              y = lambda (ox - x3) = y3 + oy   (in-place multiplication)
//!   6. x = -x                                 x = x3 - ox
//!   7. x += ox, y -= oy                       (x3, y3)

use super::euclid::{div_inplace, mul_inplace};
use super::ir::{Lower, Prog, V};
use super::modp::{cmodsub, moddbl, modhalve, modneg, modadd_classical, modsub_classical, Field};

/// x -= lam^2 mod p, as  sum_i lam_i * (2^i lam mod p)  subtracted term by term from x.
pub fn sqsub(b: &mut Prog, f: &Field, lam: &[V], x: &[V]) {
    let n = f.n;
    let mut w = b.alloc_n(n);
    for i in 0..n {
        b.cx(lam[i], w[i]);
    }
    for i in 0..n {
        cmodsub(b, f, lam[i], &w, x);
        if i + 1 < n {
            w = moddbl(b, f, &w);
        }
    }
    for _ in 0..n - 1 {
        w = modhalve(b, f, &w);
    }
    for i in 0..n {
        b.cx(lam[i], w[i]);
    }
    b.free_n(&w);
}

fn flush(b: &mut Prog, low: &mut Option<&mut Lower>) {
    if let Some(l) = low.as_mut() {
        l.emit(&b.g);
        b.g.clear();
        b.g.shrink_to_fit();
    }
}

/// Returns the registers holding (x3, y3). With `low`, gates are lowered phase by phase to bound memory.
pub fn point_add(
    b: &mut Prog,
    mut low: Option<&mut Lower>,
    f: &Field,
    x: &[V],
    y: &[V],
    ox: &[u32],
    oy: &[u32],
    steps: usize,
) -> (Vec<V>, Vec<V>) {
    let x = x.to_vec();
    modsub_classical(b, f, ox, &x);
    modsub_classical(b, f, oy, y);
    flush(b, &mut low);
    let lam = div_inplace(b, f, &x, y, steps);
    flush(b, &mut low);
    for _ in 0..3 {
        modadd_classical(b, f, ox, &x);
    }
    sqsub(b, f, &lam, &x);
    flush(b, &mut low);
    let y = mul_inplace(b, f, &x, &lam, steps);
    flush(b, &mut low);
    modneg(b, f, &x);
    modadd_classical(b, f, ox, &x);
    modsub_classical(b, f, oy, &y);
    flush(b, &mut low);
    (x, y)
}
