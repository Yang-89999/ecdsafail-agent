//! In-place modular multiplication y <- x*y and division y <- y/x by a binary Euclidean pass and a
//! Bezout reconstruction (the structure of Schrottenloher, arXiv:2606.02235; our own implementation).
//!
//! Euclidean pass on (u, v) = (p, x), u always odd. One step:
//!   e = v odd;  s = e AND (v < u);  if s: swap(u, v);  if e: v = v - u;  v = v / 2
//! Each step at least halves u*v, so 2n steps always reach (1, 0); afterwards steps are no-ops.
//! The step bits (e, s) are kept as the record; u and v end as constants and are released, so the
//! pass turns x into its record.
//!
//! The pass is a product of linear maps M with M (p, x) = (1, 0). Applying the inverse maps in
//! reverse order to (y, 0), modulo p, gives y (p, x) = (0, x y): per step
//!   B = 2B;  if e: B = B + A;  if s: swap(A, B)          (all mod p)
//! A ends at 0 and is released; B is x*y. Running the Euclidean pass backwards then restores x
//! and clears the record. Division is the same circuit run backwards.

use super::arith::{cswap, sub, with_lt};
use super::ir::{rename, Gate, Prog, V};
use super::modp::{cmodadd, moddbl, Field};

pub struct Record {
    pub e: Vec<V>,
    pub s: Vec<V>,
}

/// Consumes x (its qubits are released) and returns the step record. Requires 1 <= x < p.
pub fn euclid_fwd(b: &mut Prog, f: &Field, x: &[V], steps: usize) -> Record {
    let n = f.n;
    assert_eq!(x.len(), n);
    let u = b.alloc_n(n);
    for i in 0..n {
        if f.p.bit(i) {
            b.x(u[i]);
        }
    }
    let mut v: Vec<V> = x.to_vec();
    let mut rec = Record { e: Vec::with_capacity(steps), s: Vec::with_capacity(steps) };
    for _ in 0..steps {
        let e = b.alloc();
        b.cx(v[0], e);
        let s = b.new_v();
        with_lt(b, &v, &u, |b, lt| b.and_c_into(e, lt, s));
        cswap(b, s, &u, &v);
        let m: Vec<V> = u.iter().map(|&q| b.and_c(e, q)).collect();
        sub(b, &m, &v);
        for i in 0..n {
            b.and_u(e, u[i], m[i]);
        }
        v.rotate_left(1); // v_0 is 0: halving is a relabelling
        rec.e.push(e);
        rec.s.push(s);
    }
    b.x(u[0]); // u = 1
    b.free_n(&u);
    b.free_n(&v); // v = 0
    rec
}

/// (A, B) = (a, 0) -> (0, x*a) using the record of x. Releases a's qubits; returns B.
pub fn bezout(b: &mut Prog, f: &Field, rec: &Record, a: &[V]) -> Vec<V> {
    let n = f.n;
    let mut bb = b.alloc_n(n);
    for k in (0..rec.e.len()).rev() {
        bb = moddbl(b, f, &bb);
        cmodadd(b, f, rec.e[k], a, &bb);
        cswap(b, rec.s[k], a, &bb);
    }
    b.free_n(a); // A = a*p mod p = 0
    bb
}

/// y <- x*y mod p, x unchanged (x != 0). Returns the register now holding the product.
pub fn mul_inplace(b: &mut Prog, f: &Field, x: &[V], y: &[V], steps: usize) -> Vec<V> {
    let m = b.mark();
    let rec = euclid_fwd(b, f, x, steps);
    let fwd: Vec<Gate> = b.g[m..].to_vec();
    let out = bezout(b, f, &rec, y);
    b.extend_reversed(&fwd);
    out
}

/// z <- z/x mod p, x unchanged (x != 0). Returns the register now holding the quotient.
pub fn div_inplace(b: &mut Prog, f: &Field, x: &[V], z: &[V], steps: usize) -> Vec<V> {
    let m = b.mark();
    let rec = euclid_fwd(b, f, x, steps);
    let fwd: Vec<Gate> = b.g[m..].to_vec();
    let q = b.new_vs(f.n); // ids for the quotient
    let m2 = b.mark();
    let out = bezout(b, f, &rec, &q);
    let mut g = b.take_from(m2);
    rename(&mut g, &out, z);
    b.extend_reversed(&g);
    b.extend_reversed(&fwd);
    q
}
