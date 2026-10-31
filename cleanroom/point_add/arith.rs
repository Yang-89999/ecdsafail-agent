//! Integer arithmetic on little-endian registers of virtual qubits.
//!
//! The adder is Gidney's temporary-AND ripple-carry adder (arXiv:1709.06648): carries are computed
//! with one Toffoli each and uncomputed by measurement (AndU), so an n-bit addition costs n - 1
//! Toffolis (n with a carry-out). Every routine leaves its inputs as it found them unless it says
//! otherwise, and every ancilla it allocates is back to |0> and released when it returns.

use super::ir::{Prog, V};

/// Carry block i: given carry c (None for i = 0), compute the next carry from (a_i, t_i).
/// Afterwards a_i ^= c, t_i ^= c (undone by `carry_back` or `sum_back`).
fn carry_fwd(b: &mut Prog, a: V, t: V, c: Option<V>) -> V {
    match c {
        None => b.and_c(a, t),
        Some(c) => {
            b.cx(c, a);
            b.cx(c, t);
            let n = b.and_c(a, t);
            b.cx(c, n);
            n
        }
    }
}

/// Inverse of `carry_fwd`: releases `next`, restores a_i and t_i.
fn carry_back(b: &mut Prog, a: V, t: V, c: Option<V>, next: V) {
    match c {
        None => b.and_u(a, t, next),
        Some(c) => {
            b.cx(c, next);
            b.and_u(a, t, next);
            b.cx(c, t);
            b.cx(c, a);
        }
    }
}

/// Uncompute carry block i while writing the sum bit: releases `next`, restores a_i, t_i = a_i ^ t_i ^ c.
fn sum_back(b: &mut Prog, a: V, t: V, c: Option<V>, next: V) {
    match c {
        None => {
            b.and_u(a, t, next);
            b.cx(a, t);
        }
        Some(c) => {
            b.cx(c, next);
            b.and_u(a, t, next);
            b.cx(c, a);
            b.cx(a, t);
        }
    }
}

/// t += a  (mod 2^n, n = len). With `carry_out = Some(q)`, also q ^= carry out of bit n-1, which makes
/// (t, q) an (n+1)-bit sum when q starts at 0. Toffolis: n - 1, or n with a carry-out.
pub fn add(b: &mut Prog, a: &[V], t: &[V], carry_out: Option<V>) {
    let n = a.len();
    assert_eq!(n, t.len());
    assert!(n >= 1);
    let mut carries: Vec<V> = Vec::with_capacity(n);
    let full = carry_out.is_some();
    let last = if full { n } else { n - 1 }; // number of carry blocks
    let mut c: Option<V> = None;
    for i in 0..last {
        let nx = carry_fwd(b, a[i], t[i], c);
        carries.push(nx);
        c = Some(nx);
    }
    if let Some(q) = carry_out {
        b.cx(carries[n - 1], q);
    } else {
        // top bit: t_{n-1} ^= a_{n-1} ^ c_{n-1}
        b.cx(a[n - 1], t[n - 1]);
        if let Some(c) = c {
            b.cx(c, t[n - 1]);
        }
    }
    for i in (0..last).rev() {
        let prev = if i == 0 { None } else { Some(carries[i - 1]) };
        sum_back(b, a[i], t[i], prev, carries[i]);
    }
}

/// t -= a (mod 2^n), the exact inverse of `add` without carry-out.
pub fn sub(b: &mut Prog, a: &[V], t: &[V]) {
    let m = b.mark();
    add(b, a, t, None);
    let g = b.take_from(m);
    b.extend_reversed(&g);
}

/// Run `f` while a qubit holding carry_out(a + t) is available; a and t are restored afterwards.
/// `f` must not touch a or t (they are temporarily scrambled). Toffolis: n.
pub fn with_carry<F: FnOnce(&mut Prog, V)>(b: &mut Prog, a: &[V], t: &[V], f: F) {
    let n = a.len();
    assert_eq!(n, t.len());
    let mut carries: Vec<V> = Vec::with_capacity(n);
    let mut c: Option<V> = None;
    for i in 0..n {
        let nx = carry_fwd(b, a[i], t[i], c);
        carries.push(nx);
        c = Some(nx);
    }
    f(b, carries[n - 1]);
    for i in (0..n).rev() {
        let prev = if i == 0 { None } else { Some(carries[i - 1]) };
        carry_back(b, a[i], t[i], prev, carries[i]);
    }
}

/// Run `f` while a qubit holding [x < y] (unsigned) is available. x and y are restored; `f` must not
/// touch them. Toffolis: n. Uses x < y  <=>  y + (2^n - 1 - x) >= 2^n.
pub fn with_lt<F: FnOnce(&mut Prog, V)>(b: &mut Prog, x: &[V], y: &[V], f: F) {
    for &q in x {
        b.x(q);
    }
    with_carry(b, y, x, f);
    for &q in x {
        b.x(q);
    }
}

/// out ^= [x < y].
pub fn lt_into(b: &mut Prog, x: &[V], y: &[V], out: V) {
    with_lt(b, x, y, |b, c| b.cx(c, out));
}

/// Controlled swap of two registers: one Toffoli per bit.
pub fn cswap(b: &mut Prog, ctl: V, x: &[V], y: &[V]) {
    assert_eq!(x.len(), y.len());
    for i in 0..x.len() {
        b.cx(y[i], x[i]);
        b.ccx(ctl, x[i], y[i]);
        b.cx(y[i], x[i]);
    }
}

/// Bits of a constant, little-endian, `n` long.
pub type Bits = Vec<bool>;

/// t += k (mod 2^n), k a build-time constant; with `ctl`, only where ctl = 1. The constant is
/// written into a temporary register (X or CX gates), added, and erased. Toffolis: n - 1.
pub fn add_const(b: &mut Prog, k: &Bits, t: &[V], ctl: Option<V>) {
    let n = t.len();
    assert_eq!(k.len(), n);
    if k.iter().all(|&x| !x) {
        return;
    }
    let m = b.alloc_n(n);
    let write = |b: &mut Prog| {
        for i in 0..n {
            if k[i] {
                match ctl {
                    None => b.x(m[i]),
                    Some(c) => b.cx(c, m[i]),
                }
            }
        }
    };
    write(b);
    add(b, &m, t, None);
    write(b);
    b.free_n(&m);
}

/// t += r (mod 2^n) where r is a classical register given as bit ids (one per qubit of t).
pub fn add_classical(b: &mut Prog, bits: &[u32], t: &[V]) {
    let n = t.len();
    assert_eq!(bits.len(), n);
    let m = b.alloc_n(n);
    for i in 0..n {
        b.x_if(m[i], bits[i]);
    }
    add(b, &m, t, None);
    for i in 0..n {
        b.x_if(m[i], bits[i]);
    }
    b.free_n(&m);
}
