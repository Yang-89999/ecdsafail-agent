//! Arithmetic modulo an odd prime p < 2^n on n-qubit registers holding values in [0, p).
//!
//! Textbook constructions (compare-and-correct), each exact:
//!   modadd   t = t + a mod p           4n Toffolis
//!   cmodadd  t = t + c*a mod p         5n
//!   moddbl   t = 2t mod p (relabels)   2n
//!   modhalve t = t/2 mod p             2n  (exact inverse of moddbl)
//!   modneg   t = -t mod p (t != 0)     n
//! The secp256k1 field is `Field::secp256k1()`; tests also use small primes.

use super::arith::{add, add_const, lt_into, Bits};
use super::ir::{Prog, V};
use alloy_primitives::U256;

#[derive(Clone, Debug)]
pub struct Field {
    pub n: usize,
    pub p: U256,
}

impl Field {
    pub fn secp256k1() -> Field {
        Field {
            n: 256,
            p: U256::from_str_radix("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", 16).unwrap(),
        }
    }

    /// Bits of v (< 2^256), `len` long, little-endian; bits at or above 256 are 0.
    pub fn bits(v: U256, len: usize) -> Bits {
        (0..len).map(|i| i < 256 && v.bit(i)).collect()
    }

    /// Bits of 2^len - p, `len` long (len = n or n + 1).
    pub fn neg_p(&self, len: usize) -> Bits {
        // 2^len - p = 2^len - 2^n + (2^n - p); for len = n + 1 that is 2^n + (2^n - p).
        let two_n_minus_p = if self.n == 256 { U256::ZERO.wrapping_sub(self.p) } else { (U256::from(1) << self.n) - self.p };
        let mut b = Self::bits(two_n_minus_p, len);
        if len == self.n + 1 {
            b[self.n] = true;
        } else {
            assert_eq!(len, self.n);
        }
        b
    }

    pub fn p_bits(&self, len: usize) -> Bits {
        Self::bits(self.p, len)
    }
}

/// t = t + a mod p.  a, t: n qubits, values in [0, p). a is unchanged.
pub fn modadd(b: &mut Prog, f: &Field, a: &[V], t: &[V]) {
    let n = f.n;
    assert_eq!(a.len(), n);
    assert_eq!(t.len(), n);
    let top = b.alloc();
    add(b, a, t, Some(top)); // (t, top) = t + a, in [0, 2p - 2]
    let mut r: Vec<V> = t.to_vec();
    r.push(top);
    add_const(b, &f.neg_p(n + 1), &r, None); // r = sum - p mod 2^(n+1); top = [sum < p]
    let flag = b.alloc();
    b.cx(top, flag);
    add_const(b, &f.p_bits(n + 1), &r, Some(flag)); // add p back where sum < p; now top = 0
    b.free(top);
    // flag = [no reduction] = [result >= a]; flag ^ [result < a] = 1
    lt_into(b, t, a, flag);
    b.x(flag);
    b.free(flag);
}

/// t = t - a mod p (inverse of modadd).
pub fn modsub(b: &mut Prog, f: &Field, a: &[V], t: &[V]) {
    let m = b.mark();
    modadd(b, f, a, t);
    let g = b.take_from(m);
    b.extend_reversed(&g);
}

/// t = t + ctl*a mod p.
pub fn cmodadd(b: &mut Prog, f: &Field, ctl: V, a: &[V], t: &[V]) {
    let m: Vec<V> = a.iter().map(|&q| b.and_c(ctl, q)).collect();
    modadd(b, f, &m, t);
    for (i, &q) in a.iter().enumerate() {
        b.and_u(ctl, q, m[i]);
    }
}

/// t = t - ctl*a mod p.
pub fn cmodsub(b: &mut Prog, f: &Field, ctl: V, a: &[V], t: &[V]) {
    let m: Vec<V> = a.iter().map(|&q| b.and_c(ctl, q)).collect();
    modsub(b, f, &m, t);
    for (i, &q) in a.iter().enumerate() {
        b.and_u(ctl, q, m[i]);
    }
}

/// 2t mod p with a given id `z` for the new low qubit (allocated here). Returns the new register
/// [z, t_0 .. t_{n-2}]; t_{n-1} is released (its value moved into the arithmetic and cleared).
pub fn moddbl_with(b: &mut Prog, f: &Field, t: &[V], z: V) -> Vec<V> {
    let n = f.n;
    assert_eq!(t.len(), n);
    b.alloc_id(z);
    let mut r: Vec<V> = Vec::with_capacity(n + 1);
    r.push(z);
    r.extend_from_slice(t); // r = 2t, n + 1 bits, top = t_{n-1}
    add_const(b, &f.neg_p(n + 1), &r, None); // r = 2t - p mod 2^(n+1); top = [2t < p]
    let flag = b.alloc();
    b.cx(r[n], flag);
    add_const(b, &f.p_bits(n + 1), &r, Some(flag)); // top = 0
    b.free(r[n]);
    // flag = [no reduction] = [result even]; flag ^ result_0 = 1
    b.cx(r[0], flag);
    b.x(flag);
    b.free(flag);
    r.truncate(n);
    r
}

/// t = 2t mod p. Returns the relabelled register.
pub fn moddbl(b: &mut Prog, f: &Field, t: &[V]) -> Vec<V> {
    let z = b.new_v();
    moddbl_with(b, f, t, z)
}

/// t = t/2 mod p, built as the exact inverse of `moddbl`. Returns the relabelled register.
pub fn modhalve(b: &mut Prog, f: &Field, s: &[V]) -> Vec<V> {
    let n = f.n;
    let fresh = b.new_v();
    let mut t: Vec<V> = s[1..].to_vec();
    t.push(fresh);
    let m = b.mark();
    let out = moddbl_with(b, f, &t, s[0]);
    debug_assert_eq!(out, s.to_vec());
    let g = b.take_from(m);
    b.extend_reversed(&g);
    assert_eq!(t.len(), n);
    t
}

/// t = -t mod p, for t != 0 (t = 0 would give p): bitwise NOT, then + (p + 1) mod 2^n.
pub fn modneg(b: &mut Prog, f: &Field, t: &[V]) {
    for &q in t {
        b.x(q);
    }
    let k = if f.n == 256 { f.p.wrapping_add(U256::from(1)) } else { f.p + U256::from(1) };
    add_const(b, &Field::bits(k, f.n), t, None);
}

/// t = t + r mod p, r a classical register in [0, p) given as bit ids.
pub fn modadd_classical(b: &mut Prog, f: &Field, bits: &[u32], t: &[V]) {
    let m = b.alloc_n(f.n);
    for i in 0..f.n {
        b.x_if(m[i], bits[i]);
    }
    modadd(b, f, &m, t);
    for i in 0..f.n {
        b.x_if(m[i], bits[i]);
    }
    b.free_n(&m);
}

/// t = t - r mod p, r classical.
pub fn modsub_classical(b: &mut Prog, f: &Field, bits: &[u32], t: &[V]) {
    let m = b.alloc_n(f.n);
    for i in 0..f.n {
        b.x_if(m[i], bits[i]);
    }
    modsub(b, f, &m, t);
    for i in 0..f.n {
        b.x_if(m[i], bits[i]);
    }
    b.free_n(&m);
}
