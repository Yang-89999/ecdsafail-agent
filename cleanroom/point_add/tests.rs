//! Unit tests on the harness simulator. Run (in a harness checkout whose src/point_add is this module):
//!   cargo test --release --bin build_circuit -- --nocapture --test-threads 4
//! Every test checks values, that every ancilla is back to |0>, that the global phase is 0, and
//! prints `UNIT <name> toffoli=<per shot> width=<peak qubits>` lines that agents can read as rewards.

use super::arith::*;
use super::euclid::*;
use super::ir::{toffoli_count, Lower, Prog, V};
use super::modp::*;
use super::point::*;
use crate::sim::Simulator;
use alloy_primitives::U256;
use sha3::digest::{ExtendableOutput, Update};
use sha3::Shake256;

const TBIT: u32 = 4096; // scratch bit in tests

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn u256(&mut self) -> U256 {
        U256::from_limbs([self.next(), self.next(), self.next(), self.next()])
    }
    fn bits(&mut self, n: usize) -> U256 {
        let v = self.u256();
        if n >= 256 { v } else { v & ((U256::from(1) << n) - U256::from(1)) }
    }
    fn below(&mut self, m: U256) -> U256 {
        let n = 256 - m.leading_zeros();
        loop {
            let v = self.bits(n);
            if v < m {
                return v;
            }
        }
    }
    fn nonzero_below(&mut self, m: U256) -> U256 {
        loop {
            let v = self.below(m);
            if !v.is_zero() {
                return v;
            }
        }
    }
}

struct Out {
    vals: Vec<Vec<U256>>, // [register][shot]
    toffoli: f64,
    width: u32,
}

/// Lower `b`, bind `ins` to the first physical qubits, run `shots` (<= 64 per batch) and check
/// cleanliness. `cbits` are classical inputs: (bit ids, value per shot).
fn run(b: &Prog, ins: &[(&[V], Vec<U256>)], cbits: &[(&[u32], Vec<U256>)], outs: &[&[V]]) -> Out {
    let shots = ins.first().map(|i| i.1.len()).or(cbits.first().map(|c| c.1.len())).unwrap();
    let mut low = Lower::new(TBIT);
    let mut next = 0u32;
    for (reg, _) in ins {
        for &v in reg.iter() {
            low.bind(v, next);
            next += 1;
        }
    }
    low.emit(&b.g);
    let out_phys: Vec<Vec<u32>> = outs.iter().map(|r| r.iter().map(|&v| low.phys(v)).collect()).collect();
    let nq = low.next_phys as usize;
    let mut vals = vec![vec![U256::ZERO; shots]; outs.len()];
    let mut tof = 0u64;
    let mut xof = {
        let mut h = Shake256::default();
        h.update(b"cleanroom unit test");
        h.finalize_xof()
    };
    let mut start = 0;
    let mut batches = 0u64;
    while start < shots {
        batches += 1;
        let bs = (shots - start).min(64);
        let mask = if bs == 64 { u64::MAX } else { (1u64 << bs) - 1 };
        let mut sim = Simulator::new(nq, TBIT as usize + 1, &mut xof);
        let mut q = 0usize;
        for (reg, v) in ins {
            for i in 0..reg.len() {
                let mut w = 0u64;
                for s in 0..bs {
                    if v[start + s].bit(i) {
                        w |= 1 << s;
                    }
                }
                sim.qubits[q] = w;
                q += 1;
            }
        }
        for (bits, v) in cbits {
            for (i, &bid) in bits.iter().enumerate() {
                let mut w = 0u64;
                for s in 0..bs {
                    if v[start + s].bit(i) {
                        w |= 1 << s;
                    }
                }
                sim.bits[bid as usize] = w;
            }
        }
        sim.apply_iter(low.ops.iter());
        assert_eq!(sim.phase & mask, 0, "phase garbage");
        for (r, ph) in out_phys.iter().enumerate() {
            for s in 0..bs {
                let mut x = U256::ZERO;
                for (i, &p) in ph.iter().enumerate() {
                    if (sim.qubits[p as usize] >> s) & 1 == 1 {
                        x.set_bit(i, true);
                    }
                }
                vals[r][start + s] = x;
            }
            for &p in ph {
                sim.qubits[p as usize] = 0;
            }
        }
        for (i, &w) in sim.qubits.iter().enumerate() {
            assert_eq!(w & mask, 0, "ancilla garbage on physical qubit {i}");
        }
        tof += sim.stats.toffoli_gates;
        start += bs;
    }
    // The simulator counts a gate for all 64 lanes of a batch (as the harness does: 9024 = 141 x 64).
    Out { vals, toffoli: tof as f64 / (batches * 64) as f64, width: low.next_phys }
}

fn report(name: &str, o: &Out) {
    println!("UNIT {name} toffoli={:.1} width={}", o.toffoli, o.width);
}

fn small() -> Field {
    Field { n: 16, p: U256::from(65521u64) }
}
fn odd_small() -> Field {
    Field { n: 12, p: U256::from(2053u64) }
}
fn secp() -> Field {
    Field::secp256k1()
}

fn mask(n: usize) -> U256 {
    if n >= 256 { U256::MAX } else { (U256::from(1) << n) - U256::from(1) }
}

#[test]
fn adder() {
    let mut r = Rng(1);
    for &n in &[1usize, 2, 7, 64, 256] {
        for &carry in &[false, true] {
            let mut b = Prog::new();
            let a = b.new_vs(n);
            let t = b.new_vs(n);
            let co = b.new_vs(1);
            add(&mut b, &a, &t, if carry { Some(co[0]) } else { None });
            let shots = 100;
            let av: Vec<U256> = (0..shots).map(|_| r.bits(n)).collect();
            let tv: Vec<U256> = (0..shots).map(|_| r.bits(n)).collect();
            let cv: Vec<U256> = vec![U256::ZERO; shots];
            let o = run(&b, &[(&a, av.clone()), (&t, tv.clone()), (&co, cv)], &[], &[&a, &t, &co]);
            for s in 0..shots {
                assert_eq!(o.vals[0][s], av[s]);
                let sum = av[s].wrapping_add(tv[s]);
                assert_eq!(o.vals[1][s], sum & mask(n), "n={n} carry={carry}");
                if carry {
                    let c = if n == 256 { av[s].overflowing_add(tv[s]).1 } else { sum.bit(n) };
                    assert_eq!(o.vals[2][s], U256::from(c as u8));
                }
            }
            assert_eq!(o.toffoli as usize, if carry { n } else { n - 1 }, "n={n} carry={carry} ir={}", toffoli_count(&b.g));
            if n == 256 {
                report(&format!("add{}{}", n, if carry { "_carry" } else { "" }), &o);
            }
        }
    }
}

#[test]
fn subtract_and_compare() {
    let mut r = Rng(2);
    for &n in &[1usize, 5, 256] {
        let mut b = Prog::new();
        let a = b.new_vs(n);
        let t = b.new_vs(n);
        let out = b.new_vs(2);
        sub(&mut b, &a, &t);
        lt_into(&mut b, &a, &t, out[0]); // [a < t - a]
        let shots = 128;
        let mut av: Vec<U256> = (0..shots).map(|_| r.bits(n)).collect();
        let tv: Vec<U256> = (0..shots).map(|_| r.bits(n)).collect();
        av[0] = tv[0]; // equal case
        let o = run(&b, &[(&a, av.clone()), (&t, tv.clone()), (&out, vec![U256::ZERO; shots])], &[], &[&a, &t, &out]);
        for s in 0..shots {
            let d = tv[s].wrapping_sub(av[s]) & mask(n);
            assert_eq!(o.vals[1][s], d);
            assert_eq!(o.vals[2][s], U256::from((av[s] < d) as u8));
        }
        if n == 256 {
            report("sub256+lt256", &o);
        }
    }
}

#[test]
fn controlled_swap_and_constants() {
    let mut r = Rng(3);
    let n = 256;
    let k = r.u256();
    let mut b = Prog::new();
    let c = b.new_vs(1);
    let x = b.new_vs(n);
    let y = b.new_vs(n);
    cswap(&mut b, c[0], &x, &y);
    add_const(&mut b, &Field::bits(k, n), &x, Some(c[0]));
    let shots = 64;
    let cv: Vec<U256> = (0..shots).map(|s| U256::from((s % 2) as u8)).collect();
    let xv: Vec<U256> = (0..shots).map(|_| r.u256()).collect();
    let yv: Vec<U256> = (0..shots).map(|_| r.u256()).collect();
    let o = run(&b, &[(&c, cv.clone()), (&x, xv.clone()), (&y, yv.clone())], &[], &[&c, &x, &y]);
    for s in 0..shots {
        let (ex, ey) = if s % 2 == 1 { (yv[s].wrapping_add(k), xv[s]) } else { (xv[s], yv[s]) };
        assert_eq!(o.vals[1][s], ex);
        assert_eq!(o.vals[2][s], ey);
    }
}

fn check_modops(f: &Field, seed: u64, shots: usize) {
    let mut r = Rng(seed);
    let n = f.n;
    let p = f.p;
    // modadd, modsub, cmodadd
    {
        let mut b = Prog::new();
        let c = b.new_vs(1);
        let a = b.new_vs(n);
        let t = b.new_vs(n);
        let u = b.new_vs(n);
        modadd(&mut b, f, &a, &t);
        modsub(&mut b, f, &a, &u);
        cmodadd(&mut b, f, c[0], &a, &u);
        cmodadd(&mut b, f, c[0], &a, &u);
        let mut av: Vec<U256> = (0..shots).map(|_| r.below(p)).collect();
        let mut tv: Vec<U256> = (0..shots).map(|_| r.below(p)).collect();
        let uv: Vec<U256> = (0..shots).map(|_| r.below(p)).collect();
        let cv: Vec<U256> = (0..shots).map(|s| U256::from((s % 3 == 0) as u8)).collect();
        av[0] = p - U256::from(1);
        tv[0] = p - U256::from(1);
        av[1] = U256::ZERO;
        tv[2] = U256::ZERO;
        let o = run(&b, &[(&c, cv.clone()), (&a, av.clone()), (&t, tv.clone()), (&u, uv.clone())], &[], &[&a, &t, &u, &c]);
        for s in 0..shots {
            assert_eq!(o.vals[0][s], av[s]);
            assert_eq!(o.vals[3][s], cv[s]);
            assert_eq!(o.vals[1][s], av[s].add_mod(tv[s], p), "modadd n={n}");
            let mut e = uv[s].add_mod(p - av[s], p);
            if s % 3 == 0 {
                e = e.add_mod(av[s], p).add_mod(av[s], p);
            }
            assert_eq!(o.vals[2][s], e, "modsub/cmodadd n={n}");
        }
        if n == 256 {
            report("modadd+modsub+2cmodadd(secp)", &o);
        }
    }
    // moddbl, modhalve, modneg
    {
        let mut b = Prog::new();
        let t = b.new_vs(n);
        let w = b.new_vs(n);
        let t2 = moddbl(&mut b, f, &t);
        let w2 = modhalve(&mut b, f, &w);
        modneg(&mut b, f, &w2);
        let mut tv: Vec<U256> = (0..shots).map(|_| r.below(p)).collect();
        let mut wv: Vec<U256> = (0..shots).map(|_| r.nonzero_below(p)).collect();
        tv[0] = p - U256::from(1);
        tv[1] = U256::ZERO;
        wv[0] = U256::from(1);
        let o = run(&b, &[(&t, tv.clone()), (&w, wv.clone())], &[], &[&t2, &w2]);
        let inv2 = (p + U256::from(1)) >> 1;
        for s in 0..shots {
            assert_eq!(o.vals[0][s], tv[s].add_mod(tv[s], p), "moddbl n={n}");
            let h = wv[s].mul_mod(inv2, p);
            assert_eq!(o.vals[1][s], (p - h) % p, "modhalve+modneg n={n}");
        }
        if n == 256 {
            report("moddbl+modhalve+modneg(secp)", &o);
        }
    }
    // classical add / sub
    {
        let mut b = Prog::new();
        let t = b.new_vs(n);
        let bits: Vec<u32> = (0..n as u32).collect();
        modsub_classical(&mut b, f, &bits, &t);
        let tv: Vec<U256> = (0..shots).map(|_| r.below(p)).collect();
        let kv: Vec<U256> = (0..shots).map(|_| r.below(p)).collect();
        let o = run(&b, &[(&t, tv.clone())], &[(&bits, kv.clone())], &[&t]);
        for s in 0..shots {
            assert_eq!(o.vals[0][s], tv[s].add_mod(p - kv[s], p), "modsub_classical n={n}");
        }
    }
}

#[test]
fn modular_small() {
    check_modops(&small(), 4, 512);
    check_modops(&odd_small(), 5, 512);
}

#[test]
fn modular_secp() {
    check_modops(&secp(), 6, 128);
}

fn check_mul_div(f: &Field, seed: u64, shots: usize) {
    let mut r = Rng(seed);
    let n = f.n;
    let p = f.p;
    let steps = 2 * n;
    let mut b = Prog::new();
    let x = b.new_vs(n);
    let y = b.new_vs(n);
    let z = b.new_vs(n);
    let y2 = mul_inplace(&mut b, f, &x, &y, steps);
    let m = b.mark();
    let z2 = div_inplace(&mut b, f, &x, &z, steps);
    let div_tof = toffoli_count(&b.g[m..]);
    let mut xv: Vec<U256> = (0..shots).map(|_| r.nonzero_below(p)).collect();
    let yv: Vec<U256> = (0..shots).map(|_| r.below(p)).collect();
    let zv: Vec<U256> = (0..shots).map(|_| r.below(p)).collect();
    xv[0] = U256::from(1);
    xv[1] = p - U256::from(1);
    let o = run(&b, &[(&x, xv.clone()), (&y, yv.clone()), (&z, zv.clone())], &[], &[&x, &y2, &z2]);
    for s in 0..shots {
        assert_eq!(o.vals[0][s], xv[s]);
        assert_eq!(o.vals[1][s], xv[s].mul_mod(yv[s], p), "mul n={n} x={} y={}", xv[s], yv[s]);
        let inv = xv[s].inv_mod(p).unwrap();
        assert_eq!(o.vals[2][s], zv[s].mul_mod(inv, p), "div n={n}");
    }
    println!("UNIT mul+div n={n} toffoli={:.1} (division alone {}) width={}", o.toffoli, div_tof, o.width);
}

#[test]
fn mul_div_small() {
    check_mul_div(&small(), 7, 256);
    check_mul_div(&odd_small(), 8, 256);
}

#[test]
fn mul_div_secp() {
    check_mul_div(&secp(), 9, 64);
}

#[test]
fn squaring_small() {
    let f = small();
    let mut r = Rng(10);
    let n = f.n;
    let p = f.p;
    let mut b = Prog::new();
    let l = b.new_vs(n);
    let x = b.new_vs(n);
    sqsub(&mut b, &f, &l, &x);
    let shots = 256;
    let lv: Vec<U256> = (0..shots).map(|_| r.below(p)).collect();
    let xv: Vec<U256> = (0..shots).map(|_| r.below(p)).collect();
    let o = run(&b, &[(&l, lv.clone()), (&x, xv.clone())], &[], &[&l, &x]);
    for s in 0..shots {
        assert_eq!(o.vals[1][s], xv[s].add_mod(p - lv[s].mul_mod(lv[s], p), p));
    }
}

/// The full point addition against the harness's reference curve arithmetic.
fn check_point(seed: u64, shots: usize) {
    use crate::weierstrass_elliptic_curve::WeierstrassEllipticCurve;
    let f = secp();
    let n = f.n;
    let curve = WeierstrassEllipticCurve {
        modulus: f.p,
        a: U256::from(0),
        b: U256::from(7),
        gx: U256::from_str_radix("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16).unwrap(),
        gy: U256::from_str_radix("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16).unwrap(),
        order: U256::from_str_radix("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16).unwrap(),
    };
    let mut r = Rng(seed);
    let mut b = Prog::new();
    let x = b.new_vs(n);
    let y = b.new_vs(n);
    let ox: Vec<u32> = (0..n as u32).collect();
    let oy: Vec<u32> = (n as u32..2 * n as u32).collect();
    let (x3, y3) = point_add(&mut b, None, &f, &x, &y, &ox, &oy, 2 * n);
    let (mut xs, mut ys, mut oxs, mut oys, mut ex, mut ey) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    while xs.len() < shots {
        let t = curve.mul(curve.gx, curve.gy, r.u256());
        let o = curve.mul(curve.gx, curve.gy, r.u256());
        if t.0 == o.0 {
            continue;
        }
        let e = curve.add(t.0, t.1, o.0, o.1);
        xs.push(t.0);
        ys.push(t.1);
        oxs.push(o.0);
        oys.push(o.1);
        ex.push(e.0);
        ey.push(e.1);
    }
    let o = run(&b, &[(&x, xs), (&y, ys)], &[(&ox, oxs), (&oy, oys)], &[&x3, &y3]);
    for s in 0..shots {
        assert_eq!(o.vals[0][s], ex[s], "x3 shot {s}");
        assert_eq!(o.vals[1][s], ey[s], "y3 shot {s}");
    }
    report("point_add(secp)", &o);
}

#[test]
fn point_addition() {
    check_point(11, 64);
}
