// ─── nonce search (agentlab) ────────────────────────────────────────────────
//
// Everything above this line is the challenge's own src/bin/eval_circuit.rs, copied verbatim at build time
// (scripts/build_nonce_search.sh), so loading, hashing, input derivation and simulation are the official ones.
// The op stream's last 96 ops are X;X identity pairs whose targets encode a 48-bit nonce (TAIL_NONCE); the nonce
// changes only the Fiat-Shamir seed. This tool re-encodes that tail for many nonces and runs the official test on
// each, stopping at the first failing batch of 64 shots.
//
//   nonce_search check  <nonce>                      full official test of one nonce (no early stop)
//   nonce_search search <start> <count> [threads]    early-stop search; prints stats; CLEAN lines for clean draws
//   nonce_search selftest                            fast fixed-base k*G vs the harness's curve.mul
//
// ops.bin is read from the working directory (any nonce in its tail).

use quantum_ecc::circuit::{NO_BIT, NO_QUBIT};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtOrd};
use std::sync::Arc;
use std::time::Instant;

fn absorb_op(h: &mut Shake256, op: &Op) {
    h.update(&[op.kind as u8]);
    h.update(&op.q_control2.0.to_le_bytes());
    h.update(&op.q_control1.0.to_le_bytes());
    h.update(&op.q_target.0.to_le_bytes());
    h.update(&op.c_target.0.to_le_bytes());
    h.update(&op.c_condition.0.to_le_bytes());
    h.update(&op.r_target.0.to_le_bytes());
}

/// SHAKE256 state after the domain tag, the length and every op except the 96-op nonce tail.
fn prefix_state(ops: &[Op]) -> Shake256 {
    let mut h = Shake256::default();
    h.update(b"quantum_ecc-fiat-shamir-v2");
    h.update(&(ops.len() as u64).to_le_bytes());
    for op in &ops[..ops.len() - 96] {
        absorb_op(&mut h, op);
    }
    h
}

fn seed_for_nonce(prefix: &Shake256, tail: &[Op], nonce: u64) -> sha3::Shake256Reader {
    let mut h = prefix.clone();
    for b in 0..48 {
        let t = QubitId((nonce >> b) & 1);
        for k in 0..2 {
            let mut op = tail[2 * b + k].clone();
            op.q_target = t;
            absorb_op(&mut h, &op);
        }
    }
    h.finalize_xof()
}

// ---- fast fixed-base scalar multiplication (same affine result as curve.mul(gx, gy, k)) ----
#[derive(Clone, Copy)]
struct Jac { x: U256, y: U256, z: U256 } // z == 0 means infinity

struct Fixed { p: U256, table: Vec<(U256, U256)> } // table[i] = 2^i * G (affine; (0,0) = infinity)

impl Fixed {
    fn new(curve: &WeierstrassEllipticCurve) -> Self {
        let mut table = Vec::with_capacity(256);
        let mut b = (curve.gx, curve.gy);
        for _ in 0..256 {
            table.push(b);
            b = curve.add(b.0, b.1, b.0, b.1);
        }
        Fixed { p: curve.modulus, table }
    }
    fn sub(&self, a: U256, b: U256) -> U256 { if a >= b { a - b } else { self.p - (b - a) } }
    fn mul(&self, a: U256, b: U256) -> U256 { a.mul_mod(b, self.p) }
    fn add(&self, a: U256, b: U256) -> U256 { a.add_mod(b, self.p) }
    fn dbl(&self, a: Jac) -> Jac {
        // a = 0 (secp256k1)
        if a.z.is_zero() || a.y.is_zero() { return Jac { x: U256::ZERO, y: U256::ZERO, z: U256::ZERO }; }
        let ysq = self.mul(a.y, a.y);
        let s = self.mul(U256::from(4), self.mul(a.x, ysq));
        let m = self.mul(U256::from(3), self.mul(a.x, a.x));
        let x3 = self.sub(self.mul(m, m), self.add(s, s));
        let y3 = self.sub(self.mul(m, self.sub(s, x3)), self.mul(U256::from(8), self.mul(ysq, ysq)));
        let z3 = self.mul(U256::from(2), self.mul(a.y, a.z));
        Jac { x: x3, y: y3, z: z3 }
    }
    /// Jacobian + affine
    fn add_affine(&self, a: Jac, bx: U256, by: U256) -> Jac {
        if bx.is_zero() && by.is_zero() { return a; }
        if a.z.is_zero() { return Jac { x: bx, y: by, z: U256::from(1) }; }
        let z2 = self.mul(a.z, a.z);
        let u2 = self.mul(bx, z2);
        let s2 = self.mul(by, self.mul(z2, a.z));
        if u2 == a.x {
            if s2 == a.y { return self.dbl(a); }
            return Jac { x: U256::ZERO, y: U256::ZERO, z: U256::ZERO };
        }
        let h = self.sub(u2, a.x);
        let r = self.sub(s2, a.y);
        let hh = self.mul(h, h);
        let hhh = self.mul(hh, h);
        let v = self.mul(a.x, hh);
        let x3 = self.sub(self.sub(self.mul(r, r), hhh), self.add(v, v));
        let y3 = self.sub(self.mul(r, self.sub(v, x3)), self.mul(a.y, hhh));
        let z3 = self.mul(a.z, h);
        Jac { x: x3, y: y3, z: z3 }
    }
    fn mul_g(&self, k: U256) -> (U256, U256) {
        let mut acc = Jac { x: U256::ZERO, y: U256::ZERO, z: U256::ZERO };
        for i in 0..256 {
            if k.bit(i) {
                let (tx, ty) = self.table[i];
                acc = self.add_affine(acc, tx, ty);
            }
        }
        if acc.z.is_zero() { return (U256::ZERO, U256::ZERO); }
        let zi = acc.z.inv_mod(self.p).expect("z invertible");
        let zi2 = self.mul(zi, zi);
        (self.mul(acc.x, zi2), self.mul(acc.y, self.mul(zi2, zi)))
    }
}

struct Outcome { batches_ok: usize, batches_total_ok: usize, clean: bool, classical: usize, phase_batches: usize, ancilla_batches: usize, avg_tof: f64, shots: usize }

/// The official test for one nonce: same input derivation and per-batch checks as run_tests above.
fn test_nonce(ops: &[Op], regs: &[Vec<QubitOrBit>], total_qubits: u64, num_bits: u64, curve: &WeierstrassEllipticCurve,
              fixed: &Fixed, mut xof: sha3::Shake256Reader, early_stop: bool) -> Outcome {
    let mut ks: Vec<(U256, U256)> = Vec::with_capacity(NUM_TESTS);
    for _ in 0..NUM_TESTS {
        let mut rb = [[0u8; 32]; 2];
        XofReader::read(&mut xof, &mut rb[0]);
        XofReader::read(&mut xof, &mut rb[1]);
        ks.push((U256::from_le_bytes(rb[0]), U256::from_le_bytes(rb[1])));
    }
    // valid shots, derived lazily in order (skips are those of run_tests)
    let mut next_k = 0usize;
    let mut next_shot = || -> Option<((U256, U256), (U256, U256), (U256, U256))> {
        while next_k < ks.len() {
            let (k1, k2) = ks[next_k];
            next_k += 1;
            let t = fixed.mul_g(k1);
            let o = fixed.mul_g(k2);
            if t.0 == o.0 { continue; }
            if t.0.is_zero() && t.1.is_zero() { continue; }
            if o.0.is_zero() && o.1.is_zero() { continue; }
            let e = curve.add(t.0, t.1, o.0, o.1);
            return Some((t, o, e));
        }
        None
    };
    let mut sim = Simulator::new(total_qubits as usize, num_bits as usize, &mut xof);
    const BATCH: usize = 64;
    let (mut batches_ok, mut classical, mut phase_batches, mut ancilla_batches, mut shots, mut batches_total_ok) = (0, 0, 0, 0, 0, 0usize);
    let mut clean = true;
    loop {
        let mut batch: Vec<_> = Vec::with_capacity(BATCH);
        while batch.len() < BATCH {
            match next_shot() { Some(s) => batch.push(s), None => break }
        }
        if batch.is_empty() { break; }
        let bs = batch.len();
        shots += bs;
        let cond_mask: u64 = if bs == 64 { u64::MAX } else { (1u64 << bs) - 1 };
        sim.clear_for_shot();
        for (shot, (t, o, _)) in batch.iter().enumerate() {
            sim.set_register(&regs[0], t.0, shot);
            sim.set_register(&regs[1], t.1, shot);
            sim.set_register(&regs[2], o.0, shot);
            sim.set_register(&regs[3], o.1, shot);
        }
        sim.apply_iter(ops.iter());
        let mut ok = true;
        for (shot, (_, _, e)) in batch.iter().enumerate() {
            if sim.get_register(&regs[0], shot) != e.0 || sim.get_register(&regs[1], shot) != e.1 {
                classical += 1;
                ok = false;
            }
        }
        if sim.phase & cond_mask != 0 { phase_batches += 1; ok = false; }
        for register in regs {
            for qb in register {
                if let QubitOrBit::Qubit(q) = *qb { *sim.qubit_mut(q) = 0; }
            }
        }
        if (0..total_qubits).any(|q| sim.qubit(QubitId(q)) & cond_mask != 0) { ancilla_batches += 1; ok = false; }
        if ok { batches_total_ok += 1; if clean { batches_ok += 1; } } else { clean = false; if early_stop { break; } }
        if bs < BATCH { break; }
    }
    let avg_tof = sim.stats.toffoli_gates as f64 / shots.max(1) as f64;
    Outcome { batches_ok, batches_total_ok, clean, classical, phase_batches, ancilla_batches, avg_tof, shots }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("help");
    let curve = secp256k1();
    let fixed = Fixed::new(&curve);
    if mode == "shakebench" {
        let mut h = Shake256::default(); h.update(b"x"); let mut x = h.finalize_xof();
        let t = Instant::now(); let mut acc = 0u64; let mut buf = [0u8; 8];
        for _ in 0..778_349 { XofReader::read(&mut x, &mut buf); acc ^= u64::from_le_bytes(buf); }
        let a = t.elapsed().as_secs_f64();
        let t = Instant::now(); let mut big = vec![0u8; 778_349 * 8]; XofReader::read(&mut x, &mut big);
        println!("one batch of rng (778k x 8 bytes): {:.1} ms in 8-byte reads, {:.1} ms bulk ({acc:x} {})", a * 1e3, t.elapsed().as_secs_f64() * 1e3, big[7]);
        return;
    }
    if mode == "selftest" {
        let mut h = Shake256::default();
        h.update(b"nonce-search-selftest");
        let mut x = h.finalize_xof();
        let t0 = Instant::now();
        for i in 0..200 {
            let mut b = [0u8; 32];
            XofReader::read(&mut x, &mut b);
            let k = if i == 0 { U256::ZERO } else if i == 1 { curve.order } else { U256::from_le_bytes(b) };
            let a = curve.mul(curve.gx, curve.gy, k);
            let f = fixed.mul_g(k);
            assert_eq!(a, f, "fixed-base mismatch at {i}");
        }
        let ref_t = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for i in 0..2000u64 { let _ = fixed.mul_g(U256::from(i).wrapping_mul(U256::MAX / U256::from(7))); }
        println!("selftest OK: 200 scalars agree; reference+fast {:.1} ms/mul, fast alone {:.1} us/mul",
                 ref_t * 1e3 / 200.0, t1.elapsed().as_secs_f64() * 1e6 / 2000.0);
        return;
    }
    let ops = load_ops(OPS_PATH).unwrap_or_else(|e| panic!("could not load {OPS_PATH}: {e}"));
    let n = ops.len();
    assert!(n >= 96);
    for op in &ops[n - 96..] { assert!(op.kind == OperationType::X, "tail is not 96 X ops"); }
    let (total_qubits, num_bits, _num_regs, regs) = analyze_ops(ops.iter());
    let t0 = Instant::now();
    let prefix = prefix_state(&ops);
    let tail: Vec<Op> = ops[n - 96..].to_vec();
    eprintln!("loaded {} ops, {} qubits; prefix hashed in {:.1} s", n, total_qubits, t0.elapsed().as_secs_f64());
    match mode {
        "check" => {
            let nonce: u64 = args[2].parse().expect("nonce");
            let t = Instant::now();
            let o = test_nonce(&ops, &regs, total_qubits, num_bits, &curve, &fixed, seed_for_nonce(&prefix, &tail, nonce), false);
            println!("nonce {nonce}: clean={} shots={} classical_failures={} phase_garbage_batches={} ancilla_garbage_batches={} \
                      avg_tof={:.3} first_failing_batch={} ({:.1} s)", o.clean, o.shots, o.classical, o.phase_batches,
                     o.ancilla_batches, o.avg_tof, if o.clean { "-".into() } else { o.batches_ok.to_string() }, t.elapsed().as_secs_f64());
        }
        "search" => {
            let start: u64 = args[2].parse().expect("start");
            let count: u64 = args[3].parse().expect("count");
            let threads: usize = args.get(4).and_then(|s| s.parse().ok())
                .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
            let next = Arc::new(AtomicU64::new(start));
            let tested = Arc::new(AtomicUsize::new(0));
            let batches = Arc::new(AtomicUsize::new(0));
            let found = Arc::new(AtomicBool::new(false));
            let hist = Arc::new((0..142).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
            let t = Instant::now();
            std::thread::scope(|s| {
                for _ in 0..threads {
                    let (next, tested, batches, found, hist) = (next.clone(), tested.clone(), batches.clone(), found.clone(), hist.clone());
                    let (ops, regs, prefix, tail, curve, fixed) = (&ops, &regs, &prefix, &tail, &curve, &fixed);
                    s.spawn(move || loop {
                        let nonce = next.fetch_add(1, AtOrd::Relaxed);
                        if nonce >= start + count || found.load(AtOrd::Relaxed) && std::env::var("STOP_AT_CLEAN").is_ok() { break; }
                        let o = test_nonce(ops, regs, total_qubits, num_bits, curve, fixed, seed_for_nonce(prefix, tail, nonce), true);
                        tested.fetch_add(1, AtOrd::Relaxed);
                        batches.fetch_add(o.batches_ok + if o.clean { 0 } else { 1 }, AtOrd::Relaxed);
                        hist[o.batches_ok.min(141)].fetch_add(1, AtOrd::Relaxed);
                        if o.clean {
                            found.store(true, AtOrd::Relaxed);
                            println!("CLEAN nonce {nonce} avg_tof={:.3} shots={}", o.avg_tof, o.shots);
                        }
                        let k = tested.load(AtOrd::Relaxed);
                        if k % 50 == 0 {
                            eprintln!("  {k} nonces, {:.2} s/nonce/thread, {:.2} batches/nonce", t.elapsed().as_secs_f64() * threads as f64 / k as f64,
                                      batches.load(AtOrd::Relaxed) as f64 / k as f64);
                        }
                    });
                }
            });
            let k = tested.load(AtOrd::Relaxed).max(1);
            let b = batches.load(AtOrd::Relaxed);
            let secs = t.elapsed().as_secs_f64();
            // per-batch failure probability q (geometric): failures / batches simulated
            let fails = k - hist[141].load(AtOrd::Relaxed);
            let q = fails as f64 / b.max(1) as f64;
            let p_clean = (1.0 - q).powi(141);
            println!("tested {k} nonces in {secs:.0} s on {threads} threads: {:.3} s per nonce per thread, {:.2} batches simulated per nonce",
                     secs * threads as f64 / k as f64, b as f64 / k as f64);
            println!("per-batch failure probability q = {q:.4}; P(clean draw of 141 batches) = {p_clean:.3e}; expected nonces per clean draw = {:.3e}; \
                      expected core-hours per clean draw = {:.3e}", 1.0 / p_clean, secs * threads as f64 / k as f64 / p_clean / 3600.0);
            let h: Vec<usize> = hist.iter().map(|a| a.load(AtOrd::Relaxed)).collect();
            println!("first failing batch histogram (index: count): {}", h.iter().enumerate().filter(|(_, c)| **c > 0)
                     .map(|(i, c)| format!("{i}:{c}")).collect::<Vec<_>>().join(" "));
        }
        "sample" => {
            // full official test (no early stop) on <count> nonces from <start>: per-batch failure rate q, lambda, Toffoli spread
            let start: u64 = args[2].parse().expect("start");
            let count: u64 = args[3].parse().expect("count");
            let threads: usize = args.get(4).and_then(|s| s.parse().ok())
                .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
            let next = AtomicU64::new(start);
            let res = std::sync::Mutex::new(Vec::new());
            let t = Instant::now();
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| loop {
                        let nonce = next.fetch_add(1, AtOrd::Relaxed);
                        if nonce >= start + count { break; }
                        let o = test_nonce(&ops, &regs, total_qubits, num_bits, &curve, &fixed, seed_for_nonce(&prefix, &tail, nonce), false);
                        let failing_batches = 141 - o.batches_total_ok;
                        println!("nonce {nonce} failing_batches {failing_batches} classical {} phase_batches {} avg_tof {:.3}", o.classical, o.phase_batches, o.avg_tof);
                        res.lock().unwrap().push((failing_batches, o.classical, o.phase_batches, o.avg_tof));
                    });
                }
            });
            let r = res.into_inner().unwrap();
            let k = r.len() as f64;
            let fb: f64 = r.iter().map(|x| x.0 as f64).sum::<f64>() / k;
            let q = fb / 141.0;
            let lam = -141.0 * (1.0 - q).ln();
            let mt: f64 = r.iter().map(|x| x.3).sum::<f64>() / k;
            let sd = (r.iter().map(|x| (x.3 - mt).powi(2)).sum::<f64>() / (k - 1.0).max(1.0)).sqrt();
            let cl: f64 = r.iter().map(|x| x.1 as f64).sum::<f64>() / k;
            let ph: f64 = r.iter().map(|x| x.2 as f64).sum::<f64>() / k;
            println!("SAMPLE {} draws in {:.0} s: failing batches/draw {fb:.2} (q {q:.4}, lambda {lam:.2}, +-{:.2}), classical/draw {cl:.2}, phase batches/draw {ph:.2}, \
                      P(clean) {:.3e}, avg Toffoli {mt:.2} (sd between draws {sd:.2}, se {:.2})",
                     r.len(), t.elapsed().as_secs_f64(), lam / (fb * k).sqrt(), (-lam).exp(), sd / k.sqrt());
        }
        "fast" | "compare" => {
            // fast: multi-lane early-stop search. compare: run both the reference (test_nonce) and the fast path on the
            // same nonces and require identical first-failing batches.
            let start: u64 = args[2].parse().expect("start");
            let count: u64 = args[3].parse().expect("count");
            let threads: usize = args.get(4).and_then(|s| s.parse().ok())
                .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
            let t = Instant::now();
            let (cops, n_rng) = compile_ops(&ops);
            let fx = Fixed8::new(&curve, Fixed::new(&curve));
            eprintln!("compiled {} ops ({} rng reads per batch) and the fixed-base table in {:.1} s; {} lanes x 64 shots per pass",
                      cops.len(), n_rng, t.elapsed().as_secs_f64(), W);
            let next = AtomicU64::new(start);
            let (tr, tpts, tsim) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
            let log = std::sync::Mutex::new(Vec::new());
            let t = Instant::now();
            let results: Vec<_> = std::thread::scope(|s| {
                let hs: Vec<_> = (0..threads).map(|_| s.spawn(|| fast_search_thread(&cops, n_rng, total_qubits as usize, num_bits as usize, &regs,
                    &prefix, &tail, &curve, &fx, &next, start + count, &tr, &tpts, &tsim, if mode == "compare" { Some(&log) } else { None }))).collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let secs = t.elapsed().as_secs_f64();
            let k: usize = results.iter().map(|r| r.0).sum();
            let b: usize = results.iter().map(|r| r.1).sum();
            let cleans: usize = results.iter().map(|r| r.2.len()).sum();
            let fails = k - cleans;
            let q = fails as f64 / b.max(1) as f64;
            let p_clean = (1.0 - q).powi(141);
            let per = secs * threads as f64 / k.max(1) as f64;
            println!("FAST tested {k} nonces in {secs:.0} s on {threads} threads: {per:.3} s per nonce per thread ({:.1} ms per batch), {:.2} batches/nonce; \
                      time split rng {:.0}% points {:.0}% sim {:.0}%", per * k as f64 / b as f64 * 1e3, b as f64 / k as f64,
                     100.0 * tr.load(AtOrd::Relaxed) as f64 / (secs * 1e6 * threads as f64), 100.0 * tpts.load(AtOrd::Relaxed) as f64 / (secs * 1e6 * threads as f64),
                     100.0 * tsim.load(AtOrd::Relaxed) as f64 / (secs * 1e6 * threads as f64));
            println!("q = {q:.4}; P(clean) = {p_clean:.3e}; expected nonces per clean draw {:.3e}; expected core-hours per clean draw {:.3e}",
                     1.0 / p_clean, per / p_clean / 3600.0);
            if mode == "compare" {
                let mut l = log.into_inner().unwrap();
                l.sort();
                let mut bad = 0;
                for (nonce, batch, ok) in &l {
                    let o = test_nonce(&ops, &regs, total_qubits, num_bits, &curve, &fixed, seed_for_nonce(&prefix, &tail, *nonce), true);
                    let same = o.clean == *ok && (o.clean || o.batches_ok == *batch);
                    if !same { bad += 1; println!("MISMATCH nonce {nonce}: fast batch {batch} ok {ok}, reference batch {} clean {}", o.batches_ok, o.clean); }
                }
                println!("COMPARE {} nonces, {} mismatches", l.len(), bad);
            }
        }
        "stats" => {
            let mut counts = std::collections::BTreeMap::new();
            let (mut cond_ops, mut max_q, mut max_b) = (0usize, 0u64, 0u64);
            for op in &ops {
                *counts.entry(format!("{:?}", op.kind)).or_insert(0usize) += 1;
                if op.c_condition != NO_BIT { cond_ops += 1; }
                for q in [op.q_target, op.q_control1, op.q_control2] { if q != NO_QUBIT && q.0 > max_q { max_q = q.0; } }
                for b in [op.c_target, op.c_condition] { if b != NO_BIT && b.0 > max_b { max_b = b.0; } }
            }
            println!("ops {} kinds {:?} conditioned {} max_qubit {} max_bit {} num_bits {}", ops.len(), counts, cond_ops, max_q, max_b, num_bits);
        }
        _ => eprintln!("usage: nonce_search check <nonce> | search <start> <count> [threads] | selftest"),
    }
}

// ─── fast multi-lane simulator (same semantics as quantum_ecc::sim::Simulator::apply_iter) ────────────────────
// W independent batches of 64 shots (from different nonces) run through the op stream in one pass. Ops are
// compiled to 12 bytes (ids fit in u16: <=65534 qubits/bits), no-op kinds dropped. Each batch's random bits for
// Hmr/R come from its own Fiat-Shamir stream, squeezed before the pass, in op order, 8 bytes per Hmr/R op.
const W: usize = 8;
const K_CCX: u8 = 0; const K_CX: u8 = 1; const K_SWAP: u8 = 2; const K_X: u8 = 3; const K_CCZ: u8 = 4; const K_CZ: u8 = 5;
const K_Z: u8 = 6; const K_NEG: u8 = 7; const K_HMR: u8 = 8; const K_R: u8 = 9; const K_BINV: u8 = 10; const K_B0: u8 = 11;
const K_B1: u8 = 12; const K_PUSH: u8 = 13; const K_POP: u8 = 14;

#[derive(Clone, Copy)]
struct COp { k: u8, c: u8, qt: u16, q1: u16, q2: u16, cb: u16, cc: u16 }

fn id16(v: u64, none: u64) -> u16 {
    if v == none { return 0xFFFF; }
    assert!(v < 0xFFFF, "id {v} does not fit the compact encoding");
    v as u16
}

fn compile_ops(ops: &[Op]) -> (Vec<COp>, usize) {
    let mut out = Vec::with_capacity(ops.len());
    let mut n_rng = 0;
    for op in ops {
        let k = match op.kind {
            OperationType::CCX => K_CCX, OperationType::CX => K_CX, OperationType::Swap => K_SWAP, OperationType::X => K_X,
            OperationType::CCZ => K_CCZ, OperationType::CZ => K_CZ, OperationType::Z => K_Z, OperationType::Neg => K_NEG,
            OperationType::Hmr => K_HMR, OperationType::R => K_R, OperationType::BitInvert => K_BINV,
            OperationType::BitStore0 => K_B0, OperationType::BitStore1 => K_B1,
            OperationType::PushCondition => K_PUSH, OperationType::PopCondition => K_POP,
            OperationType::AppendToRegister | OperationType::Register | OperationType::DebugPrint => continue,
        };
        if k == K_HMR || k == K_R { n_rng += 1; }
        let c = (op.c_condition != NO_BIT) as u8;
        out.push(COp { k, c, qt: id16(op.q_target.0, NO_QUBIT.0), q1: id16(op.q_control1.0, NO_QUBIT.0),
                       q2: id16(op.q_control2.0, NO_QUBIT.0), cb: id16(op.c_target.0, NO_BIT.0), cc: id16(op.c_condition.0, NO_BIT.0) });
    }
    (out, n_rng)
}

struct Lanes { q: Vec<[u64; W]>, b: Vec<[u64; W]>, phase: [u64; W], rng: Vec<[u64; W]> }

impl Lanes {
    fn new(nq: usize, nb: usize, n_rng: usize) -> Self {
        Lanes { q: vec![[0; W]; nq], b: vec![[0; W]; nb], phase: [0; W], rng: vec![[0; W]; n_rng] }
    }
    fn clear_lane(&mut self, w: usize) {
        for e in &mut self.q { e[w] = 0; }
        for e in &mut self.b { e[w] = 0; }
        self.phase[w] = 0;
    }
    #[inline(never)]
    fn pass(&mut self, cops: &[COp]) {
        let mut stack: Vec<[u64; W]> = Vec::new();
        let mut base = [u64::MAX; W];
        let mut ri = 0usize;
        let q = &mut self.q;
        let b = &mut self.b;
        let phase = &mut self.phase;
        for op in cops {
            let mut cond = base;
            if op.c != 0 { let cb = b[op.cc as usize]; for w in 0..W { cond[w] &= cb[w]; } }
            match op.k {
                K_CX => { let a = q[op.q1 as usize]; let t = &mut q[op.qt as usize]; for w in 0..W { t[w] ^= cond[w] & a[w]; } }
                K_CCX => { let a = q[op.q1 as usize]; let c2 = q[op.q2 as usize]; let t = &mut q[op.qt as usize];
                           for w in 0..W { t[w] ^= cond[w] & a[w] & c2[w]; } }
                K_X => { let t = &mut q[op.qt as usize]; for w in 0..W { t[w] ^= cond[w]; } }
                K_CZ => { let a = q[op.qt as usize]; let c1 = q[op.q1 as usize]; for w in 0..W { phase[w] ^= cond[w] & a[w] & c1[w]; } }
                K_SWAP => {
                    let mut c1 = q[op.q1 as usize]; let mut t = q[op.qt as usize];
                    for w in 0..W { c1[w] ^= t[w]; t[w] ^= cond[w] & c1[w]; c1[w] ^= t[w]; }
                    q[op.q1 as usize] = c1; q[op.qt as usize] = t;
                }
                K_HMR => {
                    let r = self.rng[ri]; ri += 1;
                    let t = q[op.qt as usize];
                    let bb = &mut b[op.cb as usize];
                    for w in 0..W { bb[w] &= !cond[w]; bb[w] ^= r[w] & cond[w]; phase[w] ^= t[w] & r[w] & cond[w]; }
                    let t = &mut q[op.qt as usize]; for w in 0..W { t[w] &= !cond[w]; }
                }
                K_R => {
                    let r = self.rng[ri]; ri += 1;
                    let t = &mut q[op.qt as usize];
                    for w in 0..W { phase[w] ^= t[w] & r[w] & cond[w]; t[w] &= !cond[w]; }
                }
                K_Z => { let a = q[op.qt as usize]; for w in 0..W { phase[w] ^= cond[w] & a[w]; } }
                K_CCZ => { let a = q[op.qt as usize]; let c1 = q[op.q1 as usize]; let c2 = q[op.q2 as usize];
                           for w in 0..W { phase[w] ^= cond[w] & a[w] & c1[w] & c2[w]; } }
                K_NEG => { for w in 0..W { phase[w] ^= cond[w]; } }
                K_BINV => { let bb = &mut b[op.cb as usize]; for w in 0..W { bb[w] ^= cond[w]; } }
                K_B0 => { let bb = &mut b[op.cb as usize]; for w in 0..W { bb[w] &= !cond[w]; } }
                K_B1 => { let bb = &mut b[op.cb as usize]; for w in 0..W { bb[w] |= cond[w]; } }
                K_PUSH => { stack.push(base); let cb = b[op.cc as usize]; for w in 0..W { base[w] &= cb[w]; } }
                K_POP => { if let Some(v) = stack.pop() { base = v; } }
                _ => unreachable!(),
            }
        }
        debug_assert_eq!(ri, self.rng.len());
    }
}

/// Windowed fixed-base table: tab[j][d-1] = d * 2^(8j) * G.
struct Fixed8 { f: Fixed, tab: Vec<Vec<(U256, U256)>> }
impl Fixed8 {
    fn new(curve: &WeierstrassEllipticCurve, f: Fixed) -> Self {
        let mut tab = Vec::with_capacity(32);
        for j in 0..32 {
            let base = f.table[8 * j];
            let mut row = Vec::with_capacity(255);
            let mut e = base;
            row.push(e);
            for _ in 1..255 { e = curve.add(e.0, e.1, base.0, base.1); row.push(e); }
            tab.push(row);
        }
        Fixed8 { f, tab }
    }
    fn mul_g(&self, k: U256) -> (U256, U256) {
        let bytes = k.to_le_bytes::<32>();
        let mut acc = Jac { x: U256::ZERO, y: U256::ZERO, z: U256::ZERO };
        for j in 0..32 {
            let d = bytes[j] as usize;
            if d != 0 { let (tx, ty) = self.tab[j][d - 1]; acc = self.f.add_affine(acc, tx, ty); }
        }
        if acc.z.is_zero() { return (U256::ZERO, U256::ZERO); }
        let zi = acc.z.inv_mod(self.f.p).expect("z invertible");
        let zi2 = self.f.mul(zi, zi);
        (self.f.mul(acc.x, zi2), self.f.mul(acc.y, self.f.mul(zi2, zi)))
    }
}

/// One nonce in a lane: its stream (positioned after the inputs and the batches already run) and its inputs.
struct Slot { nonce: u64, xof: sha3::Shake256Reader, ks: Vec<(U256, U256)>, next_k: usize, batch: usize,
              shots: Vec<((U256, U256), (U256, U256), (U256, U256))> }

fn open_slot(prefix: &Shake256, tail: &[Op], nonce: u64) -> Slot {
    let mut xof = seed_for_nonce(prefix, tail, nonce);
    let mut ks = Vec::with_capacity(NUM_TESTS);
    for _ in 0..NUM_TESTS {
        let mut rb = [[0u8; 32]; 2];
        XofReader::read(&mut xof, &mut rb[0]);
        XofReader::read(&mut xof, &mut rb[1]);
        ks.push((U256::from_le_bytes(rb[0]), U256::from_le_bytes(rb[1])));
    }
    Slot { nonce, xof, ks, next_k: 0, batch: 0, shots: Vec::new() }
}

/// Next batch of up to 64 valid shots (run_tests' skip rules), in order.
fn fill_batch(s: &mut Slot, curve: &WeierstrassEllipticCurve, fx: &Fixed8) {
    s.shots.clear();
    while s.shots.len() < 64 && s.next_k < s.ks.len() {
        let (k1, k2) = s.ks[s.next_k];
        s.next_k += 1;
        let t = fx.mul_g(k1);
        let o = fx.mul_g(k2);
        if t.0 == o.0 { continue; }
        if t.0.is_zero() && t.1.is_zero() { continue; }
        if o.0.is_zero() && o.1.is_zero() { continue; }
        let e = curve.add(t.0, t.1, o.0, o.1);
        s.shots.push((t, o, e));
    }
}

fn set_lane_reg(store: &mut [[u64; W]], ids: &[usize], val: U256, w: usize, shot: usize) {
    for (i, &id) in ids.iter().enumerate() {
        if val.bit(i) { store[id][w] |= 1 << shot; } else { store[id][w] &= !(1 << shot); }
    }
}

fn get_lane_reg(store: &[[u64; W]], ids: &[usize], w: usize, shot: usize) -> U256 {
    let mut v = U256::ZERO;
    for (i, &id) in ids.iter().enumerate() { v.set_bit(i, (store[id][w] >> shot) & 1 != 0); }
    v
}

/// Fast early-stop search on one thread. Returns (nonces finished, batches simulated, clean nonces, first-failing-batch histogram).
fn fast_search_thread(cops: &[COp], n_rng: usize, nq: usize, nb: usize, regs: &[Vec<QubitOrBit>], prefix: &Shake256, tail: &[Op],
                      curve: &WeierstrassEllipticCurve, fx: &Fixed8, next: &AtomicU64, end: u64, t_rng: &AtomicU64,
                      t_pts: &AtomicU64, t_sim: &AtomicU64, log: Option<&std::sync::Mutex<Vec<(u64, usize, bool)>>>)
                      -> (usize, usize, Vec<(u64, f64)>, Vec<usize>) {
    let ids = |r: &Vec<QubitOrBit>| -> Vec<usize> { r.iter().map(|x| match x { QubitOrBit::Qubit(q) => q.0 as usize, QubitOrBit::Bit(b) => b.0 as usize }).collect() };
    let (rx, ry, rox, roy) = (ids(&regs[0]), ids(&regs[1]), ids(&regs[2]), ids(&regs[3]));
    let reg_qubits: Vec<usize> = rx.iter().chain(ry.iter()).copied().collect();
    let mut lanes = Lanes::new(nq, nb, n_rng);
    let mut slots: Vec<Option<Slot>> = (0..W).map(|_| None).collect();
    let (mut finished, mut batches) = (0usize, 0usize);
    let mut clean = Vec::new();
    let mut hist = vec![0usize; 142];
    let mut buf = vec![0u8; n_rng * 8];
    loop {
        let mut any = false;
        let t0 = Instant::now();
        let mut tp = 0f64;
        for w in 0..W {
            if slots[w].is_none() {
                let nonce = next.fetch_add(1, AtOrd::Relaxed);
                if nonce < end { slots[w] = Some(open_slot(prefix, tail, nonce)); }
            }
            lanes.clear_lane(w);
            if let Some(s) = slots[w].as_mut() {
                any = true;
                let tp0 = Instant::now();
                fill_batch(s, curve, fx);
                tp += tp0.elapsed().as_secs_f64();
                XofReader::read(&mut s.xof, &mut buf);
                for (k, ch) in buf.chunks_exact(8).enumerate() { lanes.rng[k][w] = u64::from_le_bytes(ch.try_into().unwrap()); }
                for (shot, (t, o, _)) in s.shots.iter().enumerate() {
                    set_lane_reg(&mut lanes.q, &rx, t.0, w, shot);
                    set_lane_reg(&mut lanes.q, &ry, t.1, w, shot);
                    set_lane_reg(&mut lanes.b, &rox, o.0, w, shot);
                    set_lane_reg(&mut lanes.b, &roy, o.1, w, shot);
                }
            }
        }
        if !any { break; }
        let prep = t0.elapsed().as_secs_f64();
        t_pts.fetch_add((tp * 1e6) as u64, AtOrd::Relaxed);
        t_rng.fetch_add(((prep - tp) * 1e6) as u64, AtOrd::Relaxed);
        let t1 = Instant::now();
        lanes.pass(cops);
        t_sim.fetch_add((t1.elapsed().as_secs_f64() * 1e6) as u64, AtOrd::Relaxed);
        for w in 0..W {
            let Some(s) = slots[w].as_mut() else { continue };
            batches += 1;
            let bs = s.shots.len();
            let mask: u64 = if bs == 64 { u64::MAX } else { (1u64 << bs) - 1 };
            let mut ok = true;
            for (shot, (_, _, e)) in s.shots.iter().enumerate() {
                if get_lane_reg(&lanes.q, &rx, w, shot) != e.0 || get_lane_reg(&lanes.q, &ry, w, shot) != e.1 { ok = false; break; }
            }
            if lanes.phase[w] & mask != 0 { ok = false; }
            if ok {
                for &qi in &reg_qubits { lanes.q[qi][w] = 0; }
                if lanes.q.iter().any(|v| v[w] & mask != 0) { ok = false; }
            }
            let last = bs < 64 || s.next_k >= s.ks.len();
            if !ok || last {
                let passed = s.batch + ok as usize;
                if ok { clean.push((s.nonce, 0.0)); println!("CLEAN nonce {}", s.nonce); }
                hist[passed.min(141)] += 1;
                if let Some(l) = log { l.lock().unwrap().push((s.nonce, s.batch, ok)); }
                finished += 1;
                slots[w] = None;
            } else {
                s.batch += 1;
            }
        }
    }
    (finished, batches, clean, hist)
}
