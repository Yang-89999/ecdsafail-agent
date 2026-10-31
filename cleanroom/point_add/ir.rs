//! Gate-level IR for the clean-room point adder.
//!
//! Circuits are written over *virtual* qubits (`V`). A subroutine appends gates to a `Prog`; any
//! `Prog` can be reversed exactly (`reversed`), which swaps compute/uncompute pairs:
//!   AndC(a, b -> t)  <->  AndU(a, b -> t)      (temporary logical AND, Gidney 2018)
//!   Alloc(t)         <->  Free(t)
//! and leaves the self-inverse gates (X, CX, CCX, Swap, CZ, Z) as they are. Lowering (`Lower`)
//! maps virtual qubits to physical ids with lowest-free-id reuse (peak width = highest id + 1) and
//! expands AndU into measurement-based uncomputation: HMR into a scratch bit, then a CZ on the two
//! inputs conditioned on that bit. AndU therefore costs no Toffoli.
//!
//! Conditions (`cond`) are classical bit ids, used only for bits that do not change while the
//! circuit runs (the offset registers), so a conditioned gate is its own inverse as well.

use crate::circuit::{BitId, Op, OperationType, QubitId, RegisterId, NO_BIT, NO_QUBIT, NO_REG};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub type V = u32;
pub const NOB: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum K {
    X,
    CX,
    CCX,
    Swap,
    CZ,
    Z,
    AndC,
    AndU,
    Alloc,
    Free,
}

/// One gate. Operand order: X(a); CX(a -> b); CCX(a, b -> c); Swap(a, b); CZ(a, b); Z(a);
/// AndC/AndU(a, b -> c); Alloc(a); Free(a). `cond` is a classical bit id or NOB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gate {
    pub k: K,
    pub a: V,
    pub b: V,
    pub c: V,
    pub cond: u32,
}

#[derive(Default)]
pub struct Prog {
    pub g: Vec<Gate>,
    next_v: V,
}

impl Prog {
    pub fn new() -> Self {
        Prog { g: Vec::new(), next_v: 0 }
    }

    /// A fresh virtual id (not allocated yet: `alloc` it, or use it as a register that already exists).
    pub fn new_v(&mut self) -> V {
        let v = self.next_v;
        self.next_v += 1;
        v
    }

    pub fn new_vs(&mut self, n: usize) -> Vec<V> {
        (0..n).map(|_| self.new_v()).collect()
    }

    #[inline]
    fn push(&mut self, k: K, a: V, b: V, c: V, cond: u32) {
        self.g.push(Gate { k, a, b, c, cond });
    }

    pub fn alloc(&mut self) -> V {
        let v = self.new_v();
        self.push(K::Alloc, v, 0, 0, NOB);
        v
    }
    pub fn alloc_n(&mut self, n: usize) -> Vec<V> {
        (0..n).map(|_| self.alloc()).collect()
    }
    /// Allocate a given (not yet live) virtual id.
    pub fn alloc_id(&mut self, v: V) {
        self.push(K::Alloc, v, 0, 0, NOB);
    }
    /// Release a qubit that is |0> on every shot.
    pub fn free(&mut self, v: V) {
        self.push(K::Free, v, 0, 0, NOB);
    }
    pub fn free_n(&mut self, vs: &[V]) {
        for &v in vs {
            self.free(v);
        }
    }

    pub fn x(&mut self, a: V) {
        self.push(K::X, a, 0, 0, NOB);
    }
    pub fn x_if(&mut self, a: V, bit: u32) {
        self.push(K::X, a, 0, 0, bit);
    }
    pub fn cx(&mut self, ctl: V, tgt: V) {
        debug_assert!(ctl != tgt);
        self.push(K::CX, ctl, tgt, 0, NOB);
    }
    pub fn ccx(&mut self, c1: V, c2: V, tgt: V) {
        debug_assert!(c1 != c2 && c1 != tgt && c2 != tgt);
        self.push(K::CCX, c1, c2, tgt, NOB);
    }
    pub fn swap(&mut self, a: V, b: V) {
        self.push(K::Swap, a, b, 0, NOB);
    }
    pub fn cz(&mut self, a: V, b: V) {
        self.push(K::CZ, a, b, 0, NOB);
    }
    pub fn z(&mut self, a: V) {
        self.push(K::Z, a, 0, 0, NOB);
    }
    /// t = a AND b into a fresh qubit (one Toffoli). Returns t.
    pub fn and_c(&mut self, a: V, b: V) -> V {
        debug_assert!(a != b);
        let t = self.new_v();
        self.push(K::AndC, a, b, t, NOB);
        t
    }
    /// Same, into a given fresh id.
    pub fn and_c_into(&mut self, a: V, b: V, t: V) {
        debug_assert!(a != b);
        self.push(K::AndC, a, b, t, NOB);
    }
    /// Uncompute t = a AND b (t must hold exactly a AND b); t is released. No Toffoli.
    pub fn and_u(&mut self, a: V, b: V, t: V) {
        self.push(K::AndU, a, b, t, NOB);
    }

    /// Append another program's gates.
    pub fn extend(&mut self, other: &[Gate]) {
        self.g.extend_from_slice(other);
    }

    /// Record a position; `take_from` cuts everything appended since.
    pub fn mark(&self) -> usize {
        self.g.len()
    }
    pub fn take_from(&mut self, m: usize) -> Vec<Gate> {
        self.g.split_off(m)
    }

    /// Append the exact inverse of `gates`.
    pub fn extend_reversed(&mut self, gates: &[Gate]) {
        self.g.reserve(gates.len());
        for g in gates.iter().rev() {
            self.g.push(inverse(*g));
        }
    }

    /// Run `f`, then append its inverse: compute; (nothing); uncompute. Useful only for testing.
    pub fn reversed(gates: &[Gate]) -> Vec<Gate> {
        gates.iter().rev().map(|g| inverse(*g)).collect()
    }
}

#[inline]
pub fn inverse(g: Gate) -> Gate {
    let k = match g.k {
        K::AndC => K::AndU,
        K::AndU => K::AndC,
        K::Alloc => K::Free,
        K::Free => K::Alloc,
        k => k,
    };
    Gate { k, ..g }
}

/// Rename virtual ids in place (`from[i]` -> `to[i]`). The `to` ids must not occur in `gates`.
pub fn rename(gates: &mut [Gate], from: &[V], to: &[V]) {
    use std::collections::HashMap;
    let m: HashMap<V, V> = from.iter().copied().zip(to.iter().copied()).collect();
    for g in gates.iter_mut() {
        let r = |v: V| *m.get(&v).unwrap_or(&v);
        match g.k {
            K::X | K::Z | K::Alloc | K::Free => g.a = r(g.a),
            K::CX | K::Swap | K::CZ => {
                g.a = r(g.a);
                g.b = r(g.b);
            }
            K::CCX | K::AndC | K::AndU => {
                g.a = r(g.a);
                g.b = r(g.b);
                g.c = r(g.c);
            }
        }
    }
}

/// Count of Toffolis a gate list executes (AndC and CCX; conditioned ones count as executed).
pub fn toffoli_count(gates: &[Gate]) -> usize {
    gates.iter().filter(|g| matches!(g.k, K::AndC | K::CCX)).count()
}

/// Maps virtual qubits to physical ids and emits harness ops.
pub struct Lower {
    map: Vec<u32>,
    free: BinaryHeap<Reverse<u32>>,
    pub next_phys: u32,
    pub ops: Vec<Op>,
    pub mbit: u32,
    pub live: usize,
    pub peak_live: usize,
}

const UNMAPPED: u32 = u32::MAX;

fn op(kind: OperationType) -> Op {
    Op {
        kind,
        q_control2: NO_QUBIT,
        q_control1: NO_QUBIT,
        q_target: NO_QUBIT,
        c_target: NO_BIT,
        c_condition: NO_BIT,
        r_target: NO_REG,
    }
}

impl Lower {
    /// `mbit` is the classical scratch bit that measurement-based uncomputation writes.
    pub fn new(mbit: u32) -> Self {
        Lower { map: Vec::new(), free: BinaryHeap::new(), next_phys: 0, ops: Vec::new(), mbit, live: 0, peak_live: 0 }
    }

    fn slot(&mut self, v: V) -> &mut u32 {
        let i = v as usize;
        if i >= self.map.len() {
            self.map.resize(i + 1 + i / 2, UNMAPPED);
        }
        &mut self.map[i]
    }

    pub fn phys(&self, v: V) -> u32 {
        let p = *self.map.get(v as usize).unwrap_or(&UNMAPPED);
        assert!(p != UNMAPPED, "virtual qubit {v} used while not allocated");
        p
    }

    /// Bind a virtual qubit to a chosen physical id (for input registers, before anything else).
    pub fn bind(&mut self, v: V, p: u32) {
        *self.slot(v) = p;
        self.next_phys = self.next_phys.max(p + 1);
        self.live += 1;
        self.peak_live = self.peak_live.max(self.live);
    }

    fn alloc(&mut self, v: V) -> u32 {
        let p = match self.free.pop() {
            Some(Reverse(p)) => p,
            None => {
                self.next_phys += 1;
                self.next_phys - 1
            }
        };
        let s = self.slot(v);
        assert!(*s == UNMAPPED, "virtual qubit {v} allocated twice");
        *s = p;
        self.live += 1;
        self.peak_live = self.peak_live.max(self.live);
        p
    }

    fn release(&mut self, v: V) {
        let p = self.phys(v);
        self.map[v as usize] = UNMAPPED;
        self.free.push(Reverse(p));
        self.live -= 1;
    }

    fn q(&self, v: V) -> QubitId {
        QubitId(self.phys(v) as u64)
    }

    pub fn emit(&mut self, gates: &[Gate]) {
        self.ops.reserve(gates.len() + gates.len() / 8);
        for g in gates {
            let cond = if g.cond == NOB { NO_BIT } else { BitId(g.cond as u64) };
            match g.k {
                K::X => {
                    let mut o = op(OperationType::X);
                    o.q_target = self.q(g.a);
                    o.c_condition = cond;
                    self.ops.push(o);
                }
                K::Z => {
                    let mut o = op(OperationType::Z);
                    o.q_target = self.q(g.a);
                    o.c_condition = cond;
                    self.ops.push(o);
                }
                K::CX => {
                    let mut o = op(OperationType::CX);
                    o.q_control1 = self.q(g.a);
                    o.q_target = self.q(g.b);
                    o.c_condition = cond;
                    self.ops.push(o);
                }
                K::CZ => {
                    let mut o = op(OperationType::CZ);
                    o.q_control1 = self.q(g.a);
                    o.q_target = self.q(g.b);
                    o.c_condition = cond;
                    self.ops.push(o);
                }
                K::Swap => {
                    let mut o = op(OperationType::Swap);
                    o.q_control1 = self.q(g.a);
                    o.q_target = self.q(g.b);
                    o.c_condition = cond;
                    self.ops.push(o);
                }
                K::CCX => {
                    let mut o = op(OperationType::CCX);
                    o.q_control1 = self.q(g.a);
                    o.q_control2 = self.q(g.b);
                    o.q_target = self.q(g.c);
                    o.c_condition = cond;
                    self.ops.push(o);
                }
                K::Alloc => {
                    self.alloc(g.a);
                }
                K::Free => self.release(g.a),
                K::AndC => {
                    let t = self.alloc(g.c);
                    let mut o = op(OperationType::CCX);
                    o.q_control1 = self.q(g.a);
                    o.q_control2 = self.q(g.b);
                    o.q_target = QubitId(t as u64);
                    self.ops.push(o);
                }
                K::AndU => {
                    let mut m = op(OperationType::Hmr);
                    m.q_target = self.q(g.c);
                    m.c_target = BitId(self.mbit as u64);
                    self.ops.push(m);
                    let mut o = op(OperationType::CZ);
                    o.q_control1 = self.q(g.a);
                    o.q_target = self.q(g.b);
                    o.c_condition = BitId(self.mbit as u64);
                    self.ops.push(o);
                    self.release(g.c);
                }
            }
        }
    }

    /// Declare a register made of the physical qubits currently holding `vs`.
    pub fn declare_qubits(&mut self, r: u64, vs: &[V]) {
        let mut o = op(OperationType::Register);
        o.r_target = RegisterId(r);
        self.ops.push(o);
        for &v in vs {
            let mut a = op(OperationType::AppendToRegister);
            a.q_target = self.q(v);
            a.r_target = RegisterId(r);
            self.ops.push(a);
        }
    }

    pub fn declare_bits(&mut self, r: u64, bits: &[u32]) {
        let mut o = op(OperationType::Register);
        o.r_target = RegisterId(r);
        self.ops.push(o);
        for &b in bits {
            let mut a = op(OperationType::AppendToRegister);
            a.c_target = BitId(b as u64);
            a.r_target = RegisterId(r);
            self.ops.push(a);
        }
    }

    /// Physical Swaps (Clifford only) that move the value of `now[i]` into physical qubit `want[i]`.
    /// Every other qubit must be |0>, so the permutation is free to move zeros around.
    pub fn route(&mut self, now: &[V], want: &[u32]) {
        use std::collections::HashMap;
        let mut at: HashMap<u32, usize> = HashMap::new(); // physical id -> index of the value it holds
        let mut pos: Vec<u32> = now.iter().map(|&v| self.phys(v)).collect();
        for (i, &p) in pos.iter().enumerate() {
            at.insert(p, i);
        }
        for i in 0..now.len() {
            let (cur, tgt) = (pos[i], want[i]);
            if cur == tgt {
                continue;
            }
            let mut o = op(OperationType::Swap);
            o.q_control1 = QubitId(cur as u64);
            o.q_target = QubitId(tgt as u64);
            self.ops.push(o);
            let other = at.remove(&tgt);
            at.remove(&cur);
            at.insert(tgt, i);
            pos[i] = tgt;
            if let Some(j) = other {
                pos[j] = cur;
                at.insert(cur, j);
            }
        }
    }
}
