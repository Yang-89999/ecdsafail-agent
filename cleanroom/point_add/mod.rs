//! Clean-room secp256k1 point adder for the ECDSA.fail harness (agentlab, 2026).
//!
//! Written from published algorithms only; no code from leaderboard submissions or from other
//! authors' implementations. Design: docs/cleanroom.md in the agentlab repository.
//!
//!   ir.rs      virtual-qubit gate IR, exact reversal, lowering with qubit reuse
//!   arith.rs   Gidney temporary-AND adders, comparators, controlled swap, constant adders
//!   modp.rs    modular add / double / halve / negate for an odd prime p
//!   euclid.rs  in-place multiplication and division (Euclidean pass + Bezout reconstruction)
//!   point.rs   the point-addition sequence
//!   tests.rs   unit tests on the harness simulator (`cargo test --release --bin build_circuit`)
//!
//! Registers: 0 = x (qubits), 1 = y (qubits), 2 = offset x (bits), 3 = offset y (bits).

pub mod arith;
pub mod euclid;
pub mod ir;
pub mod modp;
pub mod point;
#[cfg(test)]
mod tests;

use crate::circuit::Op;
use ir::{Lower, Prog};
use modp::Field;

/// Euclidean steps per pass: 2n always reaches (1, 0) (each step at least halves u*v < 2^(2n)).
pub const STEPS: usize = 512;
/// Classical scratch bit for measurement-based uncomputation (after the 512 offset bits).
pub const MBIT: u32 = 512;

pub fn build() -> Vec<Op> {
    let f = Field::secp256k1();
    let n = f.n;
    let mut b = Prog::new();
    let x = b.new_vs(n);
    let y = b.new_vs(n);
    let ox: Vec<u32> = (0..n as u32).collect();
    let oy: Vec<u32> = (n as u32..2 * n as u32).collect();
    let mut low = Lower::new(MBIT);
    for i in 0..n {
        low.bind(x[i], i as u32);
        low.bind(y[i], (n + i) as u32);
    }
    low.declare_qubits(0, &x);
    low.declare_qubits(1, &y);
    low.declare_bits(2, &ox);
    low.declare_bits(3, &oy);
    let (x3, y3) = point::point_add(&mut b, Some(&mut low), &f, &x, &y, &ox, &oy, STEPS);
    low.emit(&b.g);
    let out: Vec<_> = x3.iter().chain(y3.iter()).copied().collect();
    let want: Vec<u32> = (0..2 * n as u32).collect();
    low.route(&out, &want);
    eprintln!("cleanroom: peak width {} qubits, {} ops", low.next_phys, low.ops.len());
    low.ops
}
