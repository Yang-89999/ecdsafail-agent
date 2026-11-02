// gpu_search: TAIL_NONCE search and CRN scoring of ECDSA.fail circuits on an NVIDIA GPU.
//
// The circuit comes as a job file written by `nonce_search export` (tools/nonce_search), so op compilation, register
// ids, the Fiat-Shamir prefix and the fixed-base table are the Rust tool's. Built with nvcc this runs on the GPU;
// built with a plain C++ compiler (scripts/build_gpu_search.sh --cpu) the same per-lane code runs on the CPU, which
// is how it is checked draw by draw against nonce_search (crnfast, compare).
//
//   gpu_search <job> crn <start> <count>       full official test of CRN draws: same lines and JSON as nonce_search crnfast
//   gpu_search <job> search <start> <count>    early-stop nonce search: CLEAN lines, rate, expected cost of a clean draw;
//                                              stops after the round that finds the first clean draw (--no-stop: go on)
//   gpu_search <job> check <nonce>             full official test of one nonce
//   gpu_search <job> selftest                  GPU against this binary's CPU path (CRN draws, early-stop nonces)
//   gpu_search <job> bench [seconds]           search rate at several lane counts
// options: --lanes N (draws in flight), --block N (threads per simulator block), --seconds S (search time limit),
//          --lambda X (error rate for the cost projection, default 22.04),
//          --cpu (use the CPU path in a GPU build), --threads N (CPU threads), --device N (GPU index)
//
// One lane = one draw (a nonce or a CRN salt) holding one batch of 64 shots, bit-sliced: every qubit and bit is a
// 64-bit word, bit s = shot s. Lanes run the same op stream in lockstep (one thread per lane, the op stream staged
// through shared memory), so control flow is uniform and every state access is a coalesced row of lanes.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD static inline
#endif

// ─── op encoding (as nonce_search compile_ops) ───────────────────────────────────────────────────────────────
enum { K_CCX = 0, K_CX, K_SWAP, K_X, K_CCZ, K_CZ, K_Z, K_NEG, K_HMR, K_R, K_BINV, K_B0, K_B1, K_PUSH, K_POP };
struct COp { u32 w0, w1, w2; };  // w0 = k | c << 8 | qt << 16; w1 = q1 | q2 << 16; w2 = cb | cc << 16
// slot program (see compile_slots): the same ops on shared-memory slots, plus LOAD slot <- row, STORE slot -> row
// (w0 = k | slot << 16, w1 = row: qubit q is row q, bit b is row nq + b) and WAIT (w1 = loads allowed in flight)
enum { K_LOAD = 15, K_STORE = 16, K_WAIT = 17 };
static const u32 WAITV[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32};
static const int MAXS = 8;        // condition stack depth (a shift register: no dynamic indexing, so it stays in registers)
static const int NACC = 8;        // per-lane counters: failing, classical, phase_b, ancilla_b, shots, fail_shots, phase_shots, flags
enum { A_FAIL = 0, A_CLASS, A_PHB, A_ANCB, A_SHOTS, A_FSHOTS, A_PSHOTS, A_FLAGS };
static const int RECW = 10;       // record: id, failing|passed, classical, phase_b, ancilla_b, tof, shots, fail_shots, phase_shots, flags
static const u32 KW = 512;        // k words per batch: 64 shots x (k1, k2) x 4 limbs

// ─── Keccak-f[1600] / SHAKE256 ───────────────────────────────────────────────────────────────────────────────
#define KRC_VALUES { \
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL, 0x000000000000808bULL, \
    0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL, 0x0000000000000088ULL, \
    0x0000000080008009ULL, 0x000000008000000aULL, 0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, \
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL, \
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL }
static const u64 KRC_H[24] = KRC_VALUES;
#ifdef __CUDACC__
__constant__ u64 KRC_D[24] = KRC_VALUES;
#endif

HD u64 rotl64(u64 x, int n) { return (x << n) | (x >> (64 - n)); }

HD void keccak_f(u64* a) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
    for (int r = 0; r < 24; r++) {
        u64 c0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20], c1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21],
            c2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22], c3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23],
            c4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
        u64 d0 = c4 ^ rotl64(c1, 1), d1 = c0 ^ rotl64(c2, 1), d2 = c1 ^ rotl64(c3, 1), d3 = c2 ^ rotl64(c4, 1),
            d4 = c3 ^ rotl64(c0, 1);
        a[0] ^= d0; a[5] ^= d0; a[10] ^= d0; a[15] ^= d0; a[20] ^= d0;
        a[1] ^= d1; a[6] ^= d1; a[11] ^= d1; a[16] ^= d1; a[21] ^= d1;
        a[2] ^= d2; a[7] ^= d2; a[12] ^= d2; a[17] ^= d2; a[22] ^= d2;
        a[3] ^= d3; a[8] ^= d3; a[13] ^= d3; a[18] ^= d3; a[23] ^= d3;
        a[4] ^= d4; a[9] ^= d4; a[14] ^= d4; a[19] ^= d4; a[24] ^= d4;
        // rho + pi (generated from the rotation and lane tables of nonce_search keccak_f)
        u64 t = a[1], u;
        u = a[10]; a[10] = rotl64(t, 1); t = u; u = a[7]; a[7] = rotl64(t, 3); t = u; u = a[11]; a[11] = rotl64(t, 6); t = u;
        u = a[17]; a[17] = rotl64(t, 10); t = u; u = a[18]; a[18] = rotl64(t, 15); t = u; u = a[3]; a[3] = rotl64(t, 21); t = u;
        u = a[5]; a[5] = rotl64(t, 28); t = u; u = a[16]; a[16] = rotl64(t, 36); t = u; u = a[8]; a[8] = rotl64(t, 45); t = u;
        u = a[21]; a[21] = rotl64(t, 55); t = u; u = a[24]; a[24] = rotl64(t, 2); t = u; u = a[4]; a[4] = rotl64(t, 14); t = u;
        u = a[15]; a[15] = rotl64(t, 27); t = u; u = a[23]; a[23] = rotl64(t, 41); t = u; u = a[19]; a[19] = rotl64(t, 56); t = u;
        u = a[13]; a[13] = rotl64(t, 8); t = u; u = a[12]; a[12] = rotl64(t, 25); t = u; u = a[2]; a[2] = rotl64(t, 43); t = u;
        u = a[20]; a[20] = rotl64(t, 62); t = u; u = a[14]; a[14] = rotl64(t, 18); t = u; u = a[22]; a[22] = rotl64(t, 39); t = u;
        u = a[9]; a[9] = rotl64(t, 61); t = u; u = a[6]; a[6] = rotl64(t, 20); t = u; u = a[1]; a[1] = rotl64(t, 44); t = u;
        (void)t;
        // chi
#define CHI(j) { u64 b0 = a[j], b1 = a[j + 1], b2 = a[j + 2], b3 = a[j + 3], b4 = a[j + 4]; \
        a[j] = b0 ^ (~b1 & b2); a[j + 1] = b1 ^ (~b2 & b3); a[j + 2] = b2 ^ (~b3 & b4); a[j + 3] = b3 ^ (~b4 & b0); \
        a[j + 4] = b4 ^ (~b0 & b1); }
        CHI(0) CHI(5) CHI(10) CHI(15) CHI(20)
#undef CHI
#ifdef __CUDA_ARCH__
        a[0] ^= KRC_D[r];
#else
        a[0] ^= KRC_H[r];
#endif
    }
}

// state word i (i < 17) with constant indices only, so a register-resident state stays in registers on the GPU
HD u64 pick17(const u64* s, u32 i) {
    switch (i) {
        case 0: return s[0]; case 1: return s[1]; case 2: return s[2]; case 3: return s[3]; case 4: return s[4];
        case 5: return s[5]; case 6: return s[6]; case 7: return s[7]; case 8: return s[8]; case 9: return s[9];
        case 10: return s[10]; case 11: return s[11]; case 12: return s[12]; case 13: return s[13]; case 14: return s[14];
        case 15: return s[15]; default: return s[16];
    }
}

HD void xor_byte(u64* st, u32 pos, u64 b) { st[pos >> 3] ^= b << (8 * (pos & 7)); }

// skip n words of the squeeze (nonce_search squeeze_words semantics: permute when 17 words are used)
HD void sponge_advance(u64* st, u32& used, u64 n) {
    while (n) {
        if (used == 17) { keccak_f(st); used = 0; }
        u64 take = 17 - used;
        if (take > n) take = n;
        used += (u32)take;
        n -= take;
    }
}

// ─── secp256k1 field arithmetic, 4 x 64-bit limbs, canonical residues ─────────────────────────────────────────
struct Fe { u64 v[4]; };
static const u64 P0 = 0xFFFFFFFEFFFFFC2FULL;  // p = 2^256 - PC
static const u64 PC = 0x1000003D1ULL;

HD void mul64(u64 a, u64 b, u64& lo, u64& hi) {
#ifdef __CUDA_ARCH__
    lo = a * b;
    hi = __umul64hi(a, b);
#else
    unsigned __int128 r = (unsigned __int128)a * b;
    lo = (u64)r;
    hi = (u64)(r >> 64);
#endif
}
HD u64 addc(u64 a, u64 b, u64& c) { u64 s = a + c; u64 c1 = s < c; s += b; u64 c2 = s < b; c = c1 | c2; return s; }
HD u64 subb(u64 a, u64 b, u64& br) { u64 d = a - b; u64 b1 = a < b; u64 d2 = d - br; u64 b2 = d < br; br = b1 | b2; return d2; }
HD bool fe_is_zero(const Fe& a) { return !(a.v[0] | a.v[1] | a.v[2] | a.v[3]); }
HD bool fe_eq(const Fe& a, const Fe& b) { return a.v[0] == b.v[0] && a.v[1] == b.v[1] && a.v[2] == b.v[2] && a.v[3] == b.v[3]; }
HD Fe fe_small(u64 x) { Fe r; r.v[0] = x; r.v[1] = r.v[2] = r.v[3] = 0; return r; }
HD void fe_reduce_once(Fe& a) {
    if (a.v[3] == ~0ULL && a.v[2] == ~0ULL && a.v[1] == ~0ULL && a.v[0] >= P0) { a.v[0] -= P0; a.v[1] = a.v[2] = a.v[3] = 0; }
}
HD Fe fe_add(const Fe& a, const Fe& b) {
    Fe r; u64 c = 0;
    for (int i = 0; i < 4; i++) r.v[i] = addc(a.v[i], b.v[i], c);
    if (c) { u64 c2 = 0; r.v[0] = addc(r.v[0], PC, c2); for (int i = 1; i < 4; i++) r.v[i] = addc(r.v[i], 0, c2); }
    fe_reduce_once(r);
    return r;
}
HD Fe fe_sub(const Fe& a, const Fe& b) {
    Fe r; u64 br = 0;
    for (int i = 0; i < 4; i++) r.v[i] = subb(a.v[i], b.v[i], br);
    if (br) { u64 b2 = 0; r.v[0] = subb(r.v[0], PC, b2); for (int i = 1; i < 4; i++) r.v[i] = subb(r.v[i], 0, b2); }
    return r;
}
HD Fe fe_mul(const Fe& a, const Fe& b) {
    u64 t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) {
            u64 lo, hi; mul64(a.v[i], b.v[j], lo, hi);
            u64 c = 0; u64 s = addc(t[i + j], lo, c);
            u64 c2 = 0; s = addc(s, carry, c2);
            t[i + j] = s; carry = hi + c + c2;
        }
        t[i + 4] = carry;
    }
    Fe r; u64 carry = 0;
    for (int i = 0; i < 4; i++) {
        u64 lo, hi; mul64(t[4 + i], PC, lo, hi);
        u64 c = 0; u64 s = addc(t[i], lo, c);
        u64 c2 = 0; s = addc(s, carry, c2);
        r.v[i] = s; carry = hi + c + c2;
    }
    u64 lo, hi; mul64(carry, PC, lo, hi);
    u64 c = 0;
    r.v[0] = addc(r.v[0], lo, c); r.v[1] = addc(r.v[1], hi, c); r.v[2] = addc(r.v[2], 0, c); r.v[3] = addc(r.v[3], 0, c);
    if (c) { c = 0; r.v[0] = addc(r.v[0], PC, c); r.v[1] = addc(r.v[1], 0, c); r.v[2] = addc(r.v[2], 0, c); r.v[3] = addc(r.v[3], 0, c); }
    fe_reduce_once(r);
    return r;
}
HD Fe fe_inv(const Fe& a) {  // a^(p-2)
    const u64 e[4] = {0xFFFFFFFEFFFFFC2DULL, ~0ULL, ~0ULL, ~0ULL};
    Fe r = fe_small(1);
    for (int i = 255; i >= 0; i--) {
        r = fe_mul(r, r);
        if ((e[i >> 6] >> (i & 63)) & 1) r = fe_mul(r, a);
    }
    return r;
}

// Jacobian point; z == 0 is infinity (as nonce_search Fixed)
struct Jac { Fe x, y, z; };
HD void jac_dbl(Jac& a) {
    if (fe_is_zero(a.z) || fe_is_zero(a.y)) { a.x = a.y = a.z = fe_small(0); return; }
    Fe ysq = fe_mul(a.y, a.y);
    Fe s = fe_mul(a.x, ysq); s = fe_add(s, s); s = fe_add(s, s);
    Fe xx = fe_mul(a.x, a.x); Fe m = fe_add(fe_add(xx, xx), xx);
    Fe x3 = fe_sub(fe_mul(m, m), fe_add(s, s));
    Fe y4 = fe_mul(ysq, ysq); y4 = fe_add(y4, y4); y4 = fe_add(y4, y4); y4 = fe_add(y4, y4);
    Fe y3 = fe_sub(fe_mul(m, fe_sub(s, x3)), y4);
    Fe yz = fe_mul(a.y, a.z);
    a.x = x3; a.y = y3; a.z = fe_add(yz, yz);
}
HD void jac_add_affine(Jac& a, const Fe& bx, const Fe& by) {
    if (fe_is_zero(bx) && fe_is_zero(by)) return;
    if (fe_is_zero(a.z)) { a.x = bx; a.y = by; a.z = fe_small(1); return; }
    Fe z2 = fe_mul(a.z, a.z);
    Fe u2 = fe_mul(bx, z2);
    Fe s2 = fe_mul(by, fe_mul(z2, a.z));
    if (fe_eq(u2, a.x)) {
        if (fe_eq(s2, a.y)) { jac_dbl(a); return; }
        a.x = a.y = a.z = fe_small(0); return;
    }
    Fe h = fe_sub(u2, a.x), r = fe_sub(s2, a.y);
    Fe hh = fe_mul(h, h), hhh = fe_mul(hh, h), v = fe_mul(a.x, hh);
    Fe x3 = fe_sub(fe_sub(fe_mul(r, r), hhh), fe_add(v, v));
    Fe y3 = fe_sub(fe_mul(r, fe_sub(v, x3)), fe_mul(a.y, hhh));
    a.z = fe_mul(a.z, h); a.x = x3; a.y = y3;
}
// k*G from the windowed table tab[j][d-1] = d * 2^(8j) * G (nonce_search Fixed8::mul_g); (0,0) = infinity
HD void mul_g(const u64* tab, const u64* k, Fe& x, Fe& y) {
    Jac acc; acc.x = acc.y = acc.z = fe_small(0);
    for (int j = 0; j < 32; j++) {
        u32 d = (u32)(k[j >> 3] >> (8 * (j & 7))) & 0xFF;
        if (d) {
            const u64* e = tab + ((size_t)j * 255 + (d - 1)) * 8;
            Fe tx, ty;
            for (int i = 0; i < 4; i++) { tx.v[i] = e[i]; ty.v[i] = e[4 + i]; }
            jac_add_affine(acc, tx, ty);
        }
    }
    if (fe_is_zero(acc.z)) { x = y = fe_small(0); return; }
    Fe zi = fe_inv(acc.z), zi2 = fe_mul(zi, zi);
    x = fe_mul(acc.x, zi2);
    y = fe_mul(acc.y, fe_mul(zi2, zi));
}

// ─── device-side view of the job and the lanes ───────────────────────────────────────────────────────────────
struct Dev {
    u32 L, nq, nb, ncops, ntests;
    int seed_nonce, early;          // seed from nonce (else CRN salt); stop at the first failing batch
    const COp* ops;
    const COp* prog; u32 nprog, nslot;  // slot program (null: the plain simulator)
    const u32* reg;                 // 4 x 256: x qubits, y qubits, ox bits, oy bits
    const u8* isreg;                // per qubit: in a register (zeroed before the ancilla check)
    const u64* tab;                 // 32 x 255 x (x, y)
    const u64* pst; const u8* pbuf; u32 ppos; const u8* tail;
    u64 *Q, *B, *E, *K;             // Q[q*L+l], B[b*L+l], E[i*L+l] expected (x, y), K[l*512+i] k words
    u64 *ksp, *rsp; u32 *kused, *rused;  // sponges [i*L+l]
    u64* id; u32* batch; u32* state;     // state 0 free, 1 active, 2 no more work
    u32* acc; u64* tofacc;
    u64* next; u64* end;
    u32* outn; u64* out; u32 outcap;
};

HD u64 atomic_add64(u64* p, u64 v) {
#ifdef __CUDA_ARCH__
    return (u64)atomicAdd((unsigned long long*)p, (unsigned long long)v);
#else
    return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
#endif
}
HD u32 atomic_add32(u32* p, u32 v) {
#ifdef __CUDA_ARCH__
    return atomicAdd(p, v);
#else
    return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
#endif
}
HD void atomic_or32(u32* p, u32 v) {
#ifdef __CUDA_ARCH__
    atomicOr(p, v);
#else
    __atomic_fetch_or(p, v, __ATOMIC_RELAXED);
#endif
}
HD u32 popc64(u64 x) {
#ifdef __CUDA_ARCH__
    return (u32)__popcll(x);
#else
    return (u32)__builtin_popcountll(x);
#endif
}

// a free lane takes the next nonce or salt, derives its input stream, and positions its rng stream after the inputs
HD void seed_lane(const Dev& d, u32 l) {
    if (d.state[l] != 0) return;
    u64 n = atomic_add64(d.next, 1);
    if (n >= *d.end) { d.state[l] = 2; return; }
    u64 st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;
    if (!d.seed_nonce) {  // SHAKE256("agentlab-crn-draw" || salt, 8 bytes LE)
        const char* tag = "agentlab-crn-draw";
        u32 pos = 0;
        for (; pos < 17; pos++) xor_byte(st, pos, (u8)tag[pos]);
        for (int i = 0; i < 8; i++, pos++) xor_byte(st, pos, (n >> (8 * i)) & 0xFF);
        xor_byte(st, pos, 0x1F);
        xor_byte(st, 135, 0x80);
        keccak_f(st);
    } else {  // prefix sponge, then the 96 tail ops with q_target = nonce bit (op 2b and 2b+1 carry bit b)
        for (int i = 0; i < 25; i++) st[i] = d.pst[i];
        u32 pos = 0;
        for (; pos < d.ppos; pos++) xor_byte(st, pos, d.pbuf[pos]);
        for (u32 j = 0; j < 96; j++) {
            for (u32 o = 0; o < 49; o++) {
                u64 b = (o >= 17 && o < 25) ? (o == 17 ? (n >> (j / 2)) & 1 : 0) : d.tail[j * 49 + o];
                xor_byte(st, pos, b);
                if (++pos == 136) { keccak_f(st); pos = 0; }
            }
        }
        xor_byte(st, pos, 0x1F);
        xor_byte(st, 135, 0x80);
        keccak_f(st);
    }
    const u32 L = d.L;
    for (int i = 0; i < 25; i++) d.ksp[(size_t)i * L + l] = st[i];
    d.kused[l] = 0;
    u32 used = 0;
    sponge_advance(st, used, (u64)d.ntests * 8);
    for (int i = 0; i < 25; i++) d.rsp[(size_t)i * L + l] = st[i];
    d.rused[l] = used;
    d.id[l] = n;
    d.batch[l] = 0;
    for (int i = 0; i < NACC; i++) d.acc[(size_t)l * NACC + i] = 0;
    d.tofacc[l] = 0;
    d.state[l] = 1;
}

// the k words of this lane's next batch (64 shots)
HD void kwords_lane(const Dev& d, u32 l) {
    if (d.state[l] != 1) return;
    const u32 L = d.L;
    u64 st[25];
    for (int i = 0; i < 25; i++) st[i] = d.ksp[(size_t)i * L + l];
    u32 used = d.kused[l];
    u64* out = d.K + (size_t)l * KW;
    for (u32 i = 0; i < KW; i++) {
        if (used == 17) { keccak_f(st); used = 0; }
        out[i] = st[used++];
    }
    for (int i = 0; i < 25; i++) d.ksp[(size_t)i * L + l] = st[i];
    d.kused[l] = used;
}

// shot s of lane l: inputs t = k1 G, o = k2 G and the expected sum e = t + o. Returns true if run_tests would skip
// this pair (then the batch is not the official one and the lane is flagged).
HD bool points_shot(const Dev& d, u32 l, u32 s, Fe* v) {
    const u64* k = d.K + (size_t)l * KW + s * 8;
    mul_g(d.tab, k, v[0], v[1]);
    mul_g(d.tab, k + 4, v[2], v[3]);
    bool skip = fe_eq(v[0], v[2]) || (fe_is_zero(v[0]) && fe_is_zero(v[1])) || (fe_is_zero(v[2]) && fe_is_zero(v[3]));
    if (skip) { v[4] = v[5] = fe_small(0); return true; }
    Fe lam = fe_mul(fe_sub(v[3], v[1]), fe_inv(fe_sub(v[2], v[0])));
    Fe x3 = fe_sub(fe_sub(fe_mul(lam, lam), v[0]), v[2]);
    v[5] = fe_sub(fe_mul(lam, fe_sub(v[0], x3)), v[1]);
    v[4] = x3;
    return false;
}
// destination word of bit i of value w (0..5: t.x, t.y, o.x, o.y, e.x, e.y) for lane l
HD u64* plane(const Dev& d, u32 l, int w, u32 i) {
    const u32 L = d.L;
    if (w < 2) return d.Q + (size_t)d.reg[w * 256 + i] * L + l;
    if (w < 4) return d.B + (size_t)d.reg[w * 256 + i] * L + l;
    return d.E + (size_t)((w - 4) * 256 + i) * L + l;
}

struct Ctx { u64 phase, tof, base; u32 sp; u64 s0, s1, s2, s3, s4, s5, s6, s7; u64 st[25]; u32 used; };

HD u64 rng_next(Ctx& c) {
    if (c.used == 17) { keccak_f(c.st); c.used = 0; }
    return pick17(c.st, c.used++);
}

// n ops of the stream on one lane (Ql = &Q[lane], Bl = &B[lane], rows L apart)
HD void sim_ops(u64* __restrict__ Ql, u64* __restrict__ Bl, const u32 L, const COp* ops, u32 n, Ctx& c) {
    for (u32 i = 0; i < n; i++) {
        const u32 w0 = ops[i].w0, w1 = ops[i].w1, w2 = ops[i].w2;
        const u32 k = w0 & 0xFF, qt = w0 >> 16, q1 = w1 & 0xFFFF, q2 = w1 >> 16, cb = w2 & 0xFFFF, cc = w2 >> 16;
        u64 cond = c.base;
        if (w0 & 0xFF00) cond &= Bl[cc * L];
        switch (k) {
            case K_CX: Ql[qt * L] ^= cond & Ql[q1 * L]; break;
            case K_CCX: Ql[qt * L] ^= cond & Ql[q1 * L] & Ql[q2 * L]; c.tof += popc64(cond); break;
            case K_X: Ql[qt * L] ^= cond; break;
            case K_CZ: c.phase ^= cond & Ql[qt * L] & Ql[q1 * L]; break;
            case K_SWAP: {
                u64 a = Ql[q1 * L], t = Ql[qt * L];
                a ^= t; t ^= cond & a; a ^= t;
                Ql[q1 * L] = a; Ql[qt * L] = t;
            } break;
            case K_HMR: {
                u64 r = rng_next(c);
                u64 t = Ql[qt * L];
                u64 bb = Bl[cb * L];
                bb &= ~cond; bb ^= r & cond;
                Bl[cb * L] = bb;
                c.phase ^= t & r & cond;
                Ql[qt * L] = t & ~cond;
            } break;
            case K_R: {
                u64 r = rng_next(c);
                u64 t = Ql[qt * L];
                c.phase ^= t & r & cond;
                Ql[qt * L] = t & ~cond;
            } break;
            case K_Z: c.phase ^= cond & Ql[qt * L]; break;
            case K_CCZ: c.phase ^= cond & Ql[qt * L] & Ql[q1 * L] & Ql[q2 * L]; c.tof += popc64(cond); break;
            case K_NEG: c.phase ^= cond; break;
            case K_BINV: Bl[cb * L] ^= cond; break;
            case K_B0: Bl[cb * L] &= ~cond; break;
            case K_B1: Bl[cb * L] |= cond; break;
            case K_PUSH:
                c.s7 = c.s6; c.s6 = c.s5; c.s5 = c.s4; c.s4 = c.s3; c.s3 = c.s2; c.s2 = c.s1; c.s1 = c.s0; c.s0 = c.base;
                c.sp++; c.base &= Bl[cc * L]; break;
            case K_POP:
                if (c.sp > 0) {
                    c.base = c.s0; c.s0 = c.s1; c.s1 = c.s2; c.s2 = c.s3; c.s3 = c.s4; c.s4 = c.s5; c.s5 = c.s6; c.s6 = c.s7;
                    c.sp--;
                } break;
            default: break;
        }
    }
}

// cp.async (sm_80 and newer): a LOAD copies a row word into its shared-memory slot without stalling the thread; the
// host compiler puts a WAIT before the slot's first use, allowing the loads issued after it to stay in flight
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ASYNC_LOADS 1
#endif
HD void slot_load(u64* dst, const u64* src) {
#ifdef ASYNC_LOADS
    u32 sa = (u32)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(sa), "l"(src) : "memory");
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#else
    *dst = *src;
#endif
}
#ifdef ASYNC_LOADS
#define WAITG(n) case n: asm volatile("cp.async.wait_group " #n ";\n" ::: "memory"); break;
#endif
HD void slot_wait(u32 n) {
#ifdef ASYNC_LOADS
    switch (n) {
        WAITG(0) WAITG(1) WAITG(2) WAITG(3) WAITG(4) WAITG(5) WAITG(6) WAITG(7) WAITG(8) WAITG(10) WAITG(12) WAITG(16)
        WAITG(20) WAITG(24) WAITG(32)
        default: asm volatile("cp.async.wait_all;\n" ::: "memory"); break;
    }
#else
    (void)n;
#endif
}

// n entries of a slot program on one lane: S = this lane's slot 0 (slots BS words apart), G = its row 0 (rows L apart)
HD void sim_slots(u64* __restrict__ S, const u32 BS, u64* __restrict__ G, const u32 L, const COp* ops, u32 n, Ctx& c) {
#define SL(x) S[(x) * BS]
    for (u32 i = 0; i < n; i++) {
        const u32 w0 = ops[i].w0, w1 = ops[i].w1, w2 = ops[i].w2;
        const u32 k = w0 & 0xFF, t = w0 >> 16, a = w1 & 0xFFFF, b = w1 >> 16, cb = w2 & 0xFFFF, cc = w2 >> 16;
        u64 cond = c.base;
        if (w0 & 0xFF00) cond &= SL(cc);
        switch (k) {
            case K_CX: SL(t) ^= cond & SL(a); break;
            case K_CCX: SL(t) ^= cond & SL(a) & SL(b); c.tof += popc64(cond); break;
            case K_X: SL(t) ^= cond; break;
            case K_CZ: c.phase ^= cond & SL(t) & SL(a); break;
            case K_SWAP: {
                u64 x = SL(a), y = SL(t);
                x ^= y; y ^= cond & x; x ^= y;
                SL(a) = x; SL(t) = y;
            } break;
            case K_HMR: {
                u64 r = rng_next(c);
                u64 v = SL(t);
                u64 bb = SL(cb);
                bb &= ~cond; bb ^= r & cond;
                SL(cb) = bb;
                c.phase ^= v & r & cond;
                SL(t) = v & ~cond;
            } break;
            case K_R: {
                u64 r = rng_next(c);
                u64 v = SL(t);
                c.phase ^= v & r & cond;
                SL(t) = v & ~cond;
            } break;
            case K_Z: c.phase ^= cond & SL(t); break;
            case K_CCZ: c.phase ^= cond & SL(t) & SL(a) & SL(b); c.tof += popc64(cond); break;
            case K_NEG: c.phase ^= cond; break;
            case K_BINV: SL(cb) ^= cond; break;
            case K_B0: SL(cb) &= ~cond; break;
            case K_B1: SL(cb) |= cond; break;
            case K_PUSH:
                c.s7 = c.s6; c.s6 = c.s5; c.s5 = c.s4; c.s4 = c.s3; c.s3 = c.s2; c.s2 = c.s1; c.s1 = c.s0; c.s0 = c.base;
                c.sp++; c.base &= SL(cc); break;
            case K_POP:
                if (c.sp > 0) {
                    c.base = c.s0; c.s0 = c.s1; c.s1 = c.s2; c.s2 = c.s3; c.s3 = c.s4; c.s4 = c.s5; c.s5 = c.s6; c.s6 = c.s7;
                    c.sp--;
                } break;
            case K_LOAD: slot_load(&SL(t), G + (size_t)w1 * L); break;
            case K_STORE: G[(size_t)w1 * L] = SL(t); break;
            case K_WAIT: slot_wait(w1); break;
            default: break;
        }
    }
#undef SL
}

HD void ctx_load(const Dev& d, u32 l, Ctx& c) {
    c.phase = 0; c.tof = 0; c.base = ~0ULL; c.sp = 0;
    for (int i = 0; i < 25; i++) c.st[i] = d.rsp[(size_t)i * d.L + l];
    c.used = d.rused[l];
}

HD void emit(const Dev& d, u32 l, u64 v1) {
    u32 idx = atomic_add32(d.outn, 1);
    if (idx >= d.outcap) return;
    const u32* a = d.acc + (size_t)l * NACC;
    u64* r = d.out + (size_t)idx * RECW;
    r[0] = d.id[l]; r[1] = v1; r[2] = a[A_CLASS]; r[3] = a[A_PHB]; r[4] = a[A_ANCB]; r[5] = d.tofacc[l];
    r[6] = a[A_SHOTS]; r[7] = a[A_FSHOTS]; r[8] = a[A_PSHOTS]; r[9] = a[A_FLAGS];
}

// end of a batch: the official checks (classical per shot, phase, ancilla after zeroing register qubits), then clear
HD void finish_lane(const Dev& d, u32 l, const Ctx& c) {
    const u32 L = d.L;
    u64* Ql = d.Q + l; u64* Bl = d.B + l; const u64* El = d.E + l;
    d.rused[l] = c.used;
    for (int i = 0; i < 25; i++) d.rsp[(size_t)i * L + l] = c.st[i];
    u64 cl = 0;
    for (u32 i = 0; i < 256; i++) {
        cl |= Ql[(size_t)d.reg[i] * L] ^ El[(size_t)i * L];
        cl |= Ql[(size_t)d.reg[256 + i] * L] ^ El[(size_t)(256 + i) * L];
    }
    u64 ph = c.phase, anc = 0;
    for (u32 q = 0; q < d.nq; q++) { u64 v = Ql[(size_t)q * L]; if (!d.isreg[q]) anc |= v; Ql[(size_t)q * L] = 0; }
    for (u32 b = 0; b < d.nb; b++) Bl[(size_t)b * L] = 0;
    bool ok = !cl && !ph && !anc;
    u32* a = d.acc + (size_t)l * NACC;
    if (c.sp > MAXS) a[A_FLAGS] |= 4;  // condition stack overflow (export checks the depth; never expected)
    a[A_SHOTS] += 64;
    a[A_CLASS] += popc64(cl);
    a[A_FSHOTS] += popc64(cl | ph);
    a[A_PSHOTS] += popc64(ph & ~cl);
    a[A_PHB] += ph != 0;
    a[A_ANCB] += anc != 0;
    a[A_FAIL] += !ok;
    d.tofacc[l] += c.tof;
    u32 b = d.batch[l];
    bool last = b + 1 == d.ntests / 64;
    if (d.early) {
        if (!ok || last) {
            if (ok && last) a[A_FLAGS] |= 1;
            emit(d, l, b + (ok ? 1 : 0));
            d.state[l] = 0;
        } else d.batch[l] = b + 1;
    } else {
        if (last) {
            if (a[A_FAIL] == 0) a[A_FLAGS] |= 1;
            emit(d, l, a[A_FAIL]);
            d.state[l] = 0;
        } else d.batch[l] = b + 1;
    }
}

// ─── GPU kernels ─────────────────────────────────────────────────────────────────────────────────────────────
#ifdef __CUDACC__
static const u32 CH = 512;  // ops staged per chunk

__global__ void k_seed(Dev d) { u32 l = blockIdx.x * blockDim.x + threadIdx.x; if (l < d.L) seed_lane(d, l); }
__global__ void k_kwords(Dev d) { u32 l = blockIdx.x * blockDim.x + threadIdx.x; if (l < d.L) kwords_lane(d, l); }
__global__ void k_points(Dev d) {  // one block of 64 threads per lane
    u32 l = blockIdx.x, s = threadIdx.x;
    if (d.state[l] != 1) return;
    Fe v[6];
    if (points_shot(d, l, s, v)) atomic_or32(d.acc + (size_t)l * NACC + A_FLAGS, 2);
    u32 half = s >> 5;
    for (int w = 0; w < 6; w++) {
        for (u32 i = 0; i < 256; i++) {
            u32 bit = (u32)(v[w].v[i >> 6] >> (i & 63)) & 1;
            u32 m = __ballot_sync(0xFFFFFFFFu, bit);
            if ((s & 31) == 0) reinterpret_cast<u32*>(plane(d, l, w, i))[half] = m;
        }
    }
}
__global__ void k_sim(Dev d) {
    __shared__ COp sh[CH];
    u32 l = blockIdx.x * blockDim.x + threadIdx.x;
    bool act = l < d.L && d.state[l] == 1;
    Ctx c;
    if (act) ctx_load(d, l, c);
    u64* Ql = d.Q + (act ? l : 0);
    u64* Bl = d.B + (act ? l : 0);
    for (u32 b0 = 0; b0 < d.ncops; b0 += CH) {
        u32 n = min(CH, d.ncops - b0);
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) sh[i] = d.ops[b0 + i];
        __syncthreads();
        if (act) sim_ops(Ql, Bl, d.L, sh, n, c);
    }
    if (act) finish_lane(d, l, c);
}
// the slot program: each thread keeps K row words of its lane in shared memory (dynamic: CH staged entries, then
// K x blockDim slots), so most gates never touch global memory and the rest are prefetched with cp.async
__global__ void k_slots(Dev d) {
    extern __shared__ u64 dsm[];
    COp* sh = reinterpret_cast<COp*>(dsm);
    u64* slots = dsm + (CH * sizeof(COp) + 7) / 8;
    const u32 BS = blockDim.x;
    u32 l = blockIdx.x * BS + threadIdx.x;
    bool act = l < d.L && d.state[l] == 1;
    Ctx c;
    if (act) ctx_load(d, l, c);
    u64* S = slots + threadIdx.x;
    u64* G = d.Q + (act ? l : 0);
    for (u32 b0 = 0; b0 < d.nprog; b0 += CH) {
        u32 n = min(CH, d.nprog - b0);
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += BS) sh[i] = d.prog[b0 + i];
        __syncthreads();
        if (act) sim_slots(S, BS, G, d.L, sh, n, c);
    }
    if (act) { slot_wait(0); finish_lane(d, l, c); }
}
static size_t slots_smem(u32 K, u32 block) { return (CH * sizeof(COp) + 7) / 8 * 8 + (size_t)K * block * 8; }

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", \
    cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#endif

// ─── job file ────────────────────────────────────────────────────────────────────────────────────────────────
struct Job {
    u32 nq, nb, ncops, nrng, ntests, maxdepth, nops;
    std::vector<COp> ops;
    std::vector<u32> reg;  // 4 x 256
    std::vector<u8> isreg;
    u64 pst[25]; u8 pbuf[136]; u32 ppos;
    std::vector<u8> tail;
    std::vector<u64> tab;
    u64 tail_nonce;
};

static Job load_job(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    std::vector<u8> b;
    u8 buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
    fclose(f);
    size_t p = 0;
    auto need = [&](size_t k) { if (p + k > b.size()) { fprintf(stderr, "job file truncated\n"); exit(1); } };
    auto rd32 = [&]() { need(4); u32 v; memcpy(&v, &b[p], 4); p += 4; return v; };
    auto rd64 = [&]() { need(8); u64 v; memcpy(&v, &b[p], 8); p += 8; return v; };
    need(8);
    if (memcmp(&b[0], "AGNSJOB1", 8)) { fprintf(stderr, "%s is not a nonce_search job file\n", path); exit(1); }
    p = 8;
    Job j;
    j.nq = rd32(); j.nb = rd32(); j.ncops = rd32(); j.nrng = rd32(); j.ntests = rd32(); j.maxdepth = rd32(); j.nops = rd32();
    j.ops.resize(j.ncops);
    need((size_t)j.ncops * 12);
    memcpy(j.ops.data(), &b[p], (size_t)j.ncops * 12);
    p += (size_t)j.ncops * 12;
    j.reg.resize(1024);
    for (auto& v : j.reg) v = rd32();
    u32 nrq = rd32();
    j.isreg.assign(j.nq, 0);
    for (u32 i = 0; i < nrq; i++) { u32 q = rd32(); if (q < j.nq) j.isreg[q] = 1; }
    for (int i = 0; i < 25; i++) j.pst[i] = rd64();
    need(136); memcpy(j.pbuf, &b[p], 136); p += 136;
    j.ppos = rd32();
    need(96 * 49); j.tail.assign(b.begin() + p, b.begin() + p + 96 * 49); p += 96 * 49;
    size_t nt = (size_t)32 * 255 * 8;
    j.tab.resize(nt);
    for (auto& v : j.tab) v = rd64();
    need(8);
    if (memcmp(&b[p], "AGNSEND1", 8)) { fprintf(stderr, "job file end marker missing\n"); exit(1); }
    j.tail_nonce = 0;
    for (int bit = 0; bit < 48; bit++) j.tail_nonce |= (u64)(j.tail[(2 * bit) * 49 + 17] & 1) << bit;
    if (j.ntests % 64) { fprintf(stderr, "NUM_TESTS %u is not a multiple of 64\n", j.ntests); exit(1); }
    if (j.maxdepth > (u32)MAXS) { fprintf(stderr, "condition depth %u > %d\n", j.maxdepth, MAXS); exit(1); }
    for (int i = 0; i < 1024; i++) if (j.reg[i] >= (i < 512 ? j.nq : j.nb)) { fprintf(stderr, "register id out of range\n"); exit(1); }
    return j;
}

// ─── slot program: the op stream on K cached rows per lane ──────────────────────────────────────────────────────
// Every lane runs the same op stream, so which rows to keep on chip can be decided once, on the host, knowing the
// whole future: Belady's rule (evict the row needed again furthest ahead). A row is loaded into a slot as early as the
// slot is free (just after its previous occupant's last use, and after this row's own last store), so the load has
// time to land; a WAIT goes before the first entry that touches a slot whose load may still be in flight. A dirty row
// is stored back when evicted, and at its last use if it is a qubit (the end-of-batch checks read every qubit); a bit
// row is dead after its last use.
struct SlotProg {
    u32 K = 0;
    double alpha = 0;
    std::vector<COp> prog;
    u64 ops = 0, loads = 0, stores = 0, waits = 0, near = 0, h4 = 0, h8 = 0, h16 = 0, h32 = 0;
    double dist = 0, secs = 0;
};

// eviction score: distance to the next use + alpha x time since the last use (alpha > 0: a little more traffic, but loads
// are issued further ahead of their use)
static SlotProg compile_slots(const Job& j, u32 K, double alpha) {
    double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    const u32 nq = j.nq, NR = j.nq + j.nb, n = j.ncops, NONE = 0xFFFFFFFFu;
    if (K < 4 || K > 4096) { fprintf(stderr, "slots must be 4..4096\n"); exit(1); }
    // rows of each op (unique), read/write flags
    std::vector<u32> ofs(n + 1, 0), rows;
    std::vector<u8> wr;
    rows.reserve((size_t)n * 2 + 16); wr.reserve((size_t)n * 2 + 16);
    for (u32 i = 0; i < n; i++) {
        const COp& o = j.ops[i];
        u32 k = o.w0 & 0xFF, qt = o.w0 >> 16, q1 = o.w1 & 0xFFFF, q2 = o.w1 >> 16, cb = o.w2 & 0xFFFF, cc = o.w2 >> 16;
        size_t b = rows.size();
        auto add = [&](u32 r, bool w) {
            for (size_t t = b; t < rows.size(); t++) if (rows[t] == r) { wr[t] |= w; return; }
            rows.push_back(r); wr.push_back(w);
        };
        if (o.w0 & 0xFF00) add(nq + cc, false);
        switch (k) {
            case K_CX: add(q1, false); add(qt, true); break;
            case K_CCX: add(q1, false); add(q2, false); add(qt, true); break;
            case K_X: case K_R: add(qt, true); break;
            case K_CZ: add(qt, false); add(q1, false); break;
            case K_SWAP: add(q1, true); add(qt, true); break;
            case K_HMR: add(qt, true); add(nq + cb, true); break;
            case K_Z: add(qt, false); break;
            case K_CCZ: add(qt, false); add(q1, false); add(q2, false); break;
            case K_BINV: case K_B0: case K_B1: add(nq + cb, true); break;
            case K_PUSH: add(nq + cc, false); break;
            case K_NEG: case K_POP: break;
            default: fprintf(stderr, "unknown op kind %u\n", k); exit(1);
        }
        ofs[i + 1] = (u32)rows.size();
    }
    // next op that touches the same row (NONE: never again)
    std::vector<u32> nxt(rows.size()), last(NR, NONE);
    for (u32 i = n; i-- > 0;)
        for (u32 t = ofs[i]; t < ofs[i + 1]; t++) { nxt[t] = last[rows[t]]; last[rows[t]] = i; }
    // entries before op i, as linked lists in insertion order
    struct Item { COp e; u32 next; };
    std::vector<Item> items;
    items.reserve((size_t)n);
    std::vector<u32> head(n + 1, NONE), tail(n + 1, NONE);
    auto put = [&](u32 pos, u32 kind, u32 slot, u32 row) {
        u32 id = (u32)items.size();
        items.push_back({{kind | slot << 16, row, 0}, NONE});
        if (head[pos] == NONE) head[pos] = id; else items[tail[pos]].next = id;
        tail[pos] = id;
    };
    std::vector<int> slot_of(NR, -1);
    std::vector<u32> row_in(K, NONE), nextuse(K, NONE), lastacc(K, 0), freesince(K, 0), storepos(NR, 0), loadpos(K, 0);
    std::vector<u8> dirty(K, 0);
    std::vector<u16> aslot(rows.size());  // the slot each access uses
    SlotProg sp;
    sp.K = K; sp.alpha = alpha; sp.ops = n;
    std::vector<u32> slot_here(8);
    for (u32 i = 0; i < n; i++) {
        u32 nr = ofs[i + 1] - ofs[i];
        for (u32 t = ofs[i]; t < ofs[i + 1]; t++) {
            u32 r = rows[t];
            if (slot_of[r] < 0) {
                int s = -1;
                for (u32 x = 0; x < K; x++)  // a free slot, free the longest
                    if (row_in[x] == NONE && (s < 0 || freesince[x] < freesince[s])) s = (int)x;
                if (s < 0) {  // evict the row needed again furthest ahead (not one of this op's rows)
                    double best = -1;
                    for (u32 x = 0; x < K; x++) {
                        bool mine = false;
                        for (u32 u = ofs[i]; u < ofs[i + 1]; u++) if (rows[u] == row_in[x]) mine = true;
                        if (mine) continue;
                        double sc = (nextuse[x] == NONE ? 4e9 : (double)nextuse[x] - i) + alpha * (double)(i - lastacc[x]) + (dirty[x] ? 0 : 0.5);
                        if (sc > best) { best = sc; s = (int)x; }
                    }
                    u32 v = row_in[s];
                    if (dirty[s] && (v < nq || nextuse[s] != NONE)) { put(lastacc[s] + 1, K_STORE, s, v); storepos[v] = lastacc[s] + 1; sp.stores++; }
                    slot_of[v] = -1; row_in[s] = NONE; freesince[s] = lastacc[s] + 1;
                }
                u32 pos = std::max(freesince[s], storepos[r]);
                put(pos, K_LOAD, s, r);
                sp.loads++;
                sp.dist += i - pos;
                if (i - pos < 16) sp.near++;
                if (i - pos < 4) sp.h4++; else if (i - pos < 8) sp.h8++; else if (i - pos < 16) sp.h16++; else if (i - pos < 32) sp.h32++;
                row_in[s] = r; slot_of[r] = s; dirty[s] = 0;
            }
            u32 s = slot_of[r];
            aslot[t] = (u16)s;
            if (wr[t]) dirty[s] = 1;
            nextuse[s] = nxt[t];
            lastacc[s] = i;
        }
        // rows not needed again: a dirty qubit goes back for the end checks; the slot is free from the next op on
        for (u32 t = ofs[i]; t < ofs[i + 1]; t++) {
            if (nxt[t] != NONE) continue;
            u32 r = rows[t], s = slot_of[r];
            if (dirty[s] && r < nq) { put(i + 1, K_STORE, s, r); storepos[r] = i + 1; sp.stores++; }
            slot_of[r] = -1; row_in[s] = NONE; freesince[s] = i + 1; dirty[s] = 0;
        }
        (void)nr;
    }
    // final order with WAITs: an entry that touches a slot whose load may be in flight waits until at most w loads
    // (the ones issued after that load, rounded down to a supported count) are still pending
    std::vector<u32> pend(K, NONE);  // sequence number of the slot's last load
    u64 nl = 0, known = 0;           // loads issued; loads with sequence < known have landed
    sp.prog.reserve((size_t)n + sp.loads + sp.stores + sp.loads);
    auto touch = [&](std::initializer_list<u32> sl) {
        u64 m = 0; bool need = false;
        for (u32 x : sl) if (pend[x] != NONE && pend[x] >= known) { if (!need || pend[x] > m) m = pend[x]; need = true; }
        if (!need) return;
        u64 after = nl - 1 - m;
        u32 w = 0;
        for (u32 v : WAITV) if (v <= after) w = v;
        sp.prog.push_back({K_WAIT, w, 0});
        sp.waits++;
        known = nl - w;
    };
    auto emit_items = [&](u32 pos) {
        for (u32 id = head[pos]; id != NONE; id = items[id].next) {
            COp e = items[id].e;
            u32 k = e.w0 & 0xFF, s = e.w0 >> 16;
            touch({s});
            sp.prog.push_back(e);
            if (k == K_LOAD) pend[s] = (u32)nl++;
        }
    };
    for (u32 i = 0; i < n; i++) {
        emit_items(i);
        const COp& o = j.ops[i];
        u32 k = o.w0 & 0xFF, qt = o.w0 >> 16, q1 = o.w1 & 0xFFFF, q2 = o.w1 >> 16, cb = o.w2 & 0xFFFF, cc = o.w2 >> 16;
        bool cnd = o.w0 & 0xFF00;
        auto sl = [&](u32 r) -> u32 {  // the slot of row r in this op
            for (u32 t = ofs[i]; t < ofs[i + 1]; t++) if (rows[t] == r) return aslot[t];
            return 0;  // field not used by this kind
        };
        u32 sqt = sl(qt), sq1 = sl(q1), sq2 = sl(q2), scb = sl(nq + cb), scc = (cnd || k == K_PUSH) ? sl(nq + cc) : 0;
        // only the fields this kind uses name rows; the rest stay 0 (sl may have matched an unrelated row)
        switch (k) {
            case K_CX: case K_CZ: sq2 = 0; scb = 0; break;
            case K_CCX: case K_CCZ: scb = 0; break;
            case K_SWAP: sq2 = 0; scb = 0; break;
            case K_X: case K_R: case K_Z: sq1 = sq2 = scb = 0; break;
            case K_HMR: sq1 = sq2 = 0; break;
            case K_BINV: case K_B0: case K_B1: sqt = sq1 = sq2 = 0; break;
            case K_PUSH: case K_NEG: case K_POP: sqt = sq1 = sq2 = scb = 0; break;
        }
        std::vector<u32> used;
        if (cnd || k == K_PUSH) used.push_back(scc);
        for (u32 t = ofs[i]; t < ofs[i + 1]; t++) used.push_back(aslot[t]);
        {
            u64 m = 0; bool need = false;
            for (u32 x : used) if (pend[x] != NONE && pend[x] >= known) { if (!need || pend[x] > m) m = pend[x]; need = true; }
            if (need) {
                u64 after = nl - 1 - m;
                u32 w = 0;
                for (u32 v : WAITV) if (v <= after) w = v;
                sp.prog.push_back({K_WAIT, w, 0});
                sp.waits++;
                known = nl - w;
            }
        }
        sp.prog.push_back({k | (o.w0 & 0xFF00) | sqt << 16, sq1 | sq2 << 16, scb | scc << 16});
    }
    emit_items(n);
    sp.prog.push_back({K_WAIT, 0, 0});
    sp.dist /= std::max<u64>(sp.loads, 1);
    sp.secs = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
    if (sp.prog.size() >= (1ULL << 32)) { fprintf(stderr, "slot program too long\n"); exit(1); }
    return sp;
}

// ─── runner: lanes on the GPU or the CPU ─────────────────────────────────────────────────────────────────────
struct Rec { u64 v[RECW]; };
struct Timing { double seed = 0, kw = 0, pts = 0, sim = 0, io = 0; };

struct Runner {
    const Job& j;
    bool gpu;
    u32 L, block;
    Dev d{};
    std::vector<std::vector<u64>> h64;  // CPU storage
    std::vector<std::vector<u32>> h32;
    std::vector<std::vector<u8>> h8;
    std::vector<COp> hops;
    std::vector<void*> dptrs;
    Timing tm;
    const SlotProg* sp;

    Runner(const Job& job, bool use_gpu, u32 lanes, u32 blk, const SlotProg* slots = nullptr)
        : j(job), gpu(use_gpu), L(lanes), block(blk), sp(slots) {
        size_t rows = ((size_t)j.nq + j.nb) * L;
        if (rows >= (1ULL << 32)) { fprintf(stderr, "too many lanes for 32-bit row offsets\n"); exit(1); }
        d.L = L; d.nq = j.nq; d.nb = j.nb; d.ncops = j.ncops; d.ntests = j.ntests; d.ppos = j.ppos; d.outcap = L;
        d.ops = (const COp*)put(j.ops.data(), j.ops.size() * sizeof(COp));
        d.reg = (const u32*)put(j.reg.data(), j.reg.size() * 4);
        d.isreg = (const u8*)put(j.isreg.data(), j.isreg.size());
        d.tab = (const u64*)put(j.tab.data(), j.tab.size() * 8);
        d.pst = (const u64*)put(j.pst, sizeof j.pst);
        d.pbuf = (const u8*)put(j.pbuf, sizeof j.pbuf);
        d.tail = (const u8*)put(j.tail.data(), j.tail.size());
        d.Q = (u64*)zeros(((size_t)j.nq + j.nb) * L * 8);  // qubit rows, then bit rows (the slot program's row ids)
        d.B = d.Q + (size_t)j.nq * L;
        if (sp) {
            d.prog = (const COp*)put(sp->prog.data(), sp->prog.size() * sizeof(COp));
            d.nprog = (u32)sp->prog.size();
            d.nslot = sp->K;
#ifdef __CUDACC__
            if (gpu) CK(cudaFuncSetAttribute(k_slots, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)slots_smem(sp->K, block)));
#endif
        }
        d.E = (u64*)zeros((size_t)512 * L * 8);
        d.K = (u64*)zeros((size_t)KW * L * 8);
        d.ksp = (u64*)zeros((size_t)25 * L * 8);
        d.rsp = (u64*)zeros((size_t)25 * L * 8);
        d.kused = (u32*)zeros((size_t)L * 4);
        d.rused = (u32*)zeros((size_t)L * 4);
        d.id = (u64*)zeros((size_t)L * 8);
        d.batch = (u32*)zeros((size_t)L * 4);
        d.state = (u32*)zeros((size_t)L * 4);
        d.acc = (u32*)zeros((size_t)L * NACC * 4);
        d.tofacc = (u64*)zeros((size_t)L * 8);
        d.next = (u64*)zeros(8);
        d.end = (u64*)zeros(8);
        d.outn = (u32*)zeros(4);
        d.out = (u64*)zeros((size_t)L * RECW * 8);
    }
    ~Runner() {
#ifdef __CUDACC__
        for (void* p : dptrs) cudaFree(p);
#endif
    }
    static size_t bytes_per_lane(const Job& j) { return ((size_t)j.nq + j.nb + 512 + KW + 50 + 3 + RECW) * 8 + (4 + NACC) * 4; }

    void* put(const void* src, size_t n) {
#ifdef __CUDACC__
        if (gpu) { void* p; CK(cudaMalloc(&p, std::max<size_t>(n, 8))); CK(cudaMemcpy(p, src, n, cudaMemcpyHostToDevice)); dptrs.push_back(p); return p; }
#endif
        h8.emplace_back((const u8*)src, (const u8*)src + n);
        return h8.back().data();
    }
    void* zeros(size_t n) {
#ifdef __CUDACC__
        if (gpu) { void* p; CK(cudaMalloc(&p, std::max<size_t>(n, 8))); CK(cudaMemset(p, 0, n)); dptrs.push_back(p); return p; }
#endif
        h64.emplace_back((n + 7) / 8 + 1, 0);
        return h64.back().data();
    }
    void set64(u64* p, u64 v) {
#ifdef __CUDACC__
        if (gpu) { CK(cudaMemcpy(p, &v, 8, cudaMemcpyHostToDevice)); return; }
#endif
        *p = v;
    }
    template <class T> void get(T* dst, const T* src, size_t n) {
#ifdef __CUDACC__
        if (gpu) { CK(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost)); return; }
#endif
        memcpy(dst, src, n * sizeof(T));
    }
    void reset_lanes() {
#ifdef __CUDACC__
        if (gpu) { CK(cudaMemset(d.state, 0, (size_t)L * 4)); CK(cudaMemset(d.Q, 0, (size_t)j.nq * L * 8));
                   CK(cudaMemset(d.B, 0, (size_t)j.nb * L * 8)); CK(cudaMemset(d.outn, 0, 4)); return; }
#endif
        memset(d.state, 0, (size_t)L * 4); memset(d.Q, 0, (size_t)j.nq * L * 8); memset(d.B, 0, (size_t)j.nb * L * 8); *d.outn = 0;
    }

    static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

    // one round: every active lane runs one batch; returns the records of lanes that finished
    void round(std::vector<Rec>& recs) {
        double t0 = now();
#ifdef __CUDACC__
        if (gpu) {
            u32 nb = (L + 255) / 256;
            k_seed<<<nb, 256>>>(d); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            double t1 = now(); tm.seed += t1 - t0;
            k_kwords<<<nb, 256>>>(d); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            double t2 = now(); tm.kw += t2 - t1;
            k_points<<<L, 64>>>(d); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            double t3 = now(); tm.pts += t3 - t2;
            if (sp) k_slots<<<(L + block - 1) / block, block, slots_smem(sp->K, block)>>>(d);
            else k_sim<<<(L + block - 1) / block, block>>>(d);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            double t4 = now(); tm.sim += t4 - t3;
            u32 n; CK(cudaMemcpy(&n, d.outn, 4, cudaMemcpyDeviceToHost));
            n = std::min(n, d.outcap);
            size_t base = recs.size(); recs.resize(base + n);
            if (n) CK(cudaMemcpy(&recs[base], d.out, (size_t)n * RECW * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemset(d.outn, 0, 4));
            tm.io += now() - t4;
            return;
        }
#endif
        const Dev dd = d;
        for (u32 l = 0; l < L; l++) seed_lane(dd, l);  // serial: lanes take nonces in order
        double t1 = now(); tm.seed += t1 - t0;
#pragma omp parallel for schedule(dynamic, 1)
        for (u32 l = 0; l < L; l++) kwords_lane(dd, l);
        double t2 = now(); tm.kw += t2 - t1;
#pragma omp parallel for schedule(dynamic, 1)
        for (u32 l = 0; l < L; l++) {
            if (dd.state[l] != 1) continue;
            Fe v[6][64];
            for (u32 s = 0; s < 64; s++) {
                Fe t[6];
                if (points_shot(dd, l, s, t)) dd.acc[(size_t)l * NACC + A_FLAGS] |= 2;
                for (int w = 0; w < 6; w++) v[w][s] = t[w];
            }
            for (int w = 0; w < 6; w++)
                for (u32 i = 0; i < 256; i++) {
                    u64 m = 0;
                    for (u32 s = 0; s < 64; s++) m |= ((v[w][s].v[i >> 6] >> (i & 63)) & 1) << s;
                    *plane(dd, l, w, i) = m;
                }
        }
        double t3 = now(); tm.pts += t3 - t2;
#pragma omp parallel for schedule(dynamic, 1)
        for (u32 l = 0; l < L; l++) {
            if (dd.state[l] != 1) continue;
            Ctx c;
            ctx_load(dd, l, c);
            if (sp) {
                std::vector<u64> S(sp->K, 0);
                sim_slots(S.data(), 1, dd.Q + l, L, dd.prog, dd.nprog, c);
            } else sim_ops(dd.Q + l, dd.B + l, L, dd.ops, dd.ncops, c);
            finish_lane(dd, l, c);
        }
        double t4 = now(); tm.sim += t4 - t3;
        u32 n = std::min(*d.outn, d.outcap);
        recs.insert(recs.end(), (const Rec*)d.out, (const Rec*)d.out + n);
        *d.outn = 0;
        tm.io += now() - t4;
    }

    // runs draws start..start+count; on_round(records so far, elapsed) after every round; stops early when it returns false
    std::vector<Rec> run(bool seed_nonce, bool early, u64 start, u64 count,
                         const std::function<bool(const std::vector<Rec>&, double)>& on_round) {
        d.seed_nonce = seed_nonce; d.early = early;
        reset_lanes();
        set64(d.next, start);
        set64(d.end, start + count);
        std::vector<Rec> recs;
        double t0 = now();
        while (recs.size() < count) {
            round(recs);
            if (!on_round(recs, now() - t0)) break;
        }
        return recs;
    }
};

// ─── modes ───────────────────────────────────────────────────────────────────────────────────────────────────
static double lam(double fb) { fb = std::min(std::max(fb, 0.0), 141 - 0.05); return -141.0 * std::log(1 - fb / 141.0); }

static void print_crn(const Job& j, std::vector<Rec> r, double secs, const char* where) {
    std::sort(r.begin(), r.end(), [](const Rec& a, const Rec& b) { return a.v[0] < b.v[0]; });
    double fb = 0, mt = 0, cl = 0, fs = 0, po = 0;
    for (auto& d : r) {
        printf("crn %llu failing_batches %llu classical %llu phase_batches %llu avg_tof %.3f\n", (unsigned long long)d.v[0],
               (unsigned long long)d.v[1], (unsigned long long)d.v[2], (unsigned long long)d.v[3], (double)d.v[5] / std::max<u64>(d.v[6], 1));
        fb += d.v[1]; mt += (double)d.v[5] / std::max<u64>(d.v[6], 1); cl += d.v[2]; fs += d.v[7]; po += d.v[8];
        if (d.v[9] & 6) printf("!! draw %llu: %s\n", (unsigned long long)d.v[0], (d.v[9] & 2) ? "a skipped input pair (batch differs from the official one); rerun this draw with nonce_search" : "condition stack overflow");
    }
    double k = std::max<size_t>(r.size(), 1);
    fb /= k; mt /= k; cl /= k; fs /= k; po /= k;
    printf("CRN %zu draws: failing batches/draw %.3f, lambda %.3f, avg Toffoli %.3f (%.1f s on %s); per draw: classical failures %.3f, "
           "failing shots %.3f, phase-only shots %.3f\n", r.size(), fb, lam(fb), mt, secs, where, cl, fs, po);
    printf("{\"draws\": %zu, \"qubits\": %u, \"failing_batches_per_draw\": %.4f, \"avg_tof\": %.4f, \"seconds\": %.1f, \"per_draw\": [",
           r.size(), j.nq, fb, mt, secs);
    for (size_t i = 0; i < r.size(); i++) {
        const u64* v = r[i].v;
        printf("%s[%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu]", i ? "," : "", (unsigned long long)v[0], (unsigned long long)v[1],
               (unsigned long long)v[2], (unsigned long long)v[3], (unsigned long long)v[5], (unsigned long long)v[6],
               (unsigned long long)v[7], (unsigned long long)v[8]);
    }
    printf("]}\n");
    fflush(stdout);
}

// lane_batches counts every batch run, including those of nonces still in flight when the run stops; counting only
// finished nonces' batches leaves out the long-running ones and overstates the per-batch failure rate q
struct SearchStats { u64 finished = 0, batches = 0, lane_batches = 0, clean = 0, anomalies = 0; std::vector<u64> hist = std::vector<u64>(142, 0); };
static void add_search(SearchStats& s, const Rec& r) {
    s.finished++;
    u64 passed = r.v[1];
    bool clean = r.v[9] & 1;
    s.batches += passed + (clean ? 0 : 1);
    s.hist[std::min<u64>(passed, 141)]++;
    if (clean) s.clean++;
    if (r.v[9] & 6) s.anomalies++;
}
static void print_search(const SearchStats& s, double secs, const char* where) {
    u64 nb = std::max(s.lane_batches, s.batches);
    double q = s.finished > s.clean ? (double)(s.finished - s.clean) / std::max<u64>(nb, 1) : 0;
    double pc = std::pow(1 - q, 141);
    double rate = s.finished / std::max(secs, 1e-9);
    printf("SEARCH %llu nonces in %.0f s on %s: %.1f nonces/s, %.1f batches/s, %.2f batches/nonce; q = %.4f; P(clean) = %.3e; "
           "expected nonces per clean draw %.3e; expected hours per clean draw on this device %.2f; clean found %llu%s\n",
           (unsigned long long)s.finished, secs, where, rate, nb / std::max(secs, 1e-9), (double)nb / std::max<u64>(s.finished, 1),
           q, pc, 1 / pc, 1 / pc / rate / 3600, (unsigned long long)s.clean, s.anomalies ? " (some nonces had skipped input pairs: recheck them)" : "");
    fflush(stdout);
}

static bool same_rec(const Rec& a, const Rec& b, bool early) {
    if (early) return a.v[0] == b.v[0] && a.v[1] == b.v[1] && (a.v[9] & 1) == (b.v[9] & 1);
    for (int i = 0; i < RECW; i++) if (a.v[i] != b.v[i]) return false;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: gpu_search <job.bin> crn|search|check|selftest|bench ... [--lanes N] [--block N] [--seconds S] [--cpu] [--threads N]\n");
        return 1;
    }
    std::vector<std::string> pos;
    u32 lanes = 0, block = 128;
    bool lanes_given = false, kernel_given = false, specs_given = false;
    double seconds = 0;
    bool force_cpu = false, stop_on_clean = true;
    double proj_lambda = 22.04;  // our base and the public head measure ~22.0 on the audit CRN draws
    int device = 0;
    std::string kernel;          // "slots" (default on GPUs with cp.async, sm_80+) or "plain"
    struct Spec { u32 K; double alpha; u32 block; };
    std::vector<Spec> specs;  // --slots K[:alpha[:block]],...
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--lanes" && i + 1 < argc) { lanes = (u32)atol(argv[++i]); lanes_given = true; }
        else if (a == "--block" && i + 1 < argc) block = (u32)atol(argv[++i]);
        else if (a == "--seconds" && i + 1 < argc) seconds = atof(argv[++i]);
        else if (a == "--cpu") force_cpu = true;
        else if (a == "--no-stop") stop_on_clean = false;
        else if (a == "--lambda" && i + 1 < argc) proj_lambda = atof(argv[++i]);
        else if (a == "--device" && i + 1 < argc) device = atoi(argv[++i]);
        else if (a == "--kernel" && i + 1 < argc) { kernel = argv[++i]; kernel_given = true; }
        else if (a == "--slots" && i + 1 < argc) {
            std::string v = argv[++i];
            specs_given = true;
            for (size_t p0 = 0; p0 < v.size();) {
                size_t p1 = v.find(',', p0); if (p1 == std::string::npos) p1 = v.size();
                std::string e = v.substr(p0, p1 - p0);
                Spec sp{64, 2.0, 0};
                sp.K = (u32)atol(e.c_str());
                size_t c1 = e.find(':');
                if (c1 != std::string::npos) {
                    sp.alpha = atof(e.c_str() + c1 + 1);
                    size_t c2 = e.find(':', c1 + 1);
                    if (c2 != std::string::npos) sp.block = (u32)atol(e.c_str() + c2 + 1);
                }
                specs.push_back(sp); p0 = p1 + 1;
            }
        }
        else if (a == "--threads" && i + 1 < argc) {
            int t = atoi(argv[++i]);
#ifdef _OPENMP
            omp_set_num_threads(t);
#else
            (void)t;
#endif
        } else pos.push_back(a);
    }
    Job job = load_job(pos[0].c_str());
    std::string mode = pos[1];
    bool gpu = false;
    std::string where = "CPU";
    u32 maxl = 1u << 20;
    int cc_major = 0, nsm = 1;
#ifdef __CUDACC__
    int ndev = 0;
    if (!force_cpu && cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > device) {
        CK(cudaSetDevice(device));
        gpu = true;
        cudaDeviceProp prop;
        CK(cudaGetDeviceProperties(&prop, device));
        where = prop.name;
        cc_major = prop.major; nsm = prop.multiProcessorCount;
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        maxl = (u32)std::min<size_t>(fr * 0.8 / Runner::bytes_per_lane(job), 1u << 20);
        if (!lanes) lanes = std::min<u32>(maxl, (u32)prop.multiProcessorCount * 512);
        if (lanes > maxl) lanes = maxl;
        lanes = std::max<u32>(block, lanes / block * block);
        fprintf(stderr, "GPU %s: %d SMs, %.1f GB free; %u lanes (%.2f GB), %u threads per simulator block\n", prop.name,
                prop.multiProcessorCount, fr / 1e9, lanes, lanes * (double)Runner::bytes_per_lane(job) / 1e9, block);
    }
#endif
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    if (!gpu) {
        if (!lanes) lanes = std::max(8, threads * 2);
        char w[64]; snprintf(w, sizeof w, "CPU (%d threads)", threads); where = w;
    }
    fprintf(stderr, "job: %u ops (%u compiled, %u rng reads per batch), %u qubits, %u bits, tail nonce %llu\n", job.nops, job.ncops,
            job.nrng, job.nq, job.nb, (unsigned long long)job.tail_nonce);
    if (kernel.empty()) kernel = (gpu && cc_major >= 8) ? "slots" : "plain";
    if (kernel != "slots" && kernel != "plain") { fprintf(stderr, "--kernel is slots or plain\n"); return 1; }
    std::vector<SlotProg> progs;  // compiled once per slot count
    auto prog_for = [&](Spec sc) -> const SlotProg* {
        for (auto& p : progs) if (p.K == sc.K && p.alpha == sc.alpha) return &p;
        if (progs.size() == progs.capacity()) { fprintf(stderr, "too many slot programs\n"); exit(1); }
        u32 K = sc.K;
        progs.push_back(compile_slots(job, K, sc.alpha));
        const SlotProg& p = progs.back();
        fprintf(stderr, "slot program K=%u alpha=%g: %zu entries (%llu ops, %llu loads, %llu stores, %llu waits) in %.1f s; global words per op %.3f "
                "(plain simulator ~3); loads issued %.0f ops ahead on average (under 4: %.1f%%, 4-7: %.1f%%, 8-15: %.1f%%, 16-31: %.1f%%)\n",
                K, p.alpha, p.prog.size(), (unsigned long long)p.ops, (unsigned long long)p.loads, (unsigned long long)p.stores,
                (unsigned long long)p.waits, p.secs, (double)(p.loads + p.stores) / std::max<u64>(p.ops, 1), p.dist,
                100.0 * p.h4 / std::max<u64>(p.loads, 1), 100.0 * p.h8 / std::max<u64>(p.loads, 1), 100.0 * p.h16 / std::max<u64>(p.loads, 1),
                100.0 * p.h32 / std::max<u64>(p.loads, 1));
        return &p;
    };
    progs.reserve(64);
    if (specs.empty()) specs.push_back({64, 2.0, 0});
    if (mode == "compile") {  // slot program statistics only
        for (const Spec& sc : specs) prog_for(sc);
        return 0;
    }
    const SlotProg* SP = kernel == "slots" ? prog_for(specs[0]) : nullptr;
    if (SP && specs[0].block) block = specs[0].block;
    // lanes that fit on the GPU at once with this kernel (slots: shared memory decides)
    auto resident = [&](const SlotProg* p, u32 block) -> u32 {
#ifdef __CUDACC__
        if (gpu) {
            int nb = 0;
            if (p) {
                CK(cudaFuncSetAttribute(k_slots, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)slots_smem(p->K, block)));
                CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_slots, block, slots_smem(p->K, block)));
            } else CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_sim, block, 0));
            return (u32)std::max(nb, 1) * nsm * block;
        }
#endif
        (void)p; (void)block;
        return lanes;
    };
    auto default_lanes = [&](const SlotProg* p, u32 b) { return std::max<u32>(b, std::min<u32>(maxl, 2 * resident(p, b)) / b * b); };
    if (gpu && !lanes_given) lanes = default_lanes(SP, block);
    auto kname = [&](const SlotProg* p, u32 blk) {
        char b[96];
        if (!p) snprintf(b, sizeof b, "plain, block %u", blk);
        else snprintf(b, sizeof b, "slots K=%u alpha=%g, block %u", p->K, p->alpha, blk);
        return std::string(b);
    };
    if (gpu) fprintf(stderr, "kernel %s: %u lanes resident on the GPU at once; running %u lanes\n", kname(SP, block).c_str(),
                     resident(SP, block), lanes);
    auto quiet = [](const std::vector<Rec>&, double) { return true; };

    if (mode == "crn") {
        u64 start = strtoull(pos.at(2).c_str(), 0, 10), count = strtoull(pos.at(3).c_str(), 0, 10);
        Runner R(job, gpu, std::min<u64>(lanes, std::max<u64>(count, 1)) , block, SP);
        double t0 = Runner::now();
        auto r = R.run(false, false, start, count, quiet);
        print_crn(job, r, Runner::now() - t0, where.c_str());
        return 0;
    }
    if (mode == "check") {
        u64 nonce = strtoull(pos.at(2).c_str(), 0, 10);
        Runner R(job, gpu, 1, block, SP);  // one lane
        double t0 = Runner::now();
        auto r = R.run(true, false, nonce, 1, quiet);
        const u64* v = r.at(0).v;
        printf("nonce %llu: clean=%s shots=%llu classical_failures=%llu phase_garbage_batches=%llu ancilla_garbage_batches=%llu "
               "failing_batches=%llu avg_tof=%.3f (%.1f s on %s)\n", (unsigned long long)nonce, (v[9] & 1) ? "true" : "false",
               (unsigned long long)v[6], (unsigned long long)v[2], (unsigned long long)v[3], (unsigned long long)v[4],
               (unsigned long long)v[1], (double)v[5] / std::max<u64>(v[6], 1), Runner::now() - t0, where.c_str());
        return 0;
    }
    if (mode == "search" || mode == "bench") {
        struct Cfg { u32 L; const SlotProg* p; u32 block; };
        std::vector<Cfg> cfgs;
        u64 start, count;
        if (mode == "search") {
            start = strtoull(pos.at(2).c_str(), 0, 10); count = strtoull(pos.at(3).c_str(), 0, 10);
            cfgs.push_back({lanes, SP, block});
        } else {
            if (seconds <= 0) seconds = pos.size() > 2 ? atof(pos[2].c_str()) : 90;
            start = 1000000000000ULL; count = 1ULL << 40;
            if (gpu && !kernel_given) {  // the plain simulator, then the slot program at several slot counts
                cfgs.push_back({default_lanes(nullptr, 128), nullptr, 128});
                if (cc_major >= 8) {
                    std::vector<Spec> bs = specs_given ? specs
                        : std::vector<Spec>{{32, 2, 128}, {64, 0, 128}, {64, 2, 128}, {64, 2, 64}, {128, 2, 64}};
                    for (const Spec& sc : bs) {
                        const SlotProg* p = prog_for(sc);
                        u32 b = sc.block ? sc.block : 128;
                        cfgs.push_back({default_lanes(p, b), p, b});
                    }
                }
            } else if (gpu && SP && specs.size() > 1) {
                for (const Spec& sc : specs) {
                    const SlotProg* p = prog_for(sc);
                    u32 b = sc.block ? sc.block : 128;
                    cfgs.push_back({lanes_given ? lanes : default_lanes(p, b), p, b});
                }
            } else cfgs.push_back({lanes, SP, block});
        }
        double best_rate = 0; std::string best_name;
        for (const Cfg& cf : cfgs) {
            const u32 L = cf.L;
            std::vector<std::pair<double, u64>> trace;  // (elapsed, finished) after every round
            Runner R(job, gpu, L, cf.block, cf.p);
            SearchStats st;
            size_t seen = 0;
            double last = 0;
            bool found = false;
            char w[200]; snprintf(w, sizeof w, "%s, %s, %u lanes", where.c_str(), kname(cf.p, cf.block).c_str(), L);
            auto on = [&](const std::vector<Rec>& r, double el) {
                st.lane_batches += L;  // every lane holds a nonce and runs one batch per round
                for (; seen < r.size(); seen++) {
                    add_search(st, r[seen]);
                    if (r[seen].v[9] & 1) { printf("CLEAN nonce %llu\n", (unsigned long long)r[seen].v[0]); fflush(stdout); found = true; }
                }
                trace.push_back({el, st.finished});
                if (el - last > 60) { last = el; print_search(st, el, w); }
                if (found && stop_on_clean && mode == "search") return false;  // take the first clean draw found
                return !(seconds > 0 && el > seconds);
            };
            double t0 = Runner::now();
            R.run(true, true, start, count, on);
            double secs = Runner::now() - t0;
            print_search(st, secs, w);
            double rate = st.finished / std::max(secs, 1e-9);
            // the second half of the run alone (the GPU may slow down as it warms up)
            double rate2 = rate;
            for (auto& tr : trace) if (tr.first >= secs / 2) {
                if (trace.back().first > tr.first) rate2 = (double)(trace.back().second - tr.second) / (trace.back().first - tr.first);
                break;
            }
            if (mode == "bench") {
                printf("rate: %.1f nonces/s over the run, %.1f nonces/s over its second half\n", rate, rate2);
                rate = std::min(rate, rate2);
                if (rate > best_rate) { best_rate = rate; best_name = w; }
            }
            printf("at lambda %.2f (e^lambda = %.3e nonces per clean draw on average): %.1f hours on this device per clean draw, "
                   "%.0f devices for a 12-hour average\n", proj_lambda, std::exp(proj_lambda), std::exp(proj_lambda) / rate / 3600,
                   std::ceil(std::exp(proj_lambda) / rate / 3600 / 12));
            double tt = R.tm.seed + R.tm.kw + R.tm.pts + R.tm.sim + R.tm.io;
            printf("time split: seed %.0f%%, k words %.0f%%, points %.0f%%, simulator+checks %.0f%%, copies %.0f%%; %.0f lane-batches/s, "
                   "%.2e gate-lanes/s in the simulator\n", 100 * R.tm.seed / tt, 100 * R.tm.kw / tt, 100 * R.tm.pts / tt, 100 * R.tm.sim / tt,
                   100 * R.tm.io / tt, st.lane_batches / std::max(secs, 1e-9), (double)st.lane_batches * job.ncops / std::max(R.tm.sim, 1e-9));
            fflush(stdout);
            start += st.finished + L;
        }
        if (mode == "bench" && cfgs.size() > 1)
            printf("BEST %s: %.1f nonces/s; at lambda %.2f: %.0f hours on this device per clean draw (90%% within %.0f hours)\n", best_name.c_str(),
                   best_rate, proj_lambda, std::exp(proj_lambda) / best_rate / 3600, std::exp(proj_lambda) / best_rate / 3600 * std::log(10.0));
        return 0;
    }
    if (mode == "selftest") {
        // the GPU (or whatever this build runs on) against the CPU path of this binary, record by record
        u64 ncrn = pos.size() > 2 ? strtoull(pos[2].c_str(), 0, 10) : 4, nsearch = pos.size() > 3 ? strtoull(pos[3].c_str(), 0, 10) : 64;
        int bad = 0;
        auto cmp = [&](const char* what, std::vector<Rec> a, std::vector<Rec> b, bool early) {
            auto by = [](const Rec& x, const Rec& y) { return x.v[0] < y.v[0]; };
            std::sort(a.begin(), a.end(), by); std::sort(b.begin(), b.end(), by);
            int mism = 0;
            if (a.size() != b.size()) { printf("%s: %zu vs %zu records\n", what, a.size(), b.size()); bad++; return; }
            for (size_t i = 0; i < a.size(); i++) if (!same_rec(a[i], b[i], early)) {
                mism++;
                if (mism <= 5) printf("%s MISMATCH id %llu: device %llu/%llu/%llu tof %llu, cpu %llu/%llu/%llu tof %llu\n", what,
                    (unsigned long long)a[i].v[0], (unsigned long long)a[i].v[1], (unsigned long long)a[i].v[2], (unsigned long long)a[i].v[3],
                    (unsigned long long)a[i].v[5], (unsigned long long)b[i].v[1], (unsigned long long)b[i].v[2], (unsigned long long)b[i].v[3],
                    (unsigned long long)b[i].v[5]);
            }
            printf("%s: %zu records, %d mismatches\n", what, a.size(), mism);
            bad += mism;
        };
        u32 cl = std::max(8, threads);
        double t0 = Runner::now();
        std::vector<Rec> dc, cc, ds, cs;
        { Runner R(job, gpu, gpu ? lanes : cl, block, SP); dc = R.run(false, false, 0, ncrn, quiet); }
        printf("device CRN draws 0..%llu: %.1f s\n", (unsigned long long)ncrn - 1, Runner::now() - t0);
        t0 = Runner::now();
        { Runner R(job, gpu, gpu ? lanes : cl, block, SP); ds = R.run(true, true, 1000, nsearch, quiet); }
        printf("device search nonces 1000..%llu: %.1f s\n", (unsigned long long)(1000 + nsearch - 1), Runner::now() - t0);
        print_crn(job, dc, 0, where.c_str());
        if (gpu || SP) {  // against the plain simulator on the CPU
            t0 = Runner::now();
            { Runner R(job, false, cl, block); cc = R.run(false, false, 0, ncrn, quiet); }
            { Runner R(job, false, cl, block); cs = R.run(true, true, 1000, nsearch, quiet); }
            printf("CPU reference: %.1f s\n", Runner::now() - t0);
            cmp("CRN draws", dc, cc, false);
            cmp("search nonces", ds, cs, true);
            printf(bad ? "SELFTEST FAILED\n" : "SELFTEST OK: %s (%s) matches the plain CPU simulator record for record\n",
                   gpu ? "the GPU" : "the CPU", kname(SP, block).c_str());
        }
        for (auto& r : ds) printf("nonce %llu passed_batches %llu clean %s\n", (unsigned long long)r.v[0], (unsigned long long)r.v[1], (r.v[9] & 1) ? "true" : "false");
        return bad ? 1 : 0;
    }
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
