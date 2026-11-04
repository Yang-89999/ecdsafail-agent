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
//   gpu_search <job> bench [seconds]           search rate of the exact plain simulator and the fast filter variants
//   gpu_search <job> faildump <start> <count>  (CPU) every failing shot of CRN draws, with inputs and outputs
// options: --kernel fast|plain|slots (search and bench default to fast: the classical filter on the fast simulator,
//          every draw it passes then checked in full on the CPU; other modes default to plain, the exact simulator),
//          --fast V[,V...]|all (fast simulator variants, see FVARS), --filter (plain simulator on the filter stream),
//          --lanes N (draws in flight), --block N (threads per simulator block), --seconds S (search time limit),
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
// classical filter (filter_ops): RST resets a qubit (an HMR or R with its measurement outcome taken as 0)
enum { K_RST = 18 };
// fast simulator of the filter stream (compile_fast): 16-byte entries {k, t, a, b} on rows (qubit q: row q, bit b:
// row nq + b, then an all-ones row ONES standing in for an absent control). Almost every op is t ^= cond & a & b (CX,
// CCX and X with ONES for absent controls; a reset RST is t ^= cond & t, which clears t): those come in runs, each
// after a RUN entry {F_RUN, n} that never crosses a staging chunk; the rarer kinds stand alone. In the copy a Runner
// uploads, t, a, b are word offsets row * L, so an address is one multiply-add.
enum : u32 { F_RUN = 0, F_XOR, F_SWAP, F_B0, F_B1, F_BINV, F_PUSH, F_POP };
struct alignas(16) FOp { u32 k, t, a, b; };
static const int FMAXS = 4;       // condition depth of the fast simulator
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
    const FOp* fops; u32 nfops;     // fast filter simulator (null: not used)
    int dump;                       // CPU only: print every failing shot (faildump mode)
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
            case K_RST: Ql[qt * L] &= ~cond; break;
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

// CPU research (census mode): sim_ops, also counting per CCX/CCZ op the shots on which it fires (both controls 1
// under the condition), i.e. the shots whose result depends on that Toffoli at all
static void sim_census(u64* __restrict__ Ql, u64* __restrict__ Bl, const u32 L, const COp* ops, u32 n, Ctx& c, u32* fires) {
    for (u32 i = 0; i < n; i++) {
        const u32 w0 = ops[i].w0, w1 = ops[i].w1;
        const u32 k = w0 & 0xFF, qt = w0 >> 16, q1 = w1 & 0xFFFF, q2 = w1 >> 16;
        if (k == K_CCX || k == K_CCZ) {
            u64 cond = c.base;
            if (w0 & 0xFF00) cond &= Bl[(ops[i].w2 >> 16) * L];
            u64 f = cond & Ql[q1 * L] & Ql[q2 * L];
            if (k == K_CCZ) f &= Ql[qt * L];
            fires[2 * i] += popc64(f);
            fires[2 * i + 1] += popc64(cond);
        }
        sim_ops(Ql, Bl, L, ops + i, 1, c);
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

// ─── fast filter simulator ────────────────────────────────────────────────────────────────────────────────────
// 97% of the filter stream is t ^= cond & a & b (CX, CCX, X with ONES for absent controls) or t &= ~cond (RST), so
// both share one branch-free path: m = base & a & b, t ^= m & (t | X) with X = ~0 for XOR and 0 for a reset. There is
// no Toffoli count, no phase and no randomness. A thread runs LPT adjacent lanes (on the GPU one 16-byte access serves
// two lanes); a lane that is not active has base 0, so every op leaves its rows alone.
template <int LPT> struct VW { u64 v[LPT]; };
template <int LPT> HD VW<LPT> vld(const u64* p) {
    VW<LPT> r;
#ifdef __CUDA_ARCH__
    if constexpr (LPT == 2) { ulonglong2 x = *reinterpret_cast<const ulonglong2*>(p); r.v[0] = x.x; r.v[1] = x.y; return r; }
#endif
    for (int i = 0; i < LPT; i++) r.v[i] = p[i];
    return r;
}
template <int LPT> HD void vst(u64* p, const VW<LPT>& r) {
#ifdef __CUDA_ARCH__
    if constexpr (LPT == 2) { *reinterpret_cast<ulonglong2*>(p) = make_ulonglong2(r.v[0], r.v[1]); return; }
#endif
    for (int i = 0; i < LPT; i++) p[i] = r.v[i];
}
// base in registers; the condition stack (PUSH and POP are 0.5% of the ops) in a small per-thread array
template <int LPT> struct FCtx { u64 base[LPT]; u64 stk[FMAXS * LPT]; u32 sp; bool over; };

template <int LPT> HD void fast_slow(u64* __restrict__ R, const FOp o, FCtx<LPT>& c) {
    u64* pt = R + o.t;
    VW<LPT> t = vld<LPT>(pt);
    switch (o.k) {
        case F_SWAP: {
            u64* pa = R + o.a;
            VW<LPT> a = vld<LPT>(pa);
            for (int j = 0; j < LPT; j++) { u64 x = (a.v[j] ^ t.v[j]) & c.base[j]; a.v[j] ^= x; t.v[j] ^= x; }
            vst<LPT>(pa, a); vst<LPT>(pt, t);
        } break;
        case F_B0: for (int j = 0; j < LPT; j++) t.v[j] &= ~c.base[j]; vst<LPT>(pt, t); break;
        case F_B1: for (int j = 0; j < LPT; j++) t.v[j] |= c.base[j]; vst<LPT>(pt, t); break;
        case F_BINV: for (int j = 0; j < LPT; j++) t.v[j] ^= c.base[j]; vst<LPT>(pt, t); break;
        case F_PUSH:
            if (c.sp >= (u32)FMAXS) { c.over = true; break; }
            for (int j = 0; j < LPT; j++) { c.stk[c.sp * LPT + j] = c.base[j]; c.base[j] &= t.v[j]; }
            c.sp++; break;
        case F_POP:
            if (c.sp > 0) { c.sp--; for (int j = 0; j < LPT; j++) c.base[j] = c.stk[c.sp * LPT + j]; }
            break;
        default: break;
    }
}
template <int LPT> HD void sim_fast(u64* __restrict__ R, const FOp* ops, u32 n, FCtx<LPT>& c) {
    for (u32 i = 0; i < n;) {
        const FOp h = ops[i++];
        if (h.k == F_RUN) {
            const u32 e = i + h.t;
#ifdef __CUDA_ARCH__
#pragma unroll 4
#endif
            for (; i < e; i++) {  // t ^= cond & a & b
                const FOp o = ops[i];
                u64* pt = R + o.t;
                const VW<LPT> a = vld<LPT>(R + o.a), b = vld<LPT>(R + o.b);
                VW<LPT> t = vld<LPT>(pt);
                for (int j = 0; j < LPT; j++) t.v[j] ^= c.base[j] & a.v[j] & b.v[j];
                vst<LPT>(pt, t);
            }
        } else fast_slow<LPT>(R, h, c);
    }
}
template <int LPT> HD void fctx_init(FCtx<LPT>& c, const bool* active) {
    c.sp = 0; c.over = false;
    for (int j = 0; j < LPT; j++) c.base[j] = active[j] ? ~0ULL : 0ULL;
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

#ifndef __CUDA_ARCH__
// faildump: one line per failing shot with its inputs, the circuit's outputs and the expected sum (hex, big-endian)
static void dump_failures(const Dev& d, u32 l, u64 cl, u64 ph, u64 anc) {
    const u32 L = d.L;
    const u64* Ql = d.Q + l; const u64* El = d.E + l;
    for (u32 s = 0; s < 64; s++) {
        if (!(((cl | ph | anc) >> s) & 1)) continue;
        Fe v[6];
        points_shot(d, l, s, v);
        Fe out[2] = {fe_small(0), fe_small(0)}, ex[2] = {fe_small(0), fe_small(0)};
        for (int w = 0; w < 2; w++)
            for (u32 i = 0; i < 256; i++) {
                out[w].v[i >> 6] |= ((Ql[(size_t)d.reg[w * 256 + i] * L] >> s) & 1) << (i & 63);
                ex[w].v[i >> 6] |= ((El[(size_t)(w * 256 + i) * L] >> s) & 1) << (i & 63);
            }
        char line[1200]; int n = 0;
        n += snprintf(line + n, sizeof line - n, "FAIL %llu %u %u %d %d %d", (unsigned long long)d.id[l], d.batch[l], s,
                      (int)((cl >> s) & 1), (int)((ph >> s) & 1), (int)((anc >> s) & 1));
        const Fe* f[8] = {&v[0], &v[1], &v[2], &v[3], &out[0], &out[1], &ex[0], &ex[1]};
        for (auto* x : f) n += snprintf(line + n, sizeof line - n, " %016llx%016llx%016llx%016llx", (unsigned long long)x->v[3],
                                        (unsigned long long)x->v[2], (unsigned long long)x->v[1], (unsigned long long)x->v[0]);
#pragma omp critical
        { puts(line); }
    }
}
#endif

// end of a batch: the official checks (classical per shot, phase, ancilla after zeroing register qubits), then clear
// end of a batch: the official checks (classical per shot, phase, ancilla after zeroing register qubits), then clear
HD void finish_checks(const Dev& d, u32 l, u64 phase, u64 tof, bool overflow) {
    const u32 L = d.L;
    u64* Ql = d.Q + l; u64* Bl = d.B + l; const u64* El = d.E + l;
    u64 cl = 0;
    for (u32 i = 0; i < 256; i++) {
        cl |= Ql[(size_t)d.reg[i] * L] ^ El[(size_t)i * L];
        cl |= Ql[(size_t)d.reg[256 + i] * L] ^ El[(size_t)(256 + i) * L];
    }
    u64 ph = phase, anc = 0;
    for (u32 q = 0; q < d.nq; q++) { u64 v = Ql[(size_t)q * L]; if (!d.isreg[q]) anc |= v; }
#ifndef __CUDA_ARCH__
    if (d.dump && (cl | ph | anc)) dump_failures(d, l, cl, ph, anc);
#endif
    for (u32 q = 0; q < d.nq; q++) Ql[(size_t)q * L] = 0;
    for (u32 b = 0; b < d.nb; b++) Bl[(size_t)b * L] = 0;
    bool ok = !cl && !ph && !anc;
    u32* a = d.acc + (size_t)l * NACC;
    if (overflow) a[A_FLAGS] |= 4;  // condition stack overflow (the host checks the depth; never expected)
    a[A_SHOTS] += 64;
    a[A_CLASS] += popc64(cl);
    a[A_FSHOTS] += popc64(cl | ph);
    a[A_PSHOTS] += popc64(ph & ~cl);
    a[A_PHB] += ph != 0;
    a[A_ANCB] += anc != 0;
    a[A_FAIL] += !ok;
    d.tofacc[l] += tof;
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
HD void finish_lane(const Dev& d, u32 l, const Ctx& c) {
    d.rused[l] = c.used;
    for (int i = 0; i < 25; i++) d.rsp[(size_t)i * d.L + l] = c.st[i];
    finish_checks(d, l, c.phase, c.tof, c.sp > MAXS);
}

static const u32 FCH = 256;  // fast entries staged per chunk (4 KB: shared memory left to the L1 cache)
static const u32 FBS = 128;  // fast simulator threads per block
// fast simulator variants (lanes per thread, minimum blocks per SM); variant 0 is the default
struct FastVar { int lpt, minb; };
static const FastVar FVARS[] = {{2, 8}, {1, 8}, {2, 12}, {1, 16}, {2, 6}, {2, 16}};
static const int NFVARS = sizeof FVARS / sizeof FVARS[0];

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
// the fast filter simulator: FBS threads per block, LPT lanes per thread, at least MINB blocks per SM (which caps the
// registers per thread, so more lanes are in flight to hide memory latency)
template <int LPT, int MINB> __global__ void __launch_bounds__(FBS, MINB) k_fast(Dev d) {
    __shared__ FOp sh[FCH];
    const u32 l0 = (blockIdx.x * FBS + threadIdx.x) * LPT;
    FCtx<LPT> c;
    bool act[LPT], any = false;
    for (int j = 0; j < LPT; j++) { act[j] = l0 + j < d.L && d.state[l0 + j] == 1; any |= act[j]; }
    fctx_init<LPT>(c, act);
    u64* R = d.Q + (l0 < d.L ? l0 : 0);
    for (u32 b0 = 0; b0 < d.nfops; b0 += FCH) {
        u32 n = min(FCH, d.nfops - b0);
        __syncthreads();
        for (u32 i = threadIdx.x; i < n; i += FBS) sh[i] = d.fops[b0 + i];
        __syncthreads();
        if (any) sim_fast<LPT>(R, sh, n, c);
    }
    for (int j = 0; j < LPT; j++)
        if (act[j]) finish_checks(d, l0 + j, 0, 0, c.over || c.sp != 0);
}
typedef void (*FastKernel)(Dev);
static FastKernel fast_kernel(int v) {
    switch (v) {
        case 0: return k_fast<2, 8>;
        case 1: return k_fast<1, 8>;
        case 2: return k_fast<2, 12>;
        case 3: return k_fast<1, 16>;
        case 4: return k_fast<2, 6>;
        default: return k_fast<2, 16>;
    }
}

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

// ─── classical filter: the op stream with every measurement outcome taken as 0 ───────────────────────────────────
// A nonce search only has to find a draw with no failing shot. Most of the simulator's work that does not decide the
// classical and ancilla checks can go: the phase ops (CZ, Z, CCZ, NEG, and the phase kickback of HMR and R) and the
// random measurement outcomes, which cost one Keccak permutation per 17 outcomes. With every outcome taken as 0 an HMR
// or R only resets its qubit, every block conditioned on an outcome is dead, and the classical input bits (ox, oy) are
// the only bits that can be nonzero besides those the circuit sets itself. The filter's classical and ancilla verdict
// equals the official one whenever those checks do not depend on the outcomes (public head, CRN draws 0-99: the same
// 1,897 failing shots, see docs/gpu.md); a draw the filter passes is then checked in full with the official outcomes
// and phase, so the filter only decides which draws are worth a full check.
struct FilterStats { u64 in = 0, out = 0, phase = 0, dead = 0, rst = 0, bitops = 0, folded = 0; };
static std::vector<COp> filter_ops(const Job& j, FilterStats& st) {
    // Every bit row stays exact (a bit op is dropped only when it cannot change the row); what is known about each
    // bit (0, 1 or unknown) folds the conditions: a block or op conditioned on a known 0 is dead, on a known 1 it
    // runs unconditioned.
    enum : u8 { BZ = 0, BO = 1, BU = 2 };
    std::vector<u8> bv(j.nb, BZ);                             // rows start at 0 in every batch
    for (int i = 512; i < 1024; i++) bv[j.reg[i]] = BU;       // the classical input point
    std::vector<u8> stk;                                      // per open block: 0 dropped (condition 1), 1 emitted
    u32 emitted = 0;                                          // emitted blocks open: the base condition is unknown
    std::vector<COp> out;
    out.reserve(j.ops.size());
    int dead = 0;                                             // > 0: inside blocks whose condition is 0 (nesting)
    for (const COp& o0 : j.ops) {
        st.in++;
        COp o = o0;
        const u32 k = o.w0 & 0xFF, cb = o.w2 & 0xFFFF, cc = o.w2 >> 16;
        if (dead) { if (k == K_PUSH) dead++; else if (k == K_POP) dead--; st.dead++; continue; }
        if (k == K_PUSH) {                                    // (a PUSH carries its condition bit as a flag)
            if (bv[cc] == BZ) { dead = 1; st.dead++; }
            else if (bv[cc] == BO) { stk.push_back(0); st.folded++; }
            else { stk.push_back(1); emitted++; out.push_back(o); }
            continue;
        }
        if (k == K_POP) {
            if (stk.empty()) continue;
            if (stk.back()) { emitted--; out.push_back(o); } else st.folded++;
            stk.pop_back();
            continue;
        }
        bool fl = (o.w0 & 0xFF00) != 0;
        if (fl && bv[cc] == BZ) { st.dead++; continue; }
        if (fl && bv[cc] == BO) { o.w0 &= ~0xFF00u; fl = false; st.folded++; }
        const bool ones = !emitted && !fl;                    // this op's condition is all ones in every shot
        switch (k) {
            case K_CZ: case K_Z: case K_CCZ: case K_NEG: st.phase++; continue;
            case K_HMR: case K_B0: {                          // the bit becomes bit & ~cond (outcome 0)
                if (k == K_HMR) { COp r = o; r.w0 = (o.w0 & ~0xFFu) | K_RST; out.push_back(r); st.rst++; }
                if (bv[cb] == BZ) { st.folded++; continue; }
                COp b = o; b.w0 = (o.w0 & 0xFF00) | K_B0; if (k == K_HMR) b.w1 = 0;
                out.push_back(b); st.bitops++;
                bv[cb] = ones ? BZ : BU;
                continue;
            }
            case K_R: o.w0 = (o.w0 & ~0xFFu) | K_RST; st.rst++; break;
            case K_B1:
                if (bv[cb] == BO) { st.folded++; continue; }
                bv[cb] = ones ? BO : BU; st.bitops++; break;
            case K_BINV:
                bv[cb] = (ones && bv[cb] != BU) ? (u8)(bv[cb] ^ 1) : BU; st.bitops++; break;
            default: break;
        }
        out.push_back(o);
    }
    st.out = out.size();
    return out;
}

// the filter stream as fast-simulator entries; fails (empty) on an op the fast simulator does not take
static std::vector<FOp> compile_fast(const Job& j, std::string& err) {
    const u32 nq = j.nq, ONES = j.nq + j.nb;
    std::vector<FOp> f;
    f.reserve(j.ops.size() + j.ops.size() / 64 + 16);
    size_t run = SIZE_MAX;                                    // index of the open RUN entry
    auto xop = [&](u32 t, u32 a, u32 b) {
        if (run == SIZE_MAX || (f.size() % FCH) == 0) {       // open a run (a new chunk always starts with one)
            if (f.size() % FCH == FCH - 1) f.push_back({F_RUN, 0, 0, 0});  // no room for an op after a header here
            run = f.size(); f.push_back({F_RUN, 0, 0, 0});
        }
        f.push_back({F_XOR, t, a, b}); f[run].t++;
    };
    auto single = [&](u32 k, u32 t, u32 a) { run = SIZE_MAX; f.push_back({k, t, a, ONES}); };
    u32 dep = 0, maxdep = 0;
    for (const COp& o : j.ops) {
        const u32 w0 = o.w0, w1 = o.w1, w2 = o.w2;
        const u32 k = w0 & 0xFF, qt = w0 >> 16, q1 = w1 & 0xFFFF, q2 = w1 >> 16, cb = w2 & 0xFFFF, cc = w2 >> 16;
        if ((w0 & 0xFF00) && k != K_PUSH) { err = "an op with its own condition bit"; return {}; }
        switch (k) {
            case K_CX: xop(qt, q1, ONES); break;
            case K_CCX: xop(qt, q1, q2); break;
            case K_X: xop(qt, ONES, ONES); break;
            case K_RST: xop(qt, qt, ONES); break;             // t ^= cond & t clears t where cond is set
            case K_SWAP: single(F_SWAP, qt, q1); break;
            case K_B0: single(F_B0, nq + cb, ONES); break;
            case K_B1: single(F_B1, nq + cb, ONES); break;
            case K_BINV: single(F_BINV, nq + cb, ONES); break;
            case K_PUSH: single(F_PUSH, nq + cc, ONES); maxdep = std::max(maxdep, ++dep); break;
            case K_POP: single(F_POP, ONES, ONES); if (dep) dep--; break;
            default: { char b[64]; snprintf(b, sizeof b, "op kind %u", k); err = b; return {}; }
        }
    }
    if (maxdep > (u32)FMAXS) { err = "condition depth above " + std::to_string(FMAXS); return {}; }
    return f;
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
    const std::vector<FOp>* fast;  // fast filter simulator entries (j must then be the filter job)
    int fvar;

    Runner(const Job& job, bool use_gpu, u32 lanes, u32 blk, const SlotProg* slots = nullptr, const std::vector<FOp>* fops = nullptr,
           int fv = 0)
        : j(job), gpu(use_gpu), L(lanes), block(blk), sp(slots), fast(fops), fvar(fv) {
        size_t rows = ((size_t)j.nq + j.nb + 1) * L;
        if (rows >= (1ULL << 32)) { fprintf(stderr, "too many lanes for 32-bit row offsets\n"); exit(1); }
        d.L = L; d.nq = j.nq; d.nb = j.nb; d.ncops = j.ncops; d.ntests = j.ntests; d.ppos = j.ppos; d.outcap = L;
        d.ops = (const COp*)put(j.ops.data(), j.ops.size() * sizeof(COp));
        d.reg = (const u32*)put(j.reg.data(), j.reg.size() * 4);
        d.isreg = (const u8*)put(j.isreg.data(), j.isreg.size());
        d.tab = (const u64*)put(j.tab.data(), j.tab.size() * 8);
        d.pst = (const u64*)put(j.pst, sizeof j.pst);
        d.pbuf = (const u8*)put(j.pbuf, sizeof j.pbuf);
        d.tail = (const u8*)put(j.tail.data(), j.tail.size());
        d.Q = (u64*)zeros(((size_t)j.nq + j.nb + 1) * L * 8);  // qubit rows, bit rows (the slot program's row ids), ONES
        d.B = d.Q + (size_t)j.nq * L;
        fill_ones(d.Q + ((size_t)j.nq + j.nb) * L, L);
        if (fast) {
            if (gpu && L % FVARS[fvar].lpt) { fprintf(stderr, "lanes must be a multiple of %d\n", FVARS[fvar].lpt); exit(1); }
            std::vector<FOp> fl(*fast);  // row ids -> word offsets row * L
            for (FOp& o : fl) if (o.k != F_RUN) { o.t *= L; o.a *= L; o.b *= L; }
            d.fops = (const FOp*)put(fl.data(), fl.size() * sizeof(FOp));
            d.nfops = (u32)fl.size();
        }
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
    static size_t bytes_per_lane(const Job& j) { return ((size_t)j.nq + j.nb + 1 + 512 + KW + 50 + 3 + RECW) * 8 + (4 + NACC) * 4; }
    void fill_ones(u64* p, size_t n) {
#ifdef __CUDACC__
        if (gpu) { CK(cudaMemset(p, 0xFF, n * 8)); return; }
#endif
        memset(p, 0xFF, n * 8);
    }

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
            if (fast) { u32 per = FBS * FVARS[fvar].lpt; fast_kernel(fvar)<<<(L + per - 1) / per, FBS>>>(d); }
            else if (sp) k_slots<<<(L + block - 1) / block, block, slots_smem(sp->K, block)>>>(d);
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
            if (fast) {
                FCtx<1> fc;
                const bool act[1] = {true};
                fctx_init<1>(fc, act);
                sim_fast<1>(dd.Q + l, dd.fops, dd.nfops, fc);
                finish_checks(dd, l, 0, 0, fc.over || fc.sp != 0);
                continue;
            }
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

    // ── invariant mining (CPU, research): which wires are constant over passing shots at C checkpoints, and how
    // early failing shots break one of those constants
    struct Mine {
        u32 C = 0, NR = 0;
        std::vector<u32> cps;             // checkpoint op indices (state after that many ops)
        std::vector<u64> Z, O;            // C x NR: some passing shot had a 1 / a 0 there
        std::vector<u64> fhist, phist;    // earliest violated checkpoint (C = none) of failing / passing shots
        u64 samples = 0;
    };
    void round_mine(Mine& m, bool eval) {
        const Dev dd = d;
        for (u32 l = 0; l < L; l++) seed_lane(dd, l);
#pragma omp parallel for schedule(dynamic, 1)
        for (u32 l = 0; l < L; l++) kwords_lane(dd, l);
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
                    u64 mm = 0;
                    for (u32 s = 0; s < 64; s++) mm |= ((v[w][s].v[i >> 6] >> (i & 63)) & 1) << s;
                    *plane(dd, l, w, i) = mm;
                }
        }
        std::vector<u64> lZ, lO;
        std::vector<u64> lf(m.C + 1, 0), lp(m.C + 1, 0);
#pragma omp parallel for schedule(dynamic, 1)
        for (u32 l = 0; l < L; l++) {
            if (dd.state[l] != 1) continue;
            const u32 NR = m.NR;
            std::vector<u64> snap((size_t)m.C * NR);
            Ctx c;
            ctx_load(dd, l, c);
            u32 at = 0;
            for (u32 k = 0; k < m.C; k++) {
                sim_ops(dd.Q + l, dd.B + l, L, dd.ops + at, m.cps[k] - at, c);
                at = m.cps[k];
                for (u32 r = 0; r < NR; r++) snap[(size_t)k * NR + r] = dd.Q[(size_t)r * L + l];  // Q then B rows: one buffer
            }
            sim_ops(dd.Q + l, dd.B + l, L, dd.ops + at, dd.ncops - at, c);
            const u64* Ql = dd.Q + l; const u64* El = dd.E + l;
            u64 cl = 0;
            for (u32 i = 0; i < 256; i++) {
                cl |= Ql[(size_t)dd.reg[i] * L] ^ El[(size_t)i * L];
                cl |= Ql[(size_t)dd.reg[256 + i] * L] ^ El[(size_t)(256 + i) * L];
            }
            u64 anc = 0;
            for (u32 q = 0; q < dd.nq; q++) if (!dd.isreg[q]) anc |= Ql[(size_t)q * L];
            u64 fail = cl | c.phase | anc, pass = ~fail;
            if (!eval) {
#pragma omp critical
                {
                    for (size_t x = 0; x < snap.size(); x++) { m.Z[x] |= snap[x] & pass; m.O[x] |= ~snap[x] & pass; }
                    m.samples += popc64(pass);
                }
            } else {
                u64 seen = 0;
                u32 first[64];
                for (int s2 = 0; s2 < 64; s2++) first[s2] = m.C;
                for (u32 k = 0; k < m.C && ~seen; k++) {
                    u64 viol = 0;
                    const u64* sn = &snap[(size_t)k * NR];
                    const u64* z = &m.Z[(size_t)k * NR]; const u64* o = &m.O[(size_t)k * NR];
                    for (u32 r = 0; r < NR; r++) viol |= (z[r] == 0 ? sn[r] : 0) | (o[r] == 0 ? ~sn[r] : 0);
                    viol &= ~seen;
                    for (int s2 = 0; s2 < 64; s2++) if ((viol >> s2) & 1) first[s2] = k;
                    seen |= viol;
                }
#pragma omp critical
                for (int s2 = 0; s2 < 64; s2++) { if ((fail >> s2) & 1) m.fhist[first[s2]]++; else m.phist[first[s2]]++; }
            }
            finish_lane(dd, l, c);
        }
        *d.outn = 0;
    }

    // census: one round of 64-shot batches, every lane simulated with sim_census; fires summed into tot
    void round_census(std::vector<u64>& tot, u64& shots) {
        const Dev dd = d;
        for (u32 l = 0; l < L; l++) seed_lane(dd, l);
#pragma omp parallel for schedule(dynamic, 1)
        for (u32 l = 0; l < L; l++) kwords_lane(dd, l);
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
                    u64 mm = 0;
                    for (u32 s = 0; s < 64; s++) mm |= ((v[w][s].v[i >> 6] >> (i & 63)) & 1) << s;
                    *plane(dd, l, w, i) = mm;
                }
        }
#pragma omp parallel
        {
            std::vector<u32> f(2 * (size_t)dd.ncops, 0);
            u64 sh = 0;
#pragma omp for schedule(dynamic, 1)
            for (u32 l = 0; l < L; l++) {
                if (dd.state[l] != 1) continue;
                Ctx c;
                ctx_load(dd, l, c);
                sh += popc64(c.base);
                sim_census(dd.Q + l, dd.B + l, L, dd.ops, dd.ncops, c, f.data());
                finish_lane(dd, l, c);
            }
#pragma omp critical
            {
                for (size_t i = 0; i < 2 * (size_t)dd.ncops; i++) tot[i] += f[i];
                shots += sh;
            }
        }
        *d.outn = 0;
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

static bool same_rec(const Rec& a, const Rec& b, bool early, bool notof = false) {
    if (early) return a.v[0] == b.v[0] && a.v[1] == b.v[1] && (a.v[9] & 1) == (b.v[9] & 1);
    for (int i = 0; i < RECW; i++) if (a.v[i] != b.v[i] && !(notof && i == 5)) return false;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: gpu_search <job.bin> crn|search|check|selftest|bench ... [--lanes N] [--block N] [--seconds S] [--cpu] [--threads N]\n");
        return 1;
    }
    std::vector<std::string> pos;
    u32 lanes = 0, block = 128;
    bool lanes_given = false, kernel_given = false;
    double seconds = 0;
    bool force_cpu = false, stop_on_clean = true, filter = false;
    std::vector<int> fvars;      // --fast: fast simulator variants (FVARS indices)
    bool fvars_given = false;
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
        else if (a == "--filter") filter = true;
        else if (a == "--fast" && i + 1 < argc) {
            std::string v = argv[++i];
            fvars.clear(); fvars_given = true;
            if (v == "all") for (int k = 0; k < NFVARS; k++) fvars.push_back(k);
            else for (size_t p0 = 0; p0 < v.size();) {
                size_t p1 = v.find(',', p0); if (p1 == std::string::npos) p1 = v.size();
                int k = atoi(v.substr(p0, p1 - p0).c_str());
                if (k < 0 || k >= NFVARS) { fprintf(stderr, "--fast: variants are 0..%d\n", NFVARS - 1); return 1; }
                fvars.push_back(k); p0 = p1 + 1;
            }
        }
        else if (a == "--no-stop") stop_on_clean = false;
        else if (a == "--lambda" && i + 1 < argc) proj_lambda = atof(argv[++i]);
        else if (a == "--device" && i + 1 < argc) device = atoi(argv[++i]);
        else if (a == "--kernel" && i + 1 < argc) { kernel = argv[++i]; kernel_given = true; }
        else if (a == "--slots" && i + 1 < argc) {
            std::string v = argv[++i];
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
        maxl = std::min<u32>(maxl, (u32)(0xFFFFFFFFull / ((u64)job.nq + job.nb + 1)) / 256 * 256);  // 32-bit row offsets
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
    Job fjob;                    // the classical filter stream (filter_ops), built when needed
    std::vector<FOp> fops;       // and its fast-simulator entries (empty: the fast simulator cannot run this circuit)
    bool fbuilt = false;
    auto build_filter = [&]() {
        if (fbuilt) return;
        fbuilt = true;
        FilterStats fs;
        fjob = job;
        fjob.ops = filter_ops(job, fs);
        fjob.ncops = (u32)fjob.ops.size();
        fprintf(stderr, "classical filter: %llu of %llu ops kept (%llu phase ops and %llu ops conditioned on a 0 outcome dropped; "
                "%llu resets, %llu bit ops, %llu conditions folded)\n", (unsigned long long)fs.out, (unsigned long long)fs.in,
                (unsigned long long)fs.phase, (unsigned long long)fs.dead, (unsigned long long)fs.rst, (unsigned long long)fs.bitops,
                (unsigned long long)fs.folded);
        std::string err;
        fops = compile_fast(fjob, err);
        if (fops.empty()) fprintf(stderr, "the fast simulator cannot run this circuit (%s): using the plain simulator\n", err.c_str());
    };
    if (kernel.empty()) kernel = (mode == "search" || mode == "bench") ? "fast" : "plain";
    if (kernel != "slots" && kernel != "plain" && kernel != "fast") { fprintf(stderr, "--kernel is fast, plain or slots\n"); return 1; }
    if (kernel == "fast") { build_filter(); if (fops.empty()) kernel = "plain"; }
    if (filter || kernel == "fast") build_filter();
    if (fvars.empty()) fvars.push_back(0);
    const Job* J = (filter || kernel == "fast") ? &fjob : &job;   // the stream the runs simulate
    const std::vector<FOp>* F = kernel == "fast" ? &fops : nullptr;
    int fvar = fvars[0];
    if (F) block = FBS;
    if (mode == "compile" && fbuilt) {
        static const char* KN[] = {"CCX", "CX", "SWAP", "X", "CCZ", "CZ", "Z", "NEG", "HMR", "R", "BINV", "B0", "B1", "PUSH", "POP",
                                   "?", "?", "?", "RST"};
        u64 cnt[19] = {0}, flg = 0; u32 dep = 0, maxdep = 0;
        for (const COp& o : fjob.ops) {
            u32 k = o.w0 & 0xFF; cnt[k < 19 ? k : 15]++;
            if (k == K_PUSH) maxdep = std::max(maxdep, ++dep); else if (k == K_POP) dep--; else if (o.w0 & 0xFF00) flg++;
        }
        fprintf(stderr, "filter ops:");
        for (int k = 0; k < 19; k++) if (cnt[k]) fprintf(stderr, " %s %.2f%%", KN[k], 100.0 * cnt[k] / fjob.ncops);
        fprintf(stderr, "; flagged (not PUSH) %llu; condition depth %u; fast entries %zu\n", (unsigned long long)flg, maxdep, fops.size());
        return 0;
    }
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
    // a simulator choice: the slot program p, the fast filter simulator variant fv (-1: none), threads per block
    struct KSel { const SlotProg* p; int fv; u32 block; };
    auto lpt = [](const KSel& k) -> u32 { return k.fv >= 0 ? (u32)FVARS[k.fv].lpt : 1u; };
    // lanes that fit on the GPU at once with this kernel (slots: shared memory decides; fast: registers)
    auto resident = [&](const KSel& k) -> u32 {
#ifdef __CUDACC__
        if (gpu) {
            int nb = 0;
            if (k.fv >= 0) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fast_kernel(k.fv), FBS, 0));
            else if (k.p) {
                CK(cudaFuncSetAttribute(k_slots, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)slots_smem(k.p->K, k.block)));
                CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_slots, k.block, slots_smem(k.p->K, k.block)));
            } else CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_sim, k.block, 0));
            return (u32)std::max(nb, 1) * nsm * k.block * lpt(k);
        }
#endif
        (void)k;
        return lanes;
    };
    auto default_lanes = [&](const KSel& k) {
        u32 g = k.block * lpt(k);
        return std::max<u32>(g, std::min<u32>(maxl, 2 * resident(k)) / g * g);
    };
    auto kname = [&](const KSel& k) {
        char b[128];
        if (k.fv >= 0) snprintf(b, sizeof b, "fast filter v%d (%d lanes/thread, >= %d blocks/SM), block %u", k.fv, FVARS[k.fv].lpt,
                                FVARS[k.fv].minb, FBS);
        else if (!k.p) snprintf(b, sizeof b, "plain, block %u", k.block);
        else snprintf(b, sizeof b, "slots K=%u alpha=%g, block %u", k.p->K, k.p->alpha, k.block);
        return std::string(b);
    };
    const KSel cur{SP, F ? fvar : -1, block};
    // lane counts must be a multiple of the lanes per thread on the GPU (the CPU path runs one lane at a time)
    auto fitL = [&](u64 want, const KSel& k) -> u32 { u32 g = gpu ? lpt(k) : 1; return (u32)((std::max<u64>(want, 1) + g - 1) / g * g); };
    if (gpu && !lanes_given) lanes = default_lanes(cur);
    if (gpu) lanes = fitL(lanes, cur);
    if (gpu) fprintf(stderr, "kernel %s: %u lanes resident on the GPU at once; running %u lanes\n", kname(cur).c_str(), resident(cur), lanes);
    auto quiet = [](const std::vector<Rec>&, double) { return true; };

    if (mode == "crn") {
        u64 start = strtoull(pos.at(2).c_str(), 0, 10), count = strtoull(pos.at(3).c_str(), 0, 10);
        Runner R(*J, gpu, fitL(std::min<u64>(lanes, std::max<u64>(count, 1)), cur), block, SP, F, fvar);
        double t0 = Runner::now();
        auto r = R.run(false, false, start, count, quiet);
        print_crn(job, r, Runner::now() - t0, where.c_str());
        return 0;
    }
    if (mode == "check") {
        u64 nonce = strtoull(pos.at(2).c_str(), 0, 10);
        Runner R(*J, gpu, fitL(1, cur), block, SP, F, fvar);  // one lane
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
        struct Cfg { u32 L; KSel k; };
        std::vector<Cfg> cfgs;
        u64 start, count;
        if (mode == "search") {
            start = strtoull(pos.at(2).c_str(), 0, 10); count = strtoull(pos.at(3).c_str(), 0, 10);
            cfgs.push_back({lanes, cur});
        } else {
            if (seconds <= 0) seconds = pos.size() > 2 ? atof(pos[2].c_str()) : 90;
            start = 1000000000000ULL; count = 1ULL << 40;
            if (gpu && !kernel_given) {  // the exact plain simulator (the old search), then the fast filter variants
                KSel pk{nullptr, -1, 128};
                cfgs.push_back({default_lanes(pk), pk});
                std::vector<int> vs = fvars;
                if (!fvars_given) { vs.clear(); for (int v = 0; v < NFVARS; v++) vs.push_back(v); }
                if (!fops.empty()) for (int v : vs) { KSel k{nullptr, v, FBS}; cfgs.push_back({lanes_given ? fitL(lanes, k) : default_lanes(k), k}); }
            } else if (gpu && SP && specs.size() > 1) {
                for (const Spec& sc : specs) {
                    const SlotProg* p = prog_for(sc);
                    KSel k{p, -1, sc.block ? sc.block : 128};
                    cfgs.push_back({lanes_given ? lanes : default_lanes(k), k});
                }
            } else if (gpu && F && fvars.size() > 1) {
                for (int v : fvars) { KSel k{nullptr, v, FBS}; cfgs.push_back({lanes_given ? fitL(lanes, k) : default_lanes(k), k}); }
            } else cfgs.push_back({lanes, cur});
        }
        double best_rate = 0; std::string best_name;
        for (const Cfg& cf : cfgs) {
            const u32 L = cf.L;
            std::vector<std::pair<double, u64>> trace;  // (elapsed, finished) after every round
            const bool fastc = cf.k.fv >= 0;
            const Job& JJ = fastc ? fjob : (filter ? fjob : job);
            const std::vector<FOp>* FF = fastc ? &fops : nullptr;
            const bool filtered = fastc || filter;  // a clean record only passed the classical filter: check it in full
            Runner R(JJ, gpu, L, cf.k.block, cf.k.p, FF, fastc ? cf.k.fv : 0);
            SearchStats st;
            size_t seen = 0;
            double last = 0;
            bool found = false;
            char w[240]; snprintf(w, sizeof w, "%s, %s, %u lanes", where.c_str(), kname(cf.k).c_str(), L);
            auto on = [&](const std::vector<Rec>& r, double el) {
                st.lane_batches += L;  // every lane holds a nonce and runs one batch per round
                for (; seen < r.size(); seen++) {
                    add_search(st, r[seen]);
                    if (!(r[seen].v[9] & 1)) continue;
                    const unsigned long long nonce = (unsigned long long)r[seen].v[0];
                    if (!filtered) { printf("CLEAN nonce %llu\n", nonce); fflush(stdout); found = true; continue; }
                    printf("CANDIDATE nonce %llu passed the classical filter\n", nonce); fflush(stdout);
                    if (mode != "search") continue;
                    // the full official test (outcomes, phase, Toffoli count) on the CPU, while nothing else runs
                    Runner V(job, false, 1, 128);
                    auto vr = V.run(true, false, r[seen].v[0], 1, quiet);
                    const u64* v = vr.at(0).v;
                    printf("full check of nonce %llu: clean=%s classical_failures=%llu phase_garbage_batches=%llu ancilla_garbage_batches=%llu "
                           "failing_batches=%llu avg_tof=%.3f\n", nonce, (v[9] & 1) ? "true" : "false", (unsigned long long)v[2],
                           (unsigned long long)v[3], (unsigned long long)v[4], (unsigned long long)v[1], (double)v[5] / std::max<u64>(v[6], 1));
                    if (v[9] & 1) { printf("CLEAN nonce %llu\n", nonce); found = true; }
                    fflush(stdout);
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
                   100 * R.tm.io / tt, st.lane_batches / std::max(secs, 1e-9),
                   (double)st.lane_batches * (FF ? FF->size() : JJ.ncops) / std::max(R.tm.sim, 1e-9));
            fflush(stdout);
            start += st.finished + L;
        }
        if (mode == "bench" && cfgs.size() > 1)
            printf("BEST %s: %.1f nonces/s; at lambda %.2f: %.0f hours on this device per clean draw (90%% within %.0f hours)\n", best_name.c_str(),
                   best_rate, proj_lambda, std::exp(proj_lambda) / best_rate / 3600, std::exp(proj_lambda) / best_rate / 3600 * std::log(10.0));
        return 0;
    }
    if (mode == "mine") {  // CPU research: mine wire constants on draws [a, a+n), test them on draws [b, b+n2)
        if (gpu) { fprintf(stderr, "mine runs on the CPU (--cpu)\n"); return 1; }
        u64 a = strtoull(pos.at(2).c_str(), 0, 10), n = strtoull(pos.at(3).c_str(), 0, 10);
        u64 b = strtoull(pos.at(4).c_str(), 0, 10), n2 = strtoull(pos.at(5).c_str(), 0, 10);
        u32 C = pos.size() > 6 ? (u32)atol(pos[6].c_str()) : 200;
        Runner::Mine m;
        m.C = C; m.NR = job.nq + job.nb;
        for (u32 k = 1; k <= C; k++) m.cps.push_back((u32)((u64)job.ncops * k / (C + 1)));
        m.Z.assign((size_t)C * m.NR, 0); m.O.assign((size_t)C * m.NR, 0);
        m.fhist.assign(C + 1, 0); m.phist.assign(C + 1, 0);
        for (int ph = 0; ph < 2; ph++) {
            u64 st = ph ? b : a, cnt = ph ? n2 : n;
            Runner R(job, false, (u32)std::min<u64>(lanes, cnt), block);
            R.d.seed_nonce = 0; R.d.early = 0;
            R.reset_lanes();
            R.set64(R.d.next, st); R.set64(R.d.end, st + cnt);
            double t0 = Runner::now();
            for (u32 rr = 0; rr < 141 * ((cnt + R.L - 1) / R.L); rr++) R.round_mine(m, ph == 1);
            fprintf(stderr, "%s phase: %llu draws in %.0f s\n", ph ? "eval" : "mine", (unsigned long long)cnt, Runner::now() - t0);
            if (!ph) {
                u64 zc = 0, oc = 0;
                for (size_t x = 0; x < m.Z.size(); x++) { zc += m.Z[x] == 0; oc += m.O[x] == 0; }
                printf("mined on %llu passing shots: %llu zero and %llu one constants over %u checkpoints x %u rows\n",
                       (unsigned long long)m.samples, (unsigned long long)zc, (unsigned long long)oc, C, m.NR);
            }
        }
        u64 F = 0, Pn = 0;
        for (u32 k = 0; k <= C; k++) { F += m.fhist[k]; Pn += m.phist[k]; }
        printf("eval: %llu failing shots, %llu passing shots\n", (unsigned long long)F, (unsigned long long)Pn);
        u64 cf = 0, cp = 0;
        for (u32 k = 0; k < C; k++) {
            cf += m.fhist[k]; cp += m.phist[k];
            if (m.fhist[k] || m.phist[k] || k % 20 == 19)
                printf("checkpoint %3u (op %8u, %5.1f%%): failing shots caught so far %5.1f%%, passing shots flagged %.5f%%\n", k, m.cps[k],
                       100.0 * m.cps[k] / job.ncops, 100.0 * cf / std::max<u64>(F, 1), 100.0 * cp / std::max<u64>(Pn, 1));
        }
        return 0;
    }
    if (mode == "census") {  // CPU research: per Toffoli (k-th CCX/CCZ of the stream), on how many shots of CRN draws it fires
        if (gpu) { fprintf(stderr, "census runs on the CPU (--cpu)\n"); return 1; }
        u64 start = strtoull(pos.at(2).c_str(), 0, 10), count = strtoull(pos.at(3).c_str(), 0, 10);
        const char* outp = pos.size() > 4 ? pos[4].c_str() : "census.txt";
        Runner R(job, false, (u32)std::min<u64>(lanes, count), block);
        R.d.seed_nonce = 0; R.d.early = 0;
        R.reset_lanes();
        R.set64(R.d.next, start); R.set64(R.d.end, start + count);
        std::vector<u64> tot(2 * (size_t)job.ncops, 0);
        u64 shots = 0;
        double t0 = Runner::now();
        for (u32 rr = 0; rr < 141 * ((count + R.L - 1) / R.L); rr++) R.round_census(tot, shots);
        FILE* o = fopen(outp, "w");
        if (!o) { fprintf(stderr, "cannot write %s\n", outp); return 1; }
        fprintf(o, "# census of %llu shots (CRN draws %llu..%llu); columns: toffoli ordinal, compiled op, kind, q1 q2 qt, flagged, shots fired, shots charged (condition true)\n",
                (unsigned long long)shots, (unsigned long long)start, (unsigned long long)(start + count - 1));
        u64 ord = 0, h[8] = {0}, charged = 0, never_charged = 0, le1_charged = 0;
        for (u32 i = 0; i < job.ncops; i++) {
            const u32 w0 = job.ops[i].w0, w1 = job.ops[i].w1, k = w0 & 0xFF;
            if (k != K_CCX && k != K_CCZ) continue;
            u64 f = tot[2 * (size_t)i], ex = tot[2 * (size_t)i + 1];
            charged += ex;
            if (f == 0) never_charged += ex;
            if (f <= 1) le1_charged += ex;
            int b = f == 0 ? 0 : f <= 1 ? 1 : f <= 10 ? 2 : f <= 100 ? 3 : f <= 1000 ? 4 : f * 100 <= shots ? 5 : f * 2 <= shots ? 6 : 7;
            h[b]++;
            fprintf(o, "%llu %u %s %u %u %u %d %llu %llu\n", (unsigned long long)ord, i, k == K_CCX ? "CCX" : "CCZ", w1 & 0xFFFF, w1 >> 16, w0 >> 16,
                    (w0 & 0xFF00) ? 1 : 0, (unsigned long long)f, (unsigned long long)ex);
            ord++;
        }
        fclose(o);
        const char* nm[8] = {"never", "1 shot", "2-10", "11-100", "101-1000", "<=1%", "<=50%", ">50%"};
        printf("census: %llu shots in %.0f s, %llu Toffoli ops; fired on:\n", (unsigned long long)shots, Runner::now() - t0, (unsigned long long)ord);
        for (int b = 0; b < 8; b++) printf("  %-9s %10llu ops\n", nm[b], (unsigned long long)h[b]);
        printf("Toffoli charged per shot: %.3f; of it on ops that never fired %.3f, fired on <= 1 shot %.3f\n", (double)charged / shots,
               (double)never_charged / shots, (double)le1_charged / shots);
        printf("per-op counts: %s\n", outp);
        return 0;
    }
    if (mode == "faildump") {  // CPU, plain simulator: every failing shot of CRN draws start..start+count
        if (gpu) { fprintf(stderr, "faildump runs on the CPU (--cpu)\n"); return 1; }
        u64 start = strtoull(pos.at(2).c_str(), 0, 10), count = strtoull(pos.at(3).c_str(), 0, 10);
        Runner R(*J, false, std::min<u64>(lanes, count), block, nullptr, F, fvar);
        R.d.dump = 1;
        auto r = R.run(false, false, start, count, quiet);
        print_crn(*J, r, 0, where.c_str());
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
            for (size_t i = 0; i < a.size(); i++) if (!same_rec(a[i], b[i], early, F != nullptr)) {
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
        { Runner R(*J, gpu, gpu ? lanes : cl, block, SP, F, fvar); dc = R.run(false, false, 0, ncrn, quiet); }
        printf("device CRN draws 0..%llu: %.1f s\n", (unsigned long long)ncrn - 1, Runner::now() - t0);
        t0 = Runner::now();
        { Runner R(*J, gpu, gpu ? lanes : cl, block, SP, F, fvar); ds = R.run(true, true, 1000, nsearch, quiet); }
        printf("device search nonces 1000..%llu: %.1f s\n", (unsigned long long)(1000 + nsearch - 1), Runner::now() - t0);
        print_crn(*J, dc, 0, where.c_str());
        if (gpu || SP || F) {  // against the plain simulator on the CPU (on the same stream)
            t0 = Runner::now();
            { Runner R(*J, false, cl, 128); cc = R.run(false, false, 0, ncrn, quiet); }
            { Runner R(*J, false, cl, 128); cs = R.run(true, true, 1000, nsearch, quiet); }
            printf("CPU reference: %.1f s\n", Runner::now() - t0);
            cmp("CRN draws", dc, cc, false);
            cmp("search nonces", ds, cs, true);
            printf(bad ? "SELFTEST FAILED\n" : "SELFTEST OK: %s (%s) matches the plain CPU simulator record for record\n",
                   gpu ? "the GPU" : "the CPU", kname(cur).c_str());
        }
        for (auto& r : ds) printf("nonce %llu passed_batches %llu clean %s\n", (unsigned long long)r.v[0], (unsigned long long)r.v[1], (r.v[9] & 1) ? "true" : "false");
        return bad ? 1 : 0;
    }
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
