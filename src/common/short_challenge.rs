//! Short ternary challenge sampler.
//!
//! Each challenge has a fixed weight and a canonical-embedding norm `||sigma(c)||_inf` bounded by
//! a rejection threshold, TAU and T_OP_NORM_BOUND by default. The embeddings are the evaluations
//! at the primitive `m`-th roots of unity, `m` the conductor, one per conjugate pair. For a sparse
//! `c = sum_k eps_k * X^{p_k}`,
//!
//! ```text
//! c(zeta_m^j) = sum_k eps_k * T[(j * p_k) mod m],   gcd(j, m) = 1,  j < m/2,
//! ```
//!
//! with `T[t] = exp(2 i pi t / m)`. The hot path uses this directly via a length-`m` phase LUT
//! instead of a full FFT.
//!
//! - For `Z_q[X]/(X^N + 1)`, `m = 2N` and `j = 2j' + 1`. The power basis is orthogonal for the
//!   canonical embedding, so `||sigma(c)||_inf` is exactly the coefficient-embedding operator
//!   norm `||c||_op`.
//! - For `Z_p[X]/(X^{2n} - X^n + 1)`, `X^{2n} - X^n + 1 = Phi_m(X)` with `m = 9 * 2^k = 3 phi`,
//!   and `j = 1, 5 mod 6`. Here `||c||_op <= sqrt(3) ||sigma(c)||_inf` only (the note, Lemma 3.8),
//!   so a challenge accepted at T_OP_NORM_BOUND has operator norm at most [`GAMMA_C`].

use crate::common::config::{RingFamily, DEGREE, MOD_Q, RING_FAMILY};
use crate::common::hash::HashWrapper;
use crate::common::ring_arithmetic::{Representation, RingElement};
use num::Complex;
use std::sync::LazyLock;

include!(concat!(env!("OUT_DIR"), "/challenge.rs"));

/// Bound on the coefficient-embedding operator norm `gamma_C = max ||c||_op` of every challenge
/// [`sample_short_challenge`] accepts. The sampler bounds `||sigma(c)||_inf` by
/// T_OP_NORM_BOUND; for `X^N + 1` the two norms coincide, and for `X^{2n} - X^n + 1` the note's
/// Lemma 3.8 gives `||c||_op <= sqrt(3) ||sigma(c)||_inf`.
pub const GAMMA_C: f64 = match RING_FAMILY {
    RingFamily::PowerOfTwo => T_OP_NORM_BOUND,
    RingFamily::Trinomial => SQRT_3 * T_OP_NORM_BOUND,
};
const SQRT_3: f64 = 1.7320508075688772;

const N: usize = DEGREE;
const LOG_N: usize = N.trailing_zeros() as usize;
/// The conductor `m`: `2N` for `X^N + 1`, `9 * 2^k = 3N` for `X^{2n} - X^n + 1`.
const CONDUCTOR: usize = match RING_FAMILY {
    RingFamily::PowerOfTwo => 2 * N,
    RingFamily::Trinomial => 3 * N,
};
const PHASE_LEN: usize = CONDUCTOR;
/// Only meaningful for `X^N + 1`, where `PHASE_LEN` is a power of two.
const PHASE_MASK: usize = PHASE_LEN - 1;

// The trinomial path steps through `j = 6t + 1` and `j = 6t + 5`, `t < N/4`.
const _: () = assert!(!matches!(RING_FAMILY, RingFamily::Trinomial) || N % 12 == 0);

/// The exponents `j < m/2` with `gcd(j, m) = 1`, one per conjugate pair `(j, m - j)`, in
/// increasing order: `2j' + 1` for `X^N + 1`, `j = 1, 5 mod 6` for `X^{2n} - X^n + 1`.
pub static EVAL_EXPONENTS: LazyLock<[usize; N / 2]> = LazyLock::new(|| {
    let exponents: Vec<usize> = (1..CONDUCTOR / 2)
        .filter(|&j| gcd(j, CONDUCTOR) == 1)
        .collect();
    exponents.try_into().expect("phi(m)/2 exponents below m/2")
});

const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

static PRE_TWIST: LazyLock<[Complex<f64>; N]> = LazyLock::new(|| {
    let mut arr = [Complex::new(0.0, 0.0); N];
    for j in 0..N {
        let angle = std::f64::consts::PI * (j as f64) / (N as f64);
        arr[j] = Complex::new(angle.cos(), angle.sin());
    }
    arr
});

/// The angle of `T[t] = exp(2 i pi t / m)`. The `X^N + 1` form is kept as it was, `pi t / N`.
fn phase_angle(t: usize) -> f64 {
    match RING_FAMILY {
        RingFamily::PowerOfTwo => std::f64::consts::PI * (t as f64) / (N as f64),
        RingFamily::Trinomial => 2.0 * std::f64::consts::PI * (t as f64) / (CONDUCTOR as f64),
    }
}

static PHASE_RE: LazyLock<[f64; PHASE_LEN]> = LazyLock::new(|| {
    let mut arr = [0.0f64; PHASE_LEN];
    for m in 0..PHASE_LEN {
        arr[m] = phase_angle(m).cos();
    }
    arr
});
static PHASE_IM: LazyLock<[f64; PHASE_LEN]> = LazyLock::new(|| {
    let mut arr = [0.0f64; PHASE_LEN];
    for m in 0..PHASE_LEN {
        arr[m] = phase_angle(m).sin();
    }
    arr
});

static BIT_REVERSE: LazyLock<[usize; N]> = LazyLock::new(|| {
    let mut arr = [0usize; N];
    for i in 0..N {
        let mut x = i;
        let mut r = 0usize;
        for _ in 0..LOG_N {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        arr[i] = r;
    }
    arr
});

fn fft_in_place(a: &mut [Complex<f64>; N]) {
    let br = &*BIT_REVERSE;
    for i in 0..N {
        let j = br[i];
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= N {
        let half = len / 2;
        let theta = 2.0 * std::f64::consts::PI / (len as f64);
        let w_step = Complex::new(theta.cos(), theta.sin());
        let mut start = 0usize;
        while start < N {
            let mut w = Complex::new(1.0_f64, 0.0_f64);
            for j in 0..half {
                let t = w * a[start + half + j];
                let u = a[start + j];
                a[start + j] = u + t;
                a[start + half + j] = u - t;
                w *= w_step;
            }
            start += len;
        }
        len <<= 1;
    }
}

/// `||sigma(c)||_inf` by a dense evaluation: the twisted FFT for `X^N + 1`, and a sum over the
/// phase table at every exponent of [`EVAL_EXPONENTS`] for `X^{2n} - X^n + 1`.
pub fn op_norm(c: &[i8; N]) -> f64 {
    if RING_FAMILY == RingFamily::Trinomial {
        return op_norm_dense_trinomial(c);
    }
    let pt = &*PRE_TWIST;
    let mut a = [Complex::new(0.0, 0.0); N];
    for j in 0..N {
        if c[j] != 0 {
            let s = c[j] as f64;
            a[j] = pt[j] * s;
        }
    }
    fft_in_place(&mut a);
    let mut max_sq = 0.0f64;
    for k in 0..(N / 2) {
        let m = a[k].norm_sqr();
        if m > max_sq {
            max_sq = m;
        }
    }
    max_sq.sqrt()
}

fn op_norm_dense_trinomial(c: &[i8; N]) -> f64 {
    let mut max_sq = 0.0f64;
    for &j in EVAL_EXPONENTS.iter() {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for k in 0..N {
            if c[k] != 0 {
                let t = (j * k) % CONDUCTOR;
                re += c[k] as f64 * PHASE_RE[t];
                im += c[k] as f64 * PHASE_IM[t];
            }
        }
        max_sq = max_sq.max(re * re + im * im);
    }
    max_sq.sqrt()
}

#[inline(always)]
pub fn op_norm_sq_sparse<const W: usize>(positions: &[u8; W], signs: &[i8; W]) -> f64 {
    let in_range = positions.iter().all(|&p| (p as usize) < N);
    assert!(in_range, "position out of range");
    if RING_FAMILY == RingFamily::Trinomial {
        return op_norm_sq_sparse_trinomial(positions, signs);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        return op_norm_sq_sparse_avx512(positions, signs);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    op_norm_sq_sparse_scalar(positions, signs)
}

pub fn op_norm_sq_sparse_scalar<const W: usize>(positions: &[u8; W], signs: &[i8; W]) -> f64 {
    if RING_FAMILY == RingFamily::Trinomial {
        return op_norm_sq_sparse_trinomial(positions, signs);
    }
    let phase_re: &[f64; PHASE_LEN] = &PHASE_RE;
    let phase_im: &[f64; PHASE_LEN] = &PHASE_IM;

    let mut v_re = [0.0f64; N / 2];
    let mut v_im = [0.0f64; N / 2];

    for k in 0..W {
        let p = positions[k] as usize;
        let s = signs[k] as f64;
        let step = (2 * p) & PHASE_MASK;
        let mut idx = p & PHASE_MASK;
        for j in 0..(N / 2) {
            let pr = unsafe { *phase_re.get_unchecked(idx) };
            let pi = unsafe { *phase_im.get_unchecked(idx) };
            v_re[j] += s * pr;
            v_im[j] += s * pi;
            idx = (idx + step) & PHASE_MASK;
        }
    }

    let mut max_sq = 0.0f64;
    for j in 0..(N / 2) {
        let m = v_re[j] * v_re[j] + v_im[j] * v_im[j];
        if m > max_sq {
            max_sq = m;
        }
    }
    max_sq
}

/// The sparse sum for `X^{2n} - X^n + 1`. The exponents prime to `m = 9 * 2^k` below `m/2` are
/// `6t + 1` and `6t + 5` for `t < m/12 = N/4`, so a nonzero at position `p` steps through the
/// phases `p (6t + 1)` and `p (6t + 5)` by `6p`, modulo `m`. Slot `2t` holds `j = 6t + 1` and slot
/// `2t + 1` holds `j = 6t + 5`, the order of [`EVAL_EXPONENTS`].
fn op_norm_sq_sparse_trinomial<const W: usize>(positions: &[u8; W], signs: &[i8; W]) -> f64 {
    let phase_re: &[f64; PHASE_LEN] = &PHASE_RE;
    let phase_im: &[f64; PHASE_LEN] = &PHASE_IM;

    let mut v_re = [0.0f64; N / 2];
    let mut v_im = [0.0f64; N / 2];

    for k in 0..W {
        let p = positions[k] as usize;
        let s = signs[k] as f64;
        let step = (6 * p) % CONDUCTOR;
        let mut idx_1 = p % CONDUCTOR;
        let mut idx_5 = (5 * p) % CONDUCTOR;
        for t in 0..(N / 4) {
            v_re[2 * t] += s * phase_re[idx_1];
            v_im[2 * t] += s * phase_im[idx_1];
            v_re[2 * t + 1] += s * phase_re[idx_5];
            v_im[2 * t + 1] += s * phase_im[idx_5];
            idx_1 += step;
            if idx_1 >= CONDUCTOR {
                idx_1 -= CONDUCTOR;
            }
            idx_5 += step;
            if idx_5 >= CONDUCTOR {
                idx_5 -= CONDUCTOR;
            }
        }
    }

    let mut max_sq = 0.0f64;
    for j in 0..(N / 2) {
        let m = v_re[j] * v_re[j] + v_im[j] * v_im[j];
        if m > max_sq {
            max_sq = m;
        }
    }
    max_sq
}

/// Row `p` holds `T[p (2j + 1) mod 2N]` for `j < N/2`, real parts then imaginary parts: the
/// phases [`op_norm_sq_sparse_scalar`] steps through for a nonzero at position `p`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
static PHASE_ROWS: LazyLock<Vec<f64>> = LazyLock::new(|| {
    let mut rows = vec![0.0f64; N * N];
    for (p, row) in rows.chunks_exact_mut(N).enumerate() {
        for j in 0..N / 2 {
            let idx = (p * (2 * j + 1)) & PHASE_MASK;
            row[j] = PHASE_RE[idx];
            row[N / 2 + j] = PHASE_IM[idx];
        }
    }
    rows
});

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn op_norm_sq_sparse_avx512<const W: usize>(positions: &[u8; W], signs: &[i8; W]) -> f64 {
    use std::arch::x86_64::*;

    const NUM_BATCHES: usize = N / 16;
    const _: () = assert!(N % 16 == 0 && NUM_BATCHES * 2 <= 32);

    let rows = PHASE_ROWS.as_ptr();

    let mut vr = [_mm512_setzero_pd(); NUM_BATCHES];
    let mut vi = [_mm512_setzero_pd(); NUM_BATCHES];

    for k in 0..W {
        let s_v = _mm512_set1_pd(signs[k] as f64);
        let row = rows.add(N * positions[k] as usize);
        for b in 0..NUM_BATCHES {
            let pre = _mm512_loadu_pd(row.add(8 * b));
            let pim = _mm512_loadu_pd(row.add(N / 2 + 8 * b));
            vr[b] = _mm512_fmadd_pd(s_v, pre, vr[b]);
            vi[b] = _mm512_fmadd_pd(s_v, pim, vi[b]);
        }
    }

    let mut mm = _mm512_fmadd_pd(vr[0], vr[0], _mm512_mul_pd(vi[0], vi[0]));
    for b in 1..NUM_BATCHES {
        let m = _mm512_fmadd_pd(vr[b], vr[b], _mm512_mul_pd(vi[b], vi[b]));
        mm = _mm512_max_pd(mm, m);
    }
    _mm512_reduce_max_pd(mm)
}

/// `||sigma(c)||_inf` from the trigonometric functions, at every embedding (both members of each
/// conjugate pair): the reference for [`op_norm`] and [`op_norm_sq_sparse`].
pub fn op_norm_direct(c: &[i8; N]) -> f64 {
    if RING_FAMILY == RingFamily::Trinomial {
        return op_norm_direct_trinomial(c);
    }
    let mut max_sq = 0.0f64;
    let two_n = 2 * N;
    for j in (1..two_n).step_by(2) {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for k in 0..N {
            if c[k] == 0 {
                continue;
            }
            let angle = std::f64::consts::PI * (j as f64) * (k as f64) / (N as f64);
            let s = c[k] as f64;
            re += s * angle.cos();
            im += s * angle.sin();
        }
        let m = re * re + im * im;
        if m > max_sq {
            max_sq = m;
        }
    }
    max_sq.sqrt()
}

/// Every `j < m` prime to `m = 9 * 2^k`, that is odd and not a multiple of 3.
fn op_norm_direct_trinomial(c: &[i8; N]) -> f64 {
    let mut max_sq = 0.0f64;
    for j in (1..CONDUCTOR).filter(|j| j % 2 != 0 && j % 3 != 0) {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for k in 0..N {
            if c[k] == 0 {
                continue;
            }
            let angle =
                2.0 * std::f64::consts::PI * (((j * k) % CONDUCTOR) as f64) / (CONDUCTOR as f64);
            re += c[k] as f64 * angle.cos();
            im += c[k] as f64 * angle.sin();
        }
        max_sq = max_sq.max(re * re + im * im);
    }
    max_sq.sqrt()
}

pub const ATTEMPT_LABEL: &[u8] = b"short-ternary-challenge-attempt";
const ATTEMPT_BUF_LEN: usize = 128;

const _: () = assert!(N <= 256, "Fisher-Yates uses one byte per index sample");

#[allow(unused_assignments)]
fn sample_attempt<const W: usize>(hasher: &mut HashWrapper, label: &[u8]) -> ([u8; W], [i8; W]) {
    let mut buf = [0u8; ATTEMPT_BUF_LEN];
    hasher.fill_from_xof(label, &mut buf);
    let mut idx: usize = 0;

    macro_rules! next_byte {
        () => {{
            if idx >= ATTEMPT_BUF_LEN {
                hasher.fill_from_xof(label, &mut buf);
                idx = 0;
            }
            let b = buf[idx];
            idx += 1;
            b
        }};
    }

    let mut perm: [u8; N] = std::array::from_fn(|i| i as u8);
    for i in 0..W {
        let range = (N - i) as u32;
        let cutoff: u32 = 256 - (256 % range);
        let r = loop {
            let b = next_byte!() as u32;
            if b < cutoff {
                break b;
            }
        };
        let j = i + (r % range) as usize;
        perm.swap(i, j);
    }

    let mut positions = [0u8; W];
    let mut signs = [0i8; W];
    let mut sign_byte = 0u8;
    for i in 0..W {
        if i % 8 == 0 {
            sign_byte = next_byte!();
        }
        positions[i] = perm[i];
        let bit = (sign_byte >> (i % 8)) & 1;
        signs[i] = if bit == 1 { 1 } else { -1 };
    }
    (positions, signs)
}

pub fn sample_fixed_weight_challenge<const W: usize>(
    hasher: &mut HashWrapper,
    op_norm_bound: f64,
    label: &[u8],
) -> ([i8; N], usize) {
    assert!(0 < W && W <= N);
    let bound_sq = op_norm_bound * op_norm_bound;
    let mut attempts: usize = 0;
    loop {
        attempts += 1;
        let (positions, signs) = sample_attempt::<W>(hasher, label);
        if op_norm_sq_sparse(&positions, &signs) <= bound_sq {
            let mut c = [0i8; N];
            for i in 0..W {
                c[positions[i] as usize] = signs[i];
            }
            return (c, attempts);
        }
    }
}

pub fn sample_fixed_weight_challenge_into<const W: usize>(
    hasher: &mut HashWrapper,
    op_norm_bound: f64,
    label: &[u8],
    output: &mut RingElement,
) -> usize {
    let (c, attempts) = sample_fixed_weight_challenge::<W>(hasher, op_norm_bound, label);
    output.representation = Representation::Coefficients;
    for j in 0..N {
        output.v[j] = match c[j] {
            -1 => MOD_Q - 1,
            0 => 0,
            1 => 1,
            _ => unreachable!(),
        };
    }
    output.to_representation(Representation::IncompleteNTT);
    attempts
}

pub fn sample_short_challenge(hasher: &mut HashWrapper) -> ([i8; N], usize) {
    sample_fixed_weight_challenge::<TAU>(hasher, T_OP_NORM_BOUND, ATTEMPT_LABEL)
}

pub fn sample_short_challenge_into(hasher: &mut HashWrapper, output: &mut RingElement) -> usize {
    sample_fixed_weight_challenge_into::<TAU>(hasher, T_OP_NORM_BOUND, ATTEMPT_LABEL, output)
}

pub fn repetition_rate() -> f64 {
    let mut hasher = HashWrapper::new();
    let mut total_attempts: usize = 0;
    let trials = 10_000;
    for _ in 0..trials {
        let (_, attempts) = sample_short_challenge(&mut hasher);
        total_attempts += attempts;
    }
    (total_attempts as f64) / (trials as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_op_norm_rejects_positions_past_the_degree() {
        if N > u8::MAX as usize {
            return;
        }
        let mut positions = [0u8; TAU];
        positions[TAU - 1] = N as u8;
        let run = || op_norm_sq_sparse(&positions, &[1i8; TAU]);
        assert!(std::panic::catch_unwind(run).is_err());
    }

    fn weight(c: &[i8; N]) -> usize {
        c.iter().filter(|&&x| x != 0).count()
    }

    fn fixed_test_vector() -> ([u8; TAU], [i8; TAU]) {
        let mut positions = [0u8; TAU];
        let mut signs = [0i8; TAU];
        for i in 0..TAU {
            positions[i] = ((i * 13 + 1) % N) as u8;
            signs[i] = if i % 3 == 0 { -1 } else { 1 };
        }
        (positions, signs)
    }

    #[test]
    fn fft_op_norm_matches_direct_on_fixed_vector() {
        let (positions, signs) = fixed_test_vector();
        let mut c = [0i8; N];
        for i in 0..TAU {
            c[positions[i] as usize] = signs[i];
        }
        let fft = op_norm(&c);
        let direct = op_norm_direct(&c);
        assert!(
            (fft - direct).abs() < 1e-10,
            "fft={} direct={}",
            fft,
            direct
        );
    }

    #[test]
    fn sparse_op_norm_matches_dense_fft() {
        let mut hasher = HashWrapper::new();
        let top: [u8; TAU] = std::array::from_fn(|i| (N - 1 - i) as u8);
        for attempt in 0..2_000 {
            let (positions, signs) = match attempt {
                0 => fixed_test_vector(),
                1 => (top, [1i8; TAU]),
                2 => (top, [-1i8; TAU]),
                _ => sample_attempt::<TAU>(&mut hasher, b"sparse-vs-dense"),
            };
            let mut c = [0i8; N];
            for i in 0..TAU {
                c[positions[i] as usize] += signs[i];
            }
            let dense = op_norm(&c);
            let sparse = op_norm_sq_sparse(&positions, &signs).sqrt();
            assert!(
                (dense - sparse).abs() < 1e-10,
                "dense={dense} sparse={sparse}"
            );
        }
    }

    #[test]
    fn samples_have_correct_weight_and_op_norm() {
        let mut hasher = HashWrapper::new();
        for _ in 0..10_000 {
            let (c, _) = sample_short_challenge(&mut hasher);
            assert_eq!(weight(&c), TAU);
            for &x in &c {
                assert!(x == -1 || x == 0 || x == 1);
            }
            let n = op_norm(&c);
            assert!(n <= T_OP_NORM_BOUND, "op_norm {} > {}", n, T_OP_NORM_BOUND);
        }
    }

    #[test]
    fn parametrized_samples_have_correct_weight_and_op_norm() {
        const W: usize = 23;
        const BOUND: f64 = 8.357;
        let label = b"parametrized-challenge-test";
        let mut hasher = HashWrapper::new();
        let mut total_attempts = 0usize;
        for _ in 0..1_000 {
            let (c, attempts) = sample_fixed_weight_challenge::<W>(&mut hasher, BOUND, label);
            total_attempts += attempts;
            assert_eq!(weight(&c), W);
            for &x in &c {
                assert!(x == -1 || x == 0 || x == 1);
            }
            let n = op_norm(&c);
            assert!(n <= BOUND, "op_norm {} > {}", n, BOUND);
        }
        assert!(total_attempts > 1_000);
    }

    #[test]
    fn parametrized_sampling_is_deterministic_and_label_separated() {
        const W: usize = 23;
        const BOUND: f64 = 8.357;
        let mut h1 = HashWrapper::new();
        let mut h2 = HashWrapper::new();
        let mut h3 = HashWrapper::new();
        for _ in 0..20 {
            let (c1, _) = sample_fixed_weight_challenge::<W>(&mut h1, BOUND, b"label-a");
            let (c2, _) = sample_fixed_weight_challenge::<W>(&mut h2, BOUND, b"label-a");
            let (c3, _) = sample_fixed_weight_challenge::<W>(&mut h3, BOUND, b"label-b");
            assert_eq!(c1, c2);
            assert_ne!(c1, c3);
        }
        assert_eq!(h1.sample_bytes(64), h2.sample_bytes(64));
    }

    #[test]
    fn full_weight_bound_accepts_first_attempt() {
        const W: usize = 21;
        let mut hasher = HashWrapper::new();
        for _ in 0..100 {
            let (c, attempts) = sample_fixed_weight_challenge::<W>(&mut hasher, W as f64, b"t");
            assert_eq!(attempts, 1);
            assert_eq!(weight(&c), W);
        }
    }

    #[test]
    fn sampling_is_deterministic_and_post_state_matches() {
        let mut h1 = HashWrapper::new();
        let mut h2 = HashWrapper::new();
        for _ in 0..20 {
            let (c1, _) = sample_short_challenge(&mut h1);
            let (c2, _) = sample_short_challenge(&mut h2);
            assert_eq!(c1, c2);
        }
        assert_eq!(h1.sample_bytes(64), h2.sample_bytes(64));
    }

    #[test]
    fn challenge_stream_fingerprint() {
        let mut hasher = HashWrapper::new();
        let mut digest = blake3::Hasher::new();
        for _ in 0..100 {
            let (c, _) = sample_short_challenge(&mut hasher);
            let bytes: [u8; N] = std::array::from_fn(|i| c[i] as u8);
            digest.update(&bytes);
        }
        let actual = digest.finalize().to_hex().to_string();
        assert_eq!(
            actual, EXPECTED_FINGERPRINT,
            "challenge stream changed; update EXPECTED_FINGERPRINT if intentional"
        );
    }

    const EXPECTED_FINGERPRINT: &str =
        "5670da5c3578390a8e449b8d42526135dcdb708c95054a31c5767281218c42b1";

    /// `c(zeta_m^j)` from the trigonometric functions, for integer coefficients.
    fn evaluate(c: &[i64], j: usize) -> Complex<f64> {
        let mut z = Complex::new(0.0, 0.0);
        for (k, &x) in c.iter().enumerate() {
            let angle =
                2.0 * std::f64::consts::PI * (((j * k) % CONDUCTOR) as f64) / (CONDUCTOR as f64);
            z += Complex::new(angle.cos(), angle.sin()) * x as f64;
        }
        z
    }

    /// The ring's modulus `X^N + 1` or `X^{2n} - X^n + 1` at `zeta_m^j`.
    fn modulus_at(j: usize) -> Complex<f64> {
        let z = |e: usize| {
            let angle =
                2.0 * std::f64::consts::PI * (((j * e) % CONDUCTOR) as f64) / (CONDUCTOR as f64);
            Complex::new(angle.cos(), angle.sin())
        };
        match RING_FAMILY {
            RingFamily::PowerOfTwo => z(N) + 1.0,
            RingFamily::Trinomial => z(N) - z(N / 2) + 1.0,
        }
    }

    fn centered(x: u64) -> i64 {
        if x > MOD_Q / 2 {
            -((MOD_Q - x) as i64)
        } else {
            x as i64
        }
    }

    fn ternary_element(c: &[i8; N]) -> RingElement {
        let mut element = RingElement::new(Representation::Coefficients);
        for j in 0..N {
            element.v[j] = if c[j] < 0 { MOD_Q - 1 } else { c[j] as u64 };
        }
        element.to_representation(Representation::IncompleteNTT);
        element
    }

    fn coefficients(element: &RingElement) -> Vec<i64> {
        let mut element = element.clone();
        element.to_representation(Representation::Coefficients);
        element.v.iter().map(|&x| centered(x)).collect()
    }

    /// The evaluation points are the primitive `m`-th roots of unity, one per conjugate pair:
    /// the modulus vanishes at all `phi` of them, so (being monic of degree `phi`) it is
    /// `Phi_m`, and its roots are exactly these. It does not vanish at the other `m`-th roots.
    #[test]
    fn evaluation_points_are_the_roots_of_the_modulus() {
        let exponents = &*EVAL_EXPONENTS;
        assert!(exponents.windows(2).all(|w| w[0] < w[1]));
        assert!(exponents.iter().all(|&j| j < CONDUCTOR / 2));
        let mut roots = 0;
        for j in 1..CONDUCTOR {
            let value = modulus_at(j).norm();
            let paired = exponents.contains(&j) || exponents.contains(&(CONDUCTOR - j));
            if paired {
                assert!(value < 1e-10, "modulus at zeta^{j} = {value}");
                roots += 1;
            } else {
                assert!(
                    value > 1e-3,
                    "zeta^{j} is a root but not an evaluation point"
                );
            }
        }
        assert_eq!(roots, N);
    }

    /// The embedding is a ring homomorphism: `(ab)(zeta^j) = a(zeta^j) b(zeta^j)` for the ring's
    /// own product, at every evaluation point.
    #[test]
    fn canonical_embedding_is_multiplicative() {
        let mut hasher = HashWrapper::new();
        for _ in 0..8 {
            let (a, _) = sample_fixed_weight_challenge::<TAU>(&mut hasher, f64::INFINITY, b"a");
            let (b, _) = sample_fixed_weight_challenge::<TAU>(&mut hasher, f64::INFINITY, b"b");
            let product = coefficients(&(&ternary_element(&a) * &ternary_element(&b)));
            let a: Vec<i64> = a.iter().map(|&x| x as i64).collect();
            let b: Vec<i64> = b.iter().map(|&x| x as i64).collect();
            for &j in EVAL_EXPONENTS.iter() {
                let expected = evaluate(&a, j) * evaluate(&b, j);
                let diff = (evaluate(&product, j) - expected).norm();
                assert!(diff < 1e-9, "j = {j}: |difference| = {diff}");
            }
        }
    }

    /// The largest singular value of the multiplication matrix of `c`, by power iteration on
    /// `M^T M`. It converges from below, so it never overstates the operator norm.
    fn coefficient_operator_norm(c: &[i8; N]) -> f64 {
        let element = ternary_element(c);
        // Column `i` is `c X^i`.
        let mut matrix = vec![0.0f64; N * N];
        for i in 0..N {
            let mut monomial = RingElement::new(Representation::Coefficients);
            monomial.v[i] = 1;
            monomial.to_representation(Representation::IncompleteNTT);
            for (r, x) in coefficients(&(&element * &monomial))
                .into_iter()
                .enumerate()
            {
                matrix[r * N + i] = x as f64;
            }
        }
        let mut x: Vec<f64> = (0..N).map(|i| 1.0 + (i % 7) as f64).collect();
        let mut norm = 0.0;
        for _ in 0..100 {
            let length = x.iter().map(|a| a * a).sum::<f64>().sqrt();
            x.iter_mut().for_each(|a| *a /= length);
            let y: Vec<f64> = (0..N)
                .map(|r| (0..N).map(|i| matrix[r * N + i] * x[i]).sum())
                .collect();
            norm = y.iter().map(|a| a * a).sum::<f64>().sqrt();
            x = (0..N)
                .map(|i| (0..N).map(|r| matrix[r * N + i] * y[r]).sum())
                .collect();
        }
        norm
    }

    /// Accepted challenges have coefficient operator norm at most `GAMMA_C`. For the trinomial
    /// family it exceeds `||sigma(c)||_inf`, and so T_OP_NORM_BOUND, on some samples: the factor
    /// `sqrt(3)` of Lemma 3.8 is needed. For `X^N + 1` the two norms agree.
    #[test]
    fn challenges_have_operator_norm_at_most_gamma_c() {
        let mut hasher = HashWrapper::new();
        let mut largest_ratio = 0.0f64;
        for _ in 0..16 {
            let (c, _) = sample_short_challenge(&mut hasher);
            let operator = coefficient_operator_norm(&c);
            let sigma = op_norm(&c);
            assert!(
                operator <= GAMMA_C + 1e-9,
                "||c||_op = {operator} > {GAMMA_C}"
            );
            assert!(operator <= sigma * SQRT_3 + 1e-9);
            largest_ratio = largest_ratio.max(operator / sigma);
        }
        match RING_FAMILY {
            RingFamily::PowerOfTwo => {
                assert_eq!(GAMMA_C, T_OP_NORM_BOUND);
                assert!(largest_ratio <= 1.0 + 1e-9);
            }
            RingFamily::Trinomial => {
                assert_eq!(GAMMA_C, 3f64.sqrt() * T_OP_NORM_BOUND);
                assert!(
                    largest_ratio > 1.1,
                    "largest ||c||_op / ||sigma(c)|| = {largest_ratio}"
                );
            }
        }
    }

    #[test]
    fn ring_output_encoding_is_consistent() {
        let mut hasher = HashWrapper::new();
        let mut elem = RingElement::new(Representation::Coefficients);
        let _ = sample_short_challenge_into(&mut hasher, &mut elem);
        assert_eq!(elem.representation, Representation::IncompleteNTT);
        elem.to_representation(Representation::Coefficients);
        let mut nonzero = 0;
        for &v in &elem.v {
            assert!(v == 0 || v == 1 || v == MOD_Q - 1);
            if v != 0 {
                nonzero += 1;
            }
        }
        assert_eq!(nonzero, TAU);
    }
}
