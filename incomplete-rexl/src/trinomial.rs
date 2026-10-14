//! The slot transform of `Z_p[X]/(X^{2n} - X^n + 1)` over the Goldilocks prime `p`, the
//! cyclotomic ring of conductor `9 * 2^k` and degree `3 * 2^k`, `n = 3 * 2^{k-1}`.
//!
//! With `Y = X^3`, `X^{2n} - X^n + 1 = Phi(Y) = Y^L - Y^h + 1`, `L = 2^k`, `h = L/2`, and an element
//! is `x_0(Y) + X x_1(Y) + X^2 x_2(Y)`: part `t` of the strided layout holds
//! `x_t = (c_t, c_{t+3}, c_{t+6}, ...)`. `Phi(Y) = prod_j (Y - eta_j)` over the `L` primitive
//! `3L`-th roots of unity, which all lie in `F_p`, so the forward transform evaluates each part at
//! every `eta_j`, and slot `j` is `(x_0(eta_j), x_1(eta_j), x_2(eta_j)) = x mod (X^3 - eta_j)`.
//!
//! The transform of one part is the NTTRU construction: with `w` a primitive sixth root of unity,
//! `Phi(Y) = (Y^h - w)(Y^h - w^5)` and `w^5 = 1 - w`, so one layer `a_lo + w a_hi`,
//! `a_lo + (1 - w) a_hi` (one product per pair) splits it, and every factor `Y^m - z` then splits
//! into `Y^{m/2} -+ sqrt(z)` by Cooley-Tukey butterflies, all square roots existing because
//! `3 * 2^32` divides `p - 1`. No radix-3 layer is needed.
//!
//! Every root is a power of `theta`, a primitive `3L`-th root of unity, and the transform tracks
//! exponents: slot `j` has `eta_j = theta^{e_j}`, with `e_0 = 1`. Inputs may be any `u64`, as
//! [`goldilocks::add`], [`goldilocks::sub`] and [`goldilocks::mul`] reduce them; outputs are
//! canonical.

use crate::goldilocks::{self, add, mul, sub, P};

/// A multiplicative generator of `F_p^*`.
const GENERATOR: u64 = 7;

/// Largest supported number of slots `L = 2^k`.
const MAX_SLOTS: usize = 128;

/// Product modulo `X^3 - eta` of `a` and `b` by Karatsuba: 6 products, and 2 more by `eta`.
#[inline(always)]
pub fn cubic_mul(a: [u64; 3], b: [u64; 3], eta: u64) -> [u64; 3] {
    let [l0, l1, l2, m01, m02, m12] = cubic_mul_terms(a, b);
    [add(l0, mul(eta, m12)), add(m01, mul(eta, l2)), add(m02, l1)]
}

/// The terms of [`cubic_mul`] before the products by `eta`: `[c0, c1, c2, c0', c1']` with the
/// product `(c0 + eta c0', c1 + eta c1', c2)`, so a sum of products needs `eta` once.
#[inline(always)]
fn cubic_mul_lazy(a: [u64; 3], b: [u64; 3]) -> [u64; 5] {
    let [l0, l1, l2, m01, m02, m12] = cubic_mul_terms(a, b);
    [l0, m01, add(m02, l1), m12, l2]
}

/// `[a0 b0, a1 b1, a2 b2, a0 b1 + a1 b0, a0 b2 + a2 b0, a1 b2 + a2 b1]` from six products, the
/// last three as `(ai + aj)(bi + bj) - ai bi - aj bj`.
#[inline(always)]
fn cubic_mul_terms(a: [u64; 3], b: [u64; 3]) -> [u64; 6] {
    let (p0, p1, p2) = (mul(a[0], b[0]), mul(a[1], b[1]), mul(a[2], b[2]));
    let p01 = mul(add(a[0], a[1]), add(b[0], b[1]));
    let p02 = mul(add(a[0], a[2]), add(b[0], b[2]));
    let p12 = mul(add(a[1], a[2]), add(b[1], b[2]));
    [
        p0,
        p1,
        p2,
        sub(p01, add(p0, p1)),
        sub(p02, add(p0, p2)),
        sub(p12, add(p1, p2)),
    ]
}

/// The slot transform and slot product of `Z_p[X]/(X^{2n} - X^n + 1)`, the trinomial
/// counterpart of [`crate::SlotRing`] with the same layout and entry points.
pub struct TrinomialSlotRing {
    ring_degree: usize,
    /// `L = 2^k`, the number of slots and the length of each part.
    slots: usize,
    /// `w = theta^h`, a primitive sixth root of unity.
    omega: u64,
    /// One root per Cooley-Tukey butterfly block, in the forward order.
    forward: Vec<u64>,
    /// Their inverses, in the inverse order.
    inverse: Vec<u64>,
    /// `h^-1` and `h^-1 (w - w^5)^-1`, the scales of the last inverse layer.
    inv_scale: [u64; 2],
    /// `eta_j`, slot by slot.
    roots: Vec<u64>,
    /// `e_j`, `eta_j = theta^{e_j}`.
    root_exponents: Vec<usize>,
}

impl TrinomialSlotRing {
    /// The ring of degree `ring_degree = 3 * 2^k` over `modulus`, which must be Goldilocks.
    pub fn new(ring_degree: usize, modulus: u64) -> Self {
        assert_eq!(
            modulus, P,
            "the trinomial family is defined over Goldilocks only"
        );
        assert!(
            ring_degree % 3 == 0 && (ring_degree / 3).is_power_of_two(),
            "ring degree {ring_degree} is not 3 * 2^k"
        );
        let slots = ring_degree / 3;
        assert!(
            (4..=MAX_SLOTS).contains(&slots),
            "{slots} slots: k must lie in [2, {}]",
            MAX_SLOTS.trailing_zeros()
        );
        let (h, order) = (slots / 2, 3 * slots);
        let theta = goldilocks::pow(GENERATOR, (P - 1) / order as u64);
        let power = |e: usize| goldilocks::pow(theta, (e % order) as u64);

        // Node `i` of a layer is the block whose residue is taken modulo `Y^{2m} - theta^{E_i}`;
        // it splits into `Y^m - theta^{E_i / 2}` and `Y^m + theta^{E_i / 2}`, `-1 = theta^{3h}`.
        // `E_i` is a multiple of `2m` modulo `3L`, hence even.
        let mut exponents = vec![h, 5 * h];
        let mut forward = Vec::with_capacity(slots);
        let mut len = h / 2;
        while len >= 1 {
            let mut next = Vec::with_capacity(2 * exponents.len());
            for &e in &exponents {
                debug_assert_eq!(e % 2, 0);
                forward.push(power(e / 2));
                next.push(e / 2);
                next.push((e / 2 + 3 * h) % order);
            }
            exponents = next;
            len /= 2;
        }
        let roots = exponents.iter().map(|&e| power(e)).collect();

        // The inverse visits the layers in reverse, each layer's blocks in order.
        let mut inverse = Vec::with_capacity(forward.len());
        let mut end = forward.len();
        let mut blocks = slots / 2;
        while blocks >= 2 {
            let start = end - blocks;
            inverse.extend(forward[start..end].iter().map(|&z| goldilocks::inv(z)));
            end = start;
            blocks /= 2;
        }
        debug_assert_eq!(end, 0);

        let omega = power(h);
        let h_inv = goldilocks::inv(h as u64);
        Self {
            ring_degree,
            slots,
            omega,
            forward,
            inverse,
            inv_scale: [h_inv, mul(h_inv, goldilocks::inv(sub(omega, power(5 * h))))],
            roots,
            root_exponents: exponents,
        }
    }

    pub fn ring_degree(&self) -> usize {
        self.ring_degree
    }

    pub fn num_slots(&self) -> usize {
        self.slots
    }

    /// `eta_j`, the root of slot `j`: slot `j` is the residue modulo `X^3 - eta_j`.
    pub fn slot_roots(&self) -> &[u64] {
        &self.roots
    }

    /// `e_j` with `eta_j = theta^{e_j}` for the primitive `3L`-th root of unity
    /// `theta = eta_0` (`e_0 = 1`).
    pub fn slot_root_exponents(&self) -> &[usize] {
        &self.root_exponents
    }

    /// Evaluations of one part `a = (a_0, ..., a_{L-1})`, a polynomial in `Y` modulo `Phi(Y)`,
    /// at `eta_0, ..., eta_{L-1}`.
    fn forward_part(&self, a: &mut [u64]) {
        let (slots, h) = (self.slots, self.slots / 2);
        let a = &mut a[..slots];
        // `a mod (Y^h - w) = a_lo + w a_hi` and `a mod (Y^h - w^5) = a_lo + a_hi - w a_hi`.
        for s in 0..h {
            let t = mul(self.omega, a[s + h]);
            a[s + h] = sub(add(a[s], a[s + h]), t);
            a[s] = add(a[s], t);
        }
        let mut z = 0;
        let mut len = h / 2;
        while len >= 1 {
            for block in a.chunks_exact_mut(2 * len) {
                let zeta = self.forward[z];
                z += 1;
                let (lo, hi) = block.split_at_mut(len);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    let t = mul(zeta, *y);
                    *y = sub(*x, t);
                    *x = add(*x, t);
                }
            }
            len /= 2;
        }
    }

    /// Inverse of [`Self::forward_part`].
    fn inverse_part(&self, a: &mut [u64]) {
        let (slots, h) = (self.slots, self.slots / 2);
        let a = &mut a[..slots];
        // Gentleman-Sande: `(x, y) = (lo + z hi, lo - z hi)` gives `2 lo = x + y` and
        // `2 hi = (x - y) / z`; the factors 2 make `h` once every layer is undone.
        let mut z = 0;
        let mut len = 1;
        while len < h {
            for block in a.chunks_exact_mut(2 * len) {
                let zeta_inv = self.inverse[z];
                z += 1;
                let (lo, hi) = block.split_at_mut(len);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    let (u, v) = (*x, *y);
                    *x = add(u, v);
                    *y = mul(sub(u, v), zeta_inv);
                }
            }
            len *= 2;
        }
        // `u = h (a_lo + w a_hi)`, `v = h (a_lo + w^5 a_hi)`: `a_hi = (u - v) / (h (w - w^5))` and
        // `a_lo = u / h - w a_hi`.
        let [h_inv, hi_scale] = self.inv_scale;
        for s in 0..h {
            let (u, v) = (a[s], a[s + h]);
            let hi = mul(sub(u, v), hi_scale);
            a[s] = sub(mul(u, h_inv), mul(self.omega, hi));
            a[s + h] = hi;
        }
    }

    /// Strided coefficients to slots, in place on the first `3L` u64s of `data`.
    pub fn ntt_forward(&self, data: &mut [u64]) {
        for part in data[..self.ring_degree].chunks_exact_mut(self.slots) {
            self.forward_part(part);
        }
    }

    /// Slots to strided coefficients, in place.
    pub fn ntt_inverse(&self, data: &mut [u64]) {
        for part in data[..self.ring_degree].chunks_exact_mut(self.slots) {
            self.inverse_part(part);
        }
    }

    /// [`Self::ntt_inverse`] of `operand` into `result`.
    pub fn ntt_inverse_into(&self, result: &mut [u64], operand: &[u64]) {
        let n = self.ring_degree;
        result[..n].copy_from_slice(&operand[..n]);
        self.ntt_inverse(result);
    }

    /// Slot `j` of the element at `x`.
    #[inline(always)]
    unsafe fn slot(&self, x: *const u64, j: usize) -> [u64; 3] {
        [
            *x.add(j),
            *x.add(self.slots + j),
            *x.add(2 * self.slots + j),
        ]
    }

    /// Slot-wise product modulo `X^3 - eta_j` of two outputs of [`Self::ntt_forward`]; `result`
    /// may alias either operand.
    #[inline]
    pub unsafe fn mult(&self, result: *mut u64, op1: *const u64, op2: *const u64) {
        for (j, &eta) in self.roots.iter().enumerate() {
            let c = cubic_mul(self.slot(op1, j), self.slot(op2, j), eta);
            for (t, c) in c.into_iter().enumerate() {
                *result.add(t * self.slots + j) = c;
            }
        }
    }

    /// [`crate::SlotRing::dot_many`]: `results + o result_stride` (+)=
    /// `sum_k (op1 + k stride1) (op2 + k stride2 + o output_stride)`, strides in u64s. The terms
    /// are summed before the products by `eta_j`.
    pub unsafe fn dot_many(
        &self,
        results: *mut u64,
        result_stride: usize,
        outputs: usize,
        op1: *const u64,
        stride1: usize,
        op2: *const u64,
        stride2: usize,
        output_stride: usize,
        count: usize,
        accumulate: bool,
    ) {
        let slots = self.slots;
        for o in 0..outputs {
            let mut acc = [[0u64; 5]; MAX_SLOTS];
            for k in 0..count {
                let (a, b) = (
                    op1.add(k * stride1),
                    op2.add(k * stride2 + o * output_stride),
                );
                for (j, acc) in acc[..slots].iter_mut().enumerate() {
                    let terms = cubic_mul_lazy(self.slot(a, j), self.slot(b, j));
                    for (acc, term) in acc.iter_mut().zip(terms) {
                        *acc = add(*acc, term);
                    }
                }
            }
            let result = results.add(o * result_stride);
            for (j, (&eta, acc)) in self.roots.iter().zip(&acc[..slots]).enumerate() {
                let c = [
                    add(acc[0], mul(eta, acc[3])),
                    add(acc[1], mul(eta, acc[4])),
                    acc[2],
                ];
                for (t, c) in c.into_iter().enumerate() {
                    let at = result.add(t * slots + j);
                    *at = if accumulate { add(*at, c) } else { c };
                }
            }
        }
    }

    /// `result (+)= sum_k (op1 + k stride1) (op2 + k stride2)`, strides in u64s; `result` must
    /// not overlap the operands.
    pub unsafe fn dot(
        &self,
        result: *mut u64,
        op1: *const u64,
        stride1: usize,
        op2: *const u64,
        stride2: usize,
        count: usize,
        accumulate: bool,
    ) {
        self.dot_many(
            result, 0, 1, op1, stride1, op2, stride2, 0, count, accumulate,
        );
    }
}
