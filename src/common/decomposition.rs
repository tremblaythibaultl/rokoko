use crate::common::{
    arithmetic::{pow_mod, reduce_signed_wide},
    config::{DEGREE, MOD_Q},
    ring_arithmetic::{Representation, RingElement},
};

impl RingElement {
    pub fn bits_into(&mut self, target: &mut RingElement, from: u64, to: u64) {
        debug_assert!(from < to);

        let mask: u64 = (1u64 << (to - from)) - 1;
        for i in 0..self.v.len() {
            target.v[i] = (self.v[i] >> from) & mask;
        }
    }
}

// Decomposes each element in input into radix parts of base 2^{base_log} using signed (balanced) decomposition.
// Each element x is first shifted by adding k = (b/2) * (1 + b + b^2 + ... + b^{radix-1}) where b = 2^{base_log},
// then decomposed into radix base-b digits, and each digit is shifted back by subtracting b/2.
// This ensures each decomposed part lies in the range [-2^{base_log - 1}, 2^{base_log - 1}) for every x in the
// window [-k, b^radix - k). Since k = (b/2) * Σ b^i, the recomposition is exact: Σ d_i * b^i = (x + k) - k = x,
// with zero offset. A coefficient outside the window keeps its lower digits and carries the rest into the top one,
// which then leaves the range but still recomposes to x (see `carried_digit`); the norm checks, not this function,
// are what bound the digits.
pub fn decompose(input: &[RingElement], base_log: u64, radix: usize) -> Vec<RingElement> {
    if base_log == 1 {
        assert_eq!(
            radix, 1,
            "balanced base-2 decomposition is not supported; use decompose_bits"
        );
        return input.to_vec();
    }

    let small_shift_val = 1u64 << (base_log - 1);
    let big_shift_val = big_shift(base_log, radix, MOD_Q);
    let big_shift = RingElement::all(big_shift_val, Representation::StridedCoefficients);
    let mask = (1u64 << base_log) - 1;
    // A shifted coefficient at or past 2^window_bits is outside the window.
    let window_bits = base_log * radix as u64;

    #[cfg(feature = "debug-decomp")]
    let call_max = std::sync::atomic::AtomicI64::new(0);

    let mut decomposed: Vec<RingElement> = Vec::with_capacity(input.len() * radix);
    let slots = decomposed.spare_capacity_mut();

    let run = |slots: &mut [std::mem::MaybeUninit<RingElement>], input: &[RingElement]| {
        let mut temp = RingElement::all(0, Representation::StridedCoefficients);
        let mut digit = RingElement::new(Representation::StridedCoefficients);
        for (el, slots) in input.iter().zip(slots.chunks_exact_mut(radix)) {
            temp.set_from(el);
            temp.to_representation(Representation::StridedCoefficients);
            #[cfg(feature = "debug-decomp")]
            {
                use crate::common::arithmetic::centered_i64_from_u64_mod_q_scalar as centered;
                let mut local = 0i64;
                for &c in temp.v.iter() {
                    local = local.max(centered(c).abs());
                }
                call_max.fetch_max(local, std::sync::atomic::Ordering::Relaxed);
            }
            temp += &big_shift;
            let outside = window_bits < 64
                && temp.v.iter().fold(0, |acc, &coefficient| acc | coefficient) >> window_bits != 0;

            for (i, slot) in slots.iter_mut().enumerate() {
                let from = i as u64 * base_log;
                for (out, &coefficient) in digit.v.iter_mut().zip(temp.v.iter()) {
                    *out = balanced_digit(coefficient, from, mask, small_shift_val, MOD_Q);
                }
                if outside {
                    for (out, &coefficient) in digit.v.iter_mut().zip(temp.v.iter()) {
                        if coefficient >> window_bits != 0 {
                            *out = carried_digit(
                                coefficient,
                                big_shift_val,
                                base_log,
                                radix,
                                i,
                                MOD_Q,
                            );
                        }
                    }
                }
                digit.representation = Representation::StridedCoefficients;
                digit.to_representation(Representation::IncompleteNTT);
                slot.write(digit.clone());
            }
        }
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // A thread takes whole input elements, so the digits it writes are its own slice of the
        // output and the page faults for that slice are its own too.
        const GRAIN: usize = 256;
        slots
            .par_chunks_mut(GRAIN * radix)
            .zip(input.par_chunks(GRAIN))
            .for_each(|(slots, input)| run(slots, input));
    }
    #[cfg(not(feature = "parallel"))]
    run(slots, input);

    unsafe { decomposed.set_len(input.len() * radix) };

    #[cfg(feature = "debug-decomp")]
    let call_max = call_max.load(std::sync::atomic::Ordering::Relaxed);
    #[cfg(feature = "debug-decomp")]
    {
        let capacity_neg = big_shift_val.min(1 << 62) as i64;
        let capacity_pos = ((1u128 << window_bits) - big_shift_val as u128).min(1 << 62) as i64 - 1;
        tracing::debug!(
            "  [debug] decompose base={} radix={} n={} max|coeff|={} window=[-{}, {}]",
            base_log,
            radix,
            input.len(),
            call_max,
            capacity_neg,
            capacity_pos
        );
    }

    decomposed
}

/// The shift `k = (b/2) Σ_{i < radix} b^i`, `b = 2^base_log`. A window past `log2 q` makes
/// `k >= q`; below `2^62` it is kept whole, as it always was, and the shifting add subtracts `q`
/// once, so `t` is the integer `x + k` less `q` or not and the digits recompose to `x` mod q (the
/// proofs of the existing rings depend on these digits). Past 64 bits, or for a wide `q`, whose
/// add takes canonical operands only, it is reduced mod `q`.
fn big_shift(base_log: u64, radix: usize, q: u64) -> u64 {
    assert!(
        base_log * (radix as u64) < 128,
        "a {radix}-digit base-2^{base_log} window does not fit 128 bits"
    );
    let k: u128 = (0..radix as u64)
        .map(|i| 1u128 << (base_log - 1 + i * base_log))
        .sum();
    match u64::try_from(k) {
        Ok(k) if q < 1 << 62 => k,
        _ => (k % q as u128) as u64,
    }
}

/// The bits `from..from + base_log` of the shifted coefficient `t`, less `half = b/2`, as a
/// residue mod `q`. A `u64` has no bits from 64 on, so a digit there is `-b/2`.
#[inline(always)]
fn balanced_digit(t: u64, from: u64, mask: u64, half: u64, q: u64) -> u64 {
    let value = (t >> from.min(63)) & if from < 64 { mask } else { 0 };
    if value >= half {
        value - half
    } else {
        q - (half - value)
    }
}

/// Digit `i` of a coefficient whose shifted value `t = x + k mod q` is past the window. The
/// lower digits are those of the integer `x + k`, with `x` the centered representative, and the
/// top digit takes everything above them, so `Σ d_i b^i = x` exactly and only the top digit leaves
/// `[-b/2, b/2)`. Signed values are `i128` so that any `q < 2^64` fits.
fn carried_digit(t: u64, big_shift: u64, base_log: u64, radix: usize, i: usize, q: u64) -> u64 {
    let q_signed = q as i128;
    let x = t as i128 - big_shift as i128;
    let shifted = if x > q_signed / 2 {
        t as i128 - q_signed
    } else {
        t as i128
    };
    let half = 1i128 << (base_log - 1);
    let above = shifted >> (i as u64 * base_log);
    let digit = if i + 1 == radix {
        above - half
    } else {
        (above & ((1i128 << base_log) - 1)) - half
    };
    reduce_signed_wide(digit, q)
}

// Decomposes each element in input into radix unsigned binary digits of its representative in [0, q).
// output[index * radix + i] holds, coefficient-wise, bit i of input[index], so every digit is in {0, 1}
// and Σ d_i * 2^i = x. radix must cover the bit length of q; digits beyond it are zero padding.
pub fn decompose_bits(input: &[RingElement], radix: usize) -> Vec<RingElement> {
    assert!(
        radix as u32 > MOD_Q.ilog2(),
        "radix must cover the whole [0, q) representative"
    );
    assert!(radix <= 64, "radix must fit into a u64 representative");

    // Representatives lie in [0, q), so digits n_bits.. are zero, as is their NTT: leave those slots.
    let n_bits = ((MOD_Q.ilog2() + 1) as usize).min(radix);

    let mut decomposed =
        vec![RingElement::zero(Representation::IncompleteNTT); input.len() * radix];

    let mut temp = RingElement::all(0, Representation::StridedCoefficients);

    for (index, el) in input.iter().enumerate() {
        temp.set_from(el);
        // Bit extraction is per coefficient, so the strided ordering is irrelevant here.
        temp.to_representation(Representation::StridedCoefficients);
        debug_assert!(temp.v.iter().all(|&c| c < MOD_Q));

        let digits = &mut decomposed[index * radix..index * radix + n_bits];
        for (i, d) in digits.iter_mut().enumerate() {
            d.representation = Representation::StridedCoefficients;

            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            unsafe {
                use std::arch::x86_64::*;

                let count = _mm_cvtsi32_si128(i as i32);
                let one = _mm512_set1_epi64(1);
                let src = temp.v.as_ptr() as *const i64;
                let dst = d.v.as_mut_ptr() as *mut i64;
                for b in (0..DEGREE).step_by(8) {
                    let x = _mm512_srl_epi64(_mm512_load_epi64(src.add(b)), count);
                    _mm512_store_epi64(dst.add(b), _mm512_and_si512(x, one));
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
            for c in 0..DEGREE {
                d.v[c] = (temp.v[c] >> i) & 1;
            }

            d.to_representation(Representation::IncompleteNTT);
        }
    }

    decomposed
}

// Like decompose, but interleaves by digit index rather than by element.
// decompose([a, b], radix=2)        -> [a0, a1, b0, b1]
// decompose_chunks([a, b], radix=2) -> [a0, b0, a1, b1]
pub fn decompose_chunks_into(
    output: &mut [RingElement],
    input: &[RingElement],
    base_log: u64,
    radix: usize,
) {
    let mut flat = decompose(input, base_log, radix);
    let n = input.len();
    for index in 0..n {
        for i in 0..radix {
            std::mem::swap(&mut output[i * n + index], &mut flat[index * radix + i]);
        }
    }
}

// With the balanced decomposition using k = (b/2) * Σ b^i, the recomposition offset is zero.
// Kept for API compatibility.
pub fn get_composer_offset(_base_log: u64, _radix: usize) -> u64 {
    0
}

// With the balanced decomposition, the offset is zero, so the scaled version is also zero.
// Kept for API compatibility.
pub fn get_decomposed_offset_scaled(_base_log: u64, _radix: usize) -> u64 {
    0
}

pub fn compose_from_decomposed(
    decomposed: &[RingElement],
    base_log: u64,
    radix: usize,
) -> Vec<RingElement> {
    let mut recomposed =
        vec![RingElement::zero(Representation::IncompleteNTT); decomposed.len() / radix];

    for i in 0..recomposed.len() {
        recomposed[i] = RingElement::all(0, Representation::IncompleteNTT);
        for j in 0..radix {
            let mut term = decomposed[i * radix + j].clone();
            // Reduced mod q: 1u64 << k overflows the modulus (and then u64) once k >= log2 q.
            let shift = RingElement::constant(
                pow_mod(2, j as u64 * base_log),
                Representation::IncompleteNTT,
            );
            term *= &shift;
            recomposed[i] += &term;
        }
    }

    recomposed
}

#[test]
fn test_decompose() {
    let mut input = vec![RingElement::all(37, Representation::IncompleteNTT)];
    let base_log = 3; // base 8
    let radix = 4;
    let decomposed = decompose(&mut input, base_log, radix);
    debug_assert_eq!(
        input[0],
        RingElement::all(37, Representation::IncompleteNTT)
    );
    debug_assert_eq!(decomposed.len(), radix * 1);
    // k = 4 * (1 + 8 + 64 + 512) = 2340
    // 37 is shifted to 37 + 2340 = 2377
    // base 8 representation of 2377 = 4 * 8^3 + 5 * 8^2 + 1 * 8^1 + 1 * 8^0
    // so the decomposed elements should be [1, 1, 5, 4]
    // after removing the shift, they should be [1 - 4, 1 - 4, 5 - 4, 4 - 4] = [-3, -3, 1, 0]
    debug_assert_eq!(
        decomposed[0],
        RingElement::all(MOD_Q - 3, Representation::IncompleteNTT)
    );
    debug_assert_eq!(
        decomposed[1],
        RingElement::all(MOD_Q - 3, Representation::IncompleteNTT)
    );
    debug_assert_eq!(
        decomposed[2],
        RingElement::all(1, Representation::IncompleteNTT)
    );
    debug_assert_eq!(
        decomposed[3],
        RingElement::all(0, Representation::IncompleteNTT)
    );

    let mut recomposed = RingElement::all(0, Representation::IncompleteNTT);
    for i in 0..radix {
        let mut term = decomposed[i].clone();
        let shift =
            RingElement::constant(1u64 << (i as u64 * base_log), Representation::IncompleteNTT);
        term *= &shift;
        recomposed += &term;
    }
    debug_assert_eq!(recomposed, input[0]);
}

#[test]
fn out_of_window_coefficients_carry_into_the_top_digit() {
    use crate::common::arithmetic::{centered_i64_from_u64_mod_q_scalar as centered, signed_mod_q};
    let half_q = (MOD_Q / 2) as i64;
    for (base_log, radix) in [(8u64, 2usize), (7, 2), (4, 4)] {
        let half = 1i64 << (base_log - 1);
        let shift: i64 = (0..radix).map(|i| half << (i as u64 * base_log)).sum();
        // The window is [-shift, past).
        let past = (1i64 << (base_log * radix as u64)) - shift;
        let values = [
            0,
            -1,
            -shift,
            past - 1,
            past,
            -shift - 1,
            33242,
            -40000,
            3 * past,
            -3 * shift,
            half_q,
            -half_q,
        ];
        let mut element = RingElement::zero(Representation::Coefficients);
        for (c, &x) in element.v.iter_mut().zip(values.iter()) {
            *c = signed_mod_q(x);
        }
        element.to_representation(Representation::IncompleteNTT);

        let digits = decompose(&[element.clone()], base_log, radix);
        assert_eq!(
            compose_from_decomposed(&digits, base_log, radix)[0],
            element
        );

        // Below the top digit every digit is balanced; the top one is too inside the window.
        for (i, digit) in digits.iter().enumerate() {
            let mut digit = digit.clone();
            digit.to_representation(Representation::Coefficients);
            for (&d, &x) in digit.v.iter().zip(values.iter()) {
                if i + 1 < radix || (-shift..past).contains(&x) {
                    assert!(
                        (-half..half).contains(&centered(d)),
                        "digit {i} of {x} is {} for base 2^{base_log}",
                        centered(d)
                    );
                }
            }
        }
    }
}

#[test]
fn test_random_mod_q() {
    let r = RingElement::random(Representation::IncompleteNTT);
    let data = vec![r];
    let base_log = 13;
    // Enough digits to cover `q`: `ceil(bits(q) / 13)`, so 4 for the 50-bit rings and 5 for
    // Goldilocks.
    let radix = (64 - MOD_Q.leading_zeros() as u64).div_ceil(base_log) as usize;

    let mut decomposed = decompose(&data, base_log, radix);

    let mut recomposed = RingElement::all(0, Representation::IncompleteNTT);
    for i in 0..radix {
        let mut term = decomposed[i].clone();
        let shift =
            RingElement::constant(1u64 << (i as u64 * base_log), Representation::IncompleteNTT);
        term *= &shift;
        recomposed += &term;
    }
    debug_assert_eq!(recomposed, data[0]);

    let mut inf_norm = 0;
    for d in decomposed.iter_mut() {
        d.from_incomplete_ntt_to_strided_coefficients();
        for &v in d.v.iter() {
            let abs_v = if v > MOD_Q / 2 { MOD_Q - v } else { v };
            if abs_v > inf_norm {
                inf_norm = abs_v;
            }
        }
    }

    debug_assert_eq!(inf_norm < (1u64 << (base_log - 1)), true);
}

#[test]
fn test_compose_from_decomposed() {
    let mut input = vec![RingElement::all(37, Representation::IncompleteNTT)];
    let base_log = 3; // base 8
    let radix = 4;
    let decomposed = decompose(&mut input, base_log, radix);
    let recomposed = compose_from_decomposed(&decomposed, base_log, radix);
    debug_assert_eq!(recomposed[0], input[0]);
}

#[test]
fn test_decompose_bits_roundtrip() {
    let input: Vec<RingElement> = (0..3)
        .map(|_| RingElement::random(Representation::IncompleteNTT))
        .collect();

    // 64 exercises the composer at shifts >= log2 q, where an unreduced 2^k would wrap.
    for radix in [MOD_Q.ilog2() as usize + 1, 64] {
        let decomposed = decompose_bits(&input, radix);
        debug_assert_eq!(decomposed.len(), input.len() * radix);

        let recomposed = compose_from_decomposed(&decomposed, 1, radix);
        for i in 0..input.len() {
            debug_assert_eq!(recomposed[i], input[i]);
        }
    }
}

#[test]
fn test_decompose_bits_digits_are_binary() {
    let input: Vec<RingElement> = (0..2)
        .map(|_| RingElement::random(Representation::IncompleteNTT))
        .collect();
    let radix = 64;

    let mut decomposed = decompose_bits(&input, radix);

    for d in decomposed.iter_mut() {
        d.from_incomplete_ntt_to_strided_coefficients();
        for &v in d.v.iter() {
            debug_assert_eq!(v == 0 || v == 1, true);
        }
    }
}

#[test]
fn test_decompose_bits_padding_is_zero() {
    let input = vec![RingElement::random(Representation::IncompleteNTT)];
    let radix = 64;

    let decomposed = decompose_bits(&input, radix);

    let zero = RingElement::all(0, Representation::IncompleteNTT);
    for i in (MOD_Q.ilog2() as usize + 1)..radix {
        debug_assert_eq!(decomposed[i], zero);
    }
}

#[test]
fn test_decompose_bits_edge_values() {
    let input = vec![
        RingElement::all(MOD_Q - 1, Representation::IncompleteNTT),
        RingElement::all(0, Representation::IncompleteNTT),
    ];
    let radix = 64;

    let decomposed = decompose_bits(&input, radix);

    for i in 0..radix {
        debug_assert_eq!(
            decomposed[i],
            RingElement::all(((MOD_Q - 1) >> i) & 1, Representation::IncompleteNTT)
        );
        debug_assert_eq!(
            decomposed[radix + i],
            RingElement::all(0, Representation::IncompleteNTT)
        );
    }

    let recomposed = compose_from_decomposed(&decomposed, 1, radix);
    debug_assert_eq!(recomposed[0], input[0]);
    debug_assert_eq!(recomposed[1], input[1]);
}

#[test]
fn test_decompose_bits_non_uniform_coefficients() {
    let mut el = RingElement::zero(Representation::StridedCoefficients);
    for j in 0..el.v.len() {
        el.v[j] = (j as u64).wrapping_mul(0x9E3779B97F4A7C15) % MOD_Q;
    }
    el.to_representation(Representation::IncompleteNTT);
    let input = vec![el];
    let radix = 64;

    let decomposed = decompose_bits(&input, radix);

    let mut coefficients = input[0].clone();
    coefficients.to_representation(Representation::StridedCoefficients);
    for i in 0..radix {
        let mut digit = decomposed[i].clone();
        digit.to_representation(Representation::StridedCoefficients);
        for j in 0..digit.v.len() {
            debug_assert_eq!(digit.v[j], (coefficients.v[j] >> i) & 1);
        }
    }

    let recomposed = compose_from_decomposed(&decomposed, 1, radix);
    debug_assert_eq!(recomposed[0], input[0]);
}

#[test]
fn test_decompose_bits_preserves_input() {
    let input = vec![
        RingElement::random(Representation::IncompleteNTT),
        RingElement::random(Representation::StridedCoefficients),
    ];
    let before = input.clone();

    let _ = decompose_bits(&input, 64);

    debug_assert_eq!(input, before);
}

/// The digit arithmetic of `decompose` at the Goldilocks prime, where the centered values and the
/// shift leave `i64` and windows reach past 64 bits: every value recomposes to itself mod p, and
/// every digit but a carried top one is balanced.
#[test]
fn digits_recompose_at_goldilocks() {
    const P: u64 = incomplete_rexl::GOLDILOCKS_P;
    let half_p = (P / 2) as i128;
    let centered = |d: u64| match d > P / 2 {
        true => d as i128 - P as i128,
        false => d as i128,
    };
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut random = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state as i64 as i128 >> (state % 63)
    };
    for (base_log, radix) in [(16u64, 4usize), (22, 3), (8, 8), (13, 4), (21, 3), (9, 2)] {
        let k = big_shift(base_log, radix, P);
        let (window, half) = (base_log * radix as u64, 1i128 << (base_log - 1));
        let mut values = vec![
            0,
            1,
            -1,
            half_p,
            -half_p,
            half,
            -half - 1,
            1 << 40,
            -(1 << 40),
        ];
        values.extend((0..64).map(|_| random()));
        for x in values {
            let t = reduce_signed_wide(x + k as i128, P);
            let carried = window < 64 && t >> window != 0;
            let digits: Vec<u64> = (0..radix)
                .map(|i| match carried {
                    true => carried_digit(t, k, base_log, radix, i, P),
                    false => balanced_digit(
                        t,
                        i as u64 * base_log,
                        (1 << base_log) - 1,
                        1 << (base_log - 1),
                        P,
                    ),
                })
                .collect();
            let (mut recomposed, mut weight) = (0u128, 1u128);
            for (i, &d) in digits.iter().enumerate() {
                recomposed = (recomposed + d as u128 * weight) % P as u128;
                weight = (weight << base_log) % P as u128;
                if i + 1 < radix || !carried {
                    assert!(
                        (-half..half).contains(&centered(d)),
                        "digit {i} of {x} is {} for base 2^{base_log}",
                        centered(d)
                    );
                }
            }
            assert_eq!(
                recomposed as u64,
                reduce_signed_wide(x, P),
                "{x} in base 2^{base_log}, {radix} digits"
            );
        }
    }
}
