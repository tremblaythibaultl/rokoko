//! Arithmetic modulo the Goldilocks prime `p = 2^64 - 2^32 + 1`.
//!
//! A residue fills the whole `u64`, so none of HEXL's lazy ranges (`[0, 2q)`, `[0, 4q)`), Barrett
//! shifts or sign-bit tricks apply. Products reduce instead through `2^64 = 2^32 - 1` and
//! `2^96 = -1 (mod p)`, and sums and differences through the carry and the borrow of the `u64`
//! operation. Every function returns a canonical residue in `[0, p)`.

/// The Goldilocks prime `2^64 - 2^32 + 1`.
pub const P: u64 = 0xffff_ffff_0000_0001;

/// `2^64 mod p = 2^32 - 1`.
const EPSILON: u64 = 0xffff_ffff;

/// `x mod p` for any `u64`: `x < 2^64 < 2p`, so one subtraction suffices.
#[inline(always)]
pub fn reduce64(x: u64) -> u64 {
    if x >= P {
        x - P
    } else {
        x
    }
}

/// `x mod p` for any `x < 2^128`. With `x = x0 + 2^64 x1 + 2^96 x2`, `x1 < 2^32`, this is
/// `x0 + (2^32 - 1) x1 - x2`; a borrow out of the subtraction takes `2^64 = 2^32 - 1` too many
/// and a carry out of the addition one `2^64` too few.
#[inline(always)]
pub fn reduce128(x: u128) -> u64 {
    let (x0, high) = (x as u64, (x >> 64) as u64);
    let (x1, x2) = (high & EPSILON, high >> 32);
    let (mut t, borrow) = x0.overflowing_sub(x2);
    if borrow {
        // `t = x0 - x2 + 2^64 >= 2^64 - 2^32 + 1`: no underflow.
        t -= EPSILON;
    }
    // `x1 (2^32 - 1) <= 2^64 - 2^33 + 1`, so after a carry `s + 2^32 - 1` still fits.
    let (s, carry) = t.overflowing_add(x1 * EPSILON);
    if carry {
        s + EPSILON
    } else {
        reduce64(s)
    }
}

/// `a + b mod p` for any `a, b`: operands in `[p, 2^64)` are reduced first, as HEXL's adds
/// tolerate operands up to `2^64 - q` and callers rely on it (for example by passing `q` itself).
#[inline(always)]
pub fn add(a: u64, b: u64) -> u64 {
    let (a, b) = (reduce64(a), reduce64(b));
    let (s, carry) = a.overflowing_add(b);
    if carry {
        // `a + b - 2^64 + 2^32 - 1 <= 2^64 - 2^32`, already below p.
        s + EPSILON
    } else {
        reduce64(s)
    }
}

/// `a - b mod p` for any `a, b`, reducing operands in `[p, 2^64)` first as [`add`] does.
#[inline(always)]
pub fn sub(a: u64, b: u64) -> u64 {
    let (a, b) = (reduce64(a), reduce64(b));
    let (d, borrow) = a.overflowing_sub(b);
    if borrow {
        // `d = a - b + 2^64`, and `a - b + p = d - (2^32 - 1)`.
        d - EPSILON
    } else {
        d
    }
}

/// `-a mod p` for any `a`.
#[inline(always)]
pub fn neg(a: u64) -> u64 {
    let a = reduce64(a);
    if a == 0 {
        0
    } else {
        P - a
    }
}

/// `a b mod p` for any `a, b` (not only residues).
#[inline(always)]
pub fn mul(a: u64, b: u64) -> u64 {
    reduce128(a as u128 * b as u128)
}

/// `base^exp mod p` by square and multiply; `0^0 = 1`.
pub fn pow(base: u64, mut exp: u64) -> u64 {
    let mut base = reduce64(base);
    let mut result = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul(result, base);
        }
        base = mul(base, base);
        exp >>= 1;
    }
    result
}

/// `a^-1 mod p` by Fermat, `a^(p - 2)`. Panics on `a = 0 mod p`, as [`crate::inverse_mod`] does.
pub fn inv(a: u64) -> u64 {
    let a = reduce64(a);
    if a == 0 {
        panic!("input does not have inverse");
    }
    pow(a, P - 2)
}

pub(crate) fn eltwise_add(result: &mut [u64], operand1: &[u64], operand2: &[u64]) {
    for ((r, &a), &b) in result.iter_mut().zip(operand1).zip(operand2) {
        *r = add(a, b);
    }
}

pub(crate) fn eltwise_sub(result: &mut [u64], operand1: &[u64], operand2: &[u64]) {
    for ((r, &a), &b) in result.iter_mut().zip(operand1).zip(operand2) {
        *r = sub(a, b);
    }
}

pub(crate) fn eltwise_mult(result: &mut [u64], operand1: &[u64], operand2: &[u64]) {
    for ((r, &a), &b) in result.iter_mut().zip(operand1).zip(operand2) {
        *r = mul(a, b);
    }
}

/// `result = arg1 * arg2 (+ arg3)`.
pub(crate) fn eltwise_fma(result: &mut [u64], arg1: &[u64], arg2: u64, arg3: Option<&[u64]>) {
    let arg2 = reduce64(arg2);
    match arg3 {
        Some(arg3) => {
            for ((r, &a), &c) in result.iter_mut().zip(arg1).zip(arg3) {
                *r = add(mul(a, arg2), c);
            }
        }
        None => {
            for (r, &a) in result.iter_mut().zip(arg1) {
                *r = mul(a, arg2);
            }
        }
    }
}

pub(crate) fn eltwise_reduce(result: &mut [u64], operand: &[u64]) {
    for (r, &x) in result.iter_mut().zip(operand) {
        *r = reduce64(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(x: u128) -> u64 {
        (x % P as u128) as u64
    }

    #[test]
    fn epsilon_is_two_to_the_64_mod_p() {
        assert_eq!(reference(1 << 64), EPSILON);
        assert_eq!(reference(1 << 96), P - 1);
    }

    #[test]
    fn reduce128_matches_remainder_on_the_extremes() {
        let edges = [0, 1, EPSILON, P - 1, P, P + 1, u64::MAX, 1 << 63, 1 << 32];
        for &a in &edges {
            for &b in &edges {
                let x = (a as u128) << 64 | b as u128;
                assert_eq!(reduce128(x), reference(x), "{a:#x} {b:#x}");
            }
        }
        assert_eq!(reduce128(u128::MAX), reference(u128::MAX));
    }
}
