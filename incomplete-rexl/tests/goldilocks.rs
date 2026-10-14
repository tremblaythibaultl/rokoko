//! The Goldilocks scalar layer against the Python reference vectors
//! (`goldilocks-note/port-checks/gen_vectors.py`) and against `u128` remainders, both through
//! `goldilocks::*` and through the modulus-generic API it is dispatched from.

use incomplete_rexl::goldilocks::{self, P};
use incomplete_rexl::*;
use rand::{Rng, SeedableRng};

const SCALAR: &str = include_str!("vectors/goldilocks_scalar.txt");

/// `(op, numbers)` per vector line of a set.
fn vectors(set: &'static str) -> impl Iterator<Item = (&'static str, Vec<u128>)> {
    set.lines()
        .map(|line| line.split('#').next().unwrap().trim())
        .filter(|line| !line.is_empty())
        .map(|line| {
            let mut fields = line.split_whitespace();
            let op = fields.next().unwrap();
            let numbers = fields
                .map(|x| u128::from_str_radix(x.trim_start_matches("0x"), 16).unwrap())
                .collect();
            (op, numbers)
        })
}

fn modp(x: u128) -> u64 {
    (x % P as u128) as u64
}

#[test]
fn scalar_vectors() {
    let mut seen = std::collections::BTreeMap::new();
    for (op, v) in vectors(SCALAR) {
        let word = |i: usize| u64::try_from(v[i]).unwrap();
        let expected = word(v.len() - 1);
        let (generic, layer) = match op {
            "add" => (
                add_mod(word(0), word(1), P),
                goldilocks::add(word(0), word(1)),
            ),
            "sub" => (
                sub_mod(word(0), word(1), P),
                goldilocks::sub(word(0), word(1)),
            ),
            "mul" if word(0) < P && word(1) < P => (
                multiply_mod(word(0), word(1), P),
                goldilocks::mul(word(0), word(1)),
            ),
            "mul" => {
                let product = goldilocks::mul(word(0), word(1));
                (product, product)
            }
            "reduce64" => {
                let mut out = [0u64];
                eltwise_reduce_mod(&mut out, &[word(0)], P);
                (out[0], goldilocks::reduce64(word(0)))
            }
            "reduce128" => {
                let reduced = goldilocks::reduce128(v[0]);
                (reduced, reduced)
            }
            "inv" => (inv_mod(word(0), P), goldilocks::inv(word(0))),
            "pow" => (
                power_mod(word(0), word(1), P),
                goldilocks::pow(word(0), word(1)),
            ),
            _ => panic!("unknown op {op}"),
        };
        assert_eq!(layer, expected, "goldilocks::{op} {v:x?}");
        assert_eq!(generic, expected, "generic {op} {v:x?}");
        *seen.entry(op).or_insert(0) += 1;
    }
    for op in ["add", "sub", "mul", "reduce64", "reduce128", "inv", "pow"] {
        assert!(seen.get(op).is_some_and(|&n| n > 0), "no {op} vectors");
    }
}

/// Residues biased towards the ends of `[0, p)` and the `2^32` boundaries.
fn residue(rng: &mut impl Rng) -> u64 {
    match rng.random_range(0..4) {
        0 => P - 1 - rng.random_range(0..1 << 33),
        1 => (1 << 31) + rng.random_range(0..1 << 33),
        _ => rng.random_range(0..P),
    }
}

#[test]
fn scalar_ops_match_u128_remainders() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    for _ in 0..200_000 {
        let (a, b) = (residue(&mut rng), residue(&mut rng));
        let wide: u128 = rng.random();
        assert_eq!(goldilocks::add(a, b), modp(a as u128 + b as u128));
        assert_eq!(
            goldilocks::sub(a, b),
            modp(a as u128 + P as u128 - b as u128)
        );
        assert_eq!(goldilocks::mul(a, b), modp(a as u128 * b as u128));
        assert_eq!(goldilocks::reduce128(wide), modp(wide));
        assert_eq!(goldilocks::neg(a), modp(P as u128 - a as u128));
        assert_eq!(add_mod(a, b, P), goldilocks::add(a, b));
        assert_eq!(sub_mod(a, b, P), goldilocks::sub(a, b));
        assert_eq!(multiply_mod(a, b, P), goldilocks::mul(a, b));
    }
    for _ in 0..2_000 {
        let a = residue(&mut rng).max(1);
        assert_eq!(goldilocks::mul(a, goldilocks::inv(a)), 1);
        assert_eq!(inv_mod(a, P), goldilocks::inv(a));
    }
}

/// Moduli from 2^62 on other than p take the `u128` paths; these are primes close to 2^64, 2^63
/// and 2^62.
#[test]
fn wide_moduli_match_u128_remainders() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(2);
    for modulus in [u64::MAX - 58, (1 << 63) - 25, (1 << 62) + 135, P] {
        for _ in 0..20_000 {
            let edge = |rng: &mut rand::rngs::StdRng| match rng.random_range(0..3) {
                0 => modulus - 1 - rng.random_range(0..1 << 20),
                _ => rng.random_range(0..modulus),
            };
            let (a, b) = (edge(&mut rng), edge(&mut rng));
            let (a128, b128, q) = (a as u128, b as u128, modulus as u128);
            assert_eq!(add_mod(a, b, modulus), ((a128 + b128) % q) as u64);
            assert_eq!(sub_mod(a, b, modulus), ((a128 + q - b128) % q) as u64);
            assert_eq!(multiply_mod(a, b, modulus), (a128 * b128 % q) as u64);
            if a != 0 {
                let inverse = inv_mod(a, modulus);
                assert_eq!((a128 * inverse as u128 % q) as u64, 1, "q = {modulus}");
            }
        }
        let base = rng.random_range(2..modulus);
        assert_eq!(
            power_mod(base, modulus - 1, modulus),
            1,
            "Fermat at {modulus}"
        );
    }
}

#[test]
fn eltwise_ops_match_scalar_ops() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    for modulus in [P, u64::MAX - 58, (1 << 62) + 135] {
        for n in [1, 7, 8, 64, 257] {
            let a: Vec<u64> = (0..n).map(|_| rng.random_range(0..modulus)).collect();
            let b: Vec<u64> = (0..n)
                .map(|i| match i % 3 {
                    0 => 0,
                    1 => modulus - 1,
                    _ => rng.random_range(0..modulus),
                })
                .collect();
            let raw: Vec<u64> = (0..n).map(|_| rng.random()).collect();
            let scalar = rng.random_range(0..modulus);
            let mut out = vec![0u64; n];

            eltwise_add_mod(&mut out, &a, &b, modulus);
            assert!((0..n).all(|i| out[i] == add_mod(a[i], b[i], modulus)));
            eltwise_sub_mod(&mut out, &a, &b, modulus);
            assert!((0..n).all(|i| out[i] == sub_mod(a[i], b[i], modulus)));
            eltwise_mult_mod(&mut out, &a, &b, modulus);
            assert!((0..n).all(|i| out[i] == multiply_mod(a[i], b[i], modulus)));
            eltwise_fma_mod(&mut out, &a, scalar, &b, modulus);
            assert!((0..n).all(|i| {
                out[i] == add_mod(multiply_mod(a[i], scalar, modulus), b[i], modulus)
            }));
            eltwise_reduce_mod(&mut out, &raw, modulus);
            assert!((0..n).all(|i| out[i] == raw[i] % modulus));
        }
    }
}

/// HEXL's adds tolerate operands up to `2^64 - q`, and callers pass `q` itself; at a wide modulus
/// there is no such slack, so every add and sub reduces operands in `[q, 2^64)` instead of
/// returning a wrong residue. Run in release too, where the input `debug_assert`s are gone.
#[test]
fn wide_adds_and_subs_reduce_non_canonical_operands() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);
    for modulus in [P, u64::MAX - 58, (1 << 63) - 25, (1 << 62) + 135] {
        let q = modulus as u128;
        let n = 64;
        let mut operand = |kind: usize| match kind % 4 {
            0 => modulus,
            1 => u64::MAX - rng.random_range(0..1 << 20),
            2 => rng.random_range(modulus..=u64::MAX),
            _ => rng.random_range(0..modulus),
        };
        // Every pair of kinds, so that both operands sit near 2^64 together.
        let a: Vec<u64> = (0..n).map(|i| operand(i)).collect();
        let b: Vec<u64> = (0..n).map(|i| operand(i / 4)).collect();
        let sum = |i: usize| ((a[i] as u128 + b[i] as u128) % q) as u64;
        let difference = |i: usize| ((a[i] as u128 % q + q - b[i] as u128 % q) % q) as u64;
        for i in 0..n {
            assert_eq!(add_mod(a[i], b[i], modulus), sum(i), "q = {modulus}");
            assert_eq!(sub_mod(a[i], b[i], modulus), difference(i), "q = {modulus}");
            if modulus == P {
                assert_eq!(goldilocks::add(a[i], b[i]), sum(i));
                assert_eq!(goldilocks::sub(a[i], b[i]), difference(i));
                assert_eq!(goldilocks::neg(a[i]), ((q - a[i] as u128 % q) % q) as u64);
            }
        }
        let mut out = vec![0u64; n];
        eltwise_add_mod(&mut out, &a, &b, modulus);
        assert!((0..n).all(|i| out[i] == sum(i)), "q = {modulus}");
        eltwise_sub_mod(&mut out, &a, &b, modulus);
        assert!((0..n).all(|i| out[i] == difference(i)), "q = {modulus}");
        let scalar = rng.random_range(0..modulus);
        eltwise_fma_mod(&mut out, &a, scalar, &b, modulus);
        assert!(
            (0..n).all(|i| out[i] as u128 == (a[i] as u128 * scalar as u128 + b[i] as u128) % q),
            "q = {modulus}"
        );
    }
}

#[test]
#[should_panic(expected = "lazy [0, 4q) butterflies")]
fn ntt_rejects_wide_moduli() {
    ntt_forward_in_place(&mut [0u64; 8], 8, P);
}
