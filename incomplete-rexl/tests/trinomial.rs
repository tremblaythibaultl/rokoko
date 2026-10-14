//! The slot transform of `Z_p[X]/(X^{2n} - X^n + 1)` ([`TrinomialSlotRing`]) at `k = 5` and
//! `k = 6`: slots against direct evaluations, products against the coefficient-domain reference
//! vectors (`goldilocks-note/port-checks/gen_vectors.py`) and against a schoolbook cubic product.

use incomplete_rexl::goldilocks::{self, P};
use incomplete_rexl::trinomial::cubic_mul;
use incomplete_rexl::TrinomialSlotRing;
use rand::{Rng, SeedableRng};

const K5: &str = include_str!("vectors/trinomial_k5.txt");
const K6: &str = include_str!("vectors/trinomial_k6.txt");

/// `(degree, vectors)` for both rings.
fn rings() -> [(usize, &'static str); 2] {
    [(96, K5), (192, K6)]
}

/// `(op, numbers)` per vector line of a set.
fn vectors(set: &'static str) -> impl Iterator<Item = (&'static str, Vec<u64>)> {
    set.lines()
        .map(|line| line.split('#').next().unwrap().trim())
        .filter(|line| !line.is_empty())
        .map(|line| {
            let mut fields = line.split_whitespace();
            let op = fields.next().unwrap();
            let numbers = fields
                .map(|x| u64::from_str_radix(x.trim_start_matches("0x"), 16).unwrap())
                .collect();
            (op, numbers)
        })
}

fn strided(coefficients: &[u64]) -> Vec<u64> {
    let slots = coefficients.len() / 3;
    let mut out = vec![0; coefficients.len()];
    for (s, chunk) in coefficients.chunks_exact(3).enumerate() {
        for (t, &c) in chunk.iter().enumerate() {
            out[t * slots + s] = c;
        }
    }
    out
}

fn coefficients(strided: &[u64]) -> Vec<u64> {
    let slots = strided.len() / 3;
    (0..strided.len())
        .map(|i| strided[(i % 3) * slots + i / 3])
        .collect()
}

/// The product in `R_p` through the slots.
fn ring_mul(ring: &TrinomialSlotRing, a: &[u64], b: &[u64]) -> Vec<u64> {
    let (mut a, mut b) = (strided(a), strided(b));
    ring.ntt_forward(&mut a);
    ring.ntt_forward(&mut b);
    let mut c = vec![0; a.len()];
    unsafe { ring.mult(c.as_mut_ptr(), a.as_ptr(), b.as_ptr()) };
    ring.ntt_inverse(&mut c);
    coefficients(&c)
}

fn schoolbook_cubic(a: [u64; 3], b: [u64; 3], eta: u64) -> [u64; 3] {
    let mut c = [0u128; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut term = a[i] as u128 * b[j] as u128 % P as u128;
            if i + j >= 3 {
                term = term * eta as u128 % P as u128;
            }
            c[(i + j) % 3] = (c[(i + j) % 3] + term) % P as u128;
        }
    }
    c.map(|c| c as u64)
}

fn random_residues(rng: &mut impl Rng, n: usize) -> Vec<u64> {
    (0..n)
        .map(|_| match rng.random_range(0..4) {
            0 => P - 1 - rng.random_range(0..4),
            _ => rng.random_range(0..P),
        })
        .collect()
}

#[test]
fn slot_roots_are_the_primitive_roots_of_phi() {
    for (degree, _) in rings() {
        let ring = TrinomialSlotRing::new(degree, P);
        let (slots, order) = (degree / 3, degree as u64);
        let roots = ring.slot_roots();
        assert_eq!(roots.len(), slots);
        let theta = roots[0];
        assert_eq!(ring.slot_root_exponents()[0], 1);
        let mut seen = std::collections::BTreeSet::new();
        for (&eta, &e) in roots.iter().zip(ring.slot_root_exponents()) {
            // Order exactly 3L: eta^{3L/2} = -1 and eta^L != 1, so eta^h is a primitive sixth
            // root and eta a root of Y^L - Y^{L/2} + 1.
            assert_eq!(goldilocks::pow(eta, order / 2), P - 1);
            assert_ne!(goldilocks::pow(eta, order / 3), 1);
            let z = goldilocks::pow(eta, slots as u64 / 2);
            assert_eq!(
                goldilocks::add(goldilocks::sub(goldilocks::mul(z, z), z), 1),
                0
            );
            assert_eq!(goldilocks::pow(theta, e as u64), eta);
            assert!(seen.insert(eta), "eta repeated");
        }
    }
}

#[test]
fn forward_evaluates_each_part_at_every_root() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    for (degree, _) in rings() {
        let ring = TrinomialSlotRing::new(degree, P);
        let slots = degree / 3;
        let x = random_residues(&mut rng, degree);
        let mut y = x.clone();
        ring.ntt_forward(&mut y);
        for (j, &eta) in ring.slot_roots().iter().enumerate() {
            for t in 0..3 {
                let part = &x[t * slots..(t + 1) * slots];
                let value = part
                    .iter()
                    .rev()
                    .fold(0, |acc, &c| goldilocks::add(goldilocks::mul(acc, eta), c));
                assert_eq!(
                    y[t * slots + j],
                    value,
                    "degree {degree}, slot {j}, part {t}"
                );
            }
        }
    }
}

/// Inputs anywhere in `u64`, `[p, 2^64)` included, are read modulo `p`.
#[test]
fn transforms_round_trip_and_reduce_on_load() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(2);
    for (degree, _) in rings() {
        let ring = TrinomialSlotRing::new(degree, P);
        for _ in 0..20 {
            // Some coefficients below `2^64 - p`, which the lift below can push past `p`.
            let mut x = random_residues(&mut rng, degree);
            for c in x.iter_mut().step_by(3) {
                *c %= u64::MAX - P;
            }
            let mut y = x.clone();
            ring.ntt_forward(&mut y);
            let mut z = y.clone();
            ring.ntt_inverse(&mut z);
            assert_eq!(z, x);
            let mut w = vec![0; degree];
            ring.ntt_inverse_into(&mut w, &y);
            assert_eq!(w, x);

            // `x + p` where it fits, plus words of `[p, 2^64)`.
            let lift = |v: &[u64], rng: &mut rand::rngs::StdRng| -> Vec<u64> {
                v.iter()
                    .map(|&c| match c.checked_add(P) {
                        Some(lifted) if rng.random_bool(0.5) => lifted,
                        _ => c,
                    })
                    .collect()
            };
            let mut unreduced = lift(&x, &mut rng);
            assert!(unreduced.iter().any(|&c| c >= P));
            ring.ntt_forward(&mut unreduced);
            assert_eq!(unreduced, y);
            let mut small = y.clone();
            for c in small.iter_mut().step_by(2) {
                *c %= u64::MAX - P;
            }
            let mut expected = small.clone();
            ring.ntt_inverse(&mut expected);
            let mut unreduced = lift(&small, &mut rng);
            assert!(unreduced.iter().any(|&c| c >= P));
            ring.ntt_inverse(&mut unreduced);
            assert_eq!(unreduced, expected);
            let mut unreduced = lift(&y, &mut rng);
            ring.ntt_inverse(&mut unreduced);
            assert_eq!(unreduced, x);
        }
        let mut top = vec![u64::MAX; degree];
        let mut reduced = vec![u64::MAX - P; degree];
        ring.ntt_forward(&mut top);
        ring.ntt_forward(&mut reduced);
        assert_eq!(top, reduced);
    }
}

#[test]
fn cubic_product_matches_schoolbook() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    for _ in 0..2000 {
        let v = random_residues(&mut rng, 7);
        let (a, b, eta) = ([v[0], v[1], v[2]], [v[3], v[4], v[5]], v[6]);
        assert_eq!(cubic_mul(a, b, eta), schoolbook_cubic(a, b, eta));
    }
    let top = [P - 1; 3];
    assert_eq!(
        cubic_mul(top, top, P - 1),
        schoolbook_cubic(top, top, P - 1)
    );
    // Unreduced operands.
    let wide = [u64::MAX, P, P + 1];
    let reduced = wide.map(goldilocks::reduce64);
    assert_eq!(
        cubic_mul(wide, wide, 5),
        schoolbook_cubic(reduced, reduced, 5)
    );
}

#[test]
fn slot_product_is_the_cubic_product_per_slot() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);
    for (degree, _) in rings() {
        let ring = TrinomialSlotRing::new(degree, P);
        let slots = degree / 3;
        let (a, b) = (
            random_residues(&mut rng, degree),
            random_residues(&mut rng, degree),
        );
        let mut c = vec![0; degree];
        unsafe { ring.mult(c.as_mut_ptr(), a.as_ptr(), b.as_ptr()) };
        let slot = |x: &[u64], j: usize| [x[j], x[slots + j], x[2 * slots + j]];
        for (j, &eta) in ring.slot_roots().iter().enumerate() {
            assert_eq!(slot(&c, j), schoolbook_cubic(slot(&a, j), slot(&b, j), eta));
        }
        // `result` may alias an operand.
        let mut aliased = a.clone();
        unsafe { ring.mult(aliased.as_mut_ptr(), aliased.as_ptr(), b.as_ptr()) };
        assert_eq!(aliased, c);
    }
}

#[test]
fn ring_vectors() {
    for (degree, set) in rings() {
        let ring = TrinomialSlotRing::new(degree, P);
        let slots = degree / 3;
        let mut counts = std::collections::BTreeMap::new();
        let mut tau = None;
        for (op, v) in vectors(set) {
            match op {
                "mul" => {
                    let (a, rest) = v.split_at(degree);
                    let (b, c) = rest.split_at(degree);
                    assert_eq!(ring_mul(&ring, a, b), c, "degree {degree}");
                }
                "eval" => {
                    let (eta, a, r) = (v[0], &v[1..1 + degree], &v[1 + degree..]);
                    let j = ring.slot_roots().iter().position(|&x| x == eta).unwrap();
                    let mut y = strided(a);
                    ring.ntt_forward(&mut y);
                    assert_eq!([y[j], y[slots + j], y[2 * slots + j]], r);
                }
                "tau" => tau = Some(v),
                // Ring-level operations, outside the slot transform: not checked here.
                "conj" | "trace" => {}
                _ => panic!("unknown op {op}"),
            }
            *counts.entry(op).or_insert(0) += 1;
        }
        assert!(counts["mul"] >= 10 && counts["eval"] >= 4, "{counts:?}");
        // tau^3 - 3 tau + 1 = 0.
        let tau = tau.unwrap();
        let tau3 = ring_mul(&ring, &ring_mul(&ring, &tau, &tau), &tau);
        let mut minimal = tau3.clone();
        for (m, &t) in minimal.iter_mut().zip(&tau) {
            *m = goldilocks::sub(*m, goldilocks::mul(3, t));
        }
        minimal[0] = goldilocks::add(minimal[0], 1);
        assert!(minimal.iter().all(|&m| m == 0), "degree {degree}");
    }
}

#[test]
fn dot_products_match_sums_of_products() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    for (degree, _) in rings() {
        let ring = TrinomialSlotRing::new(degree, P);
        let (count, outputs) = (5, 3);
        // `count` left operands with stride `degree + 3`, and `count * outputs` right ones.
        let stride1 = degree + 3;
        let a = random_residues(&mut rng, count * stride1);
        let b = random_residues(&mut rng, count * outputs * degree);
        let expected: Vec<Vec<u64>> = (0..outputs)
            .map(|o| {
                let mut sum = vec![0; degree];
                for k in 0..count {
                    let mut product = vec![0; degree];
                    let right = b[(k * outputs + o) * degree..].as_ptr();
                    unsafe { ring.mult(product.as_mut_ptr(), a[k * stride1..].as_ptr(), right) };
                    for (s, p) in sum.iter_mut().zip(product) {
                        *s = goldilocks::add(*s, p);
                    }
                }
                sum
            })
            .collect();

        let initial = random_residues(&mut rng, outputs * degree);
        for accumulate in [false, true] {
            let mut results = initial.clone();
            unsafe {
                ring.dot_many(
                    results.as_mut_ptr(),
                    degree,
                    outputs,
                    a.as_ptr(),
                    stride1,
                    b.as_ptr(),
                    outputs * degree,
                    degree,
                    count,
                    accumulate,
                )
            };
            for o in 0..outputs {
                let got = &results[o * degree..(o + 1) * degree];
                let want: Vec<u64> = match accumulate {
                    false => expected[o].clone(),
                    true => expected[o]
                        .iter()
                        .zip(&initial[o * degree..])
                        .map(|(&e, &i)| goldilocks::add(e, i))
                        .collect(),
                };
                assert_eq!(got, &want[..], "output {o}, accumulate {accumulate}");
            }
        }

        let mut single = vec![0; degree];
        unsafe {
            ring.dot(
                single.as_mut_ptr(),
                a.as_ptr(),
                stride1,
                b.as_ptr(),
                outputs * degree,
                count,
                false,
            )
        };
        assert_eq!(single, expected[0]);
    }
}

#[test]
#[should_panic(expected = "Goldilocks only")]
fn rejects_other_moduli() {
    TrinomialSlotRing::new(96, (1 << 50) - 2687);
}
