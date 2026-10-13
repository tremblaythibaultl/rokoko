# Porting RoKoko to Goldilocks

This plan adapts the implementation to the Goldilocks prime `p = 2^64 - 2^32 + 1` and the cyclotomic ring of conductor `m = 9 * 2^k`:

```
R_p = Z_p[X] / (X^{2n} - X^n + 1),   n = 3 * 2^{k-1},   phi = 3 * 2^k   (k = 5: phi = 96, k = 6: phi = 192)
```

It follows the note `goldilocks-note/rokoko-goldilocks.tex`, abbreviated "the note" below. The ring splits into `2^k` cubic slots `F_p[X]/(X^3 - eta_j)` with `eta_j` the primitive `3 * 2^k`-th roots of unity, and contains the subfield `F_{p^3} = Z_p[tau]` with `tau = X^{2^{k-1}} + X^{2^k} - X^{2^{k+1}}`.

The facts marked *(checked)* were verified numerically for `k = 5` and `k = 6` by the scripts in `goldilocks-note/port-checks/`.

## Decisions

| Question | Decision |
|---|---|
| Degree | Support `k = 5` and `k = 6` through ring spec files; develop and test on `k = 5` first. |
| Fine projection rows `n_rp` | `n_rp = 384`, with the JL constants re-derived. Fallback: keep 256 and zero-pad the image to a power of two of ring elements. |
| Arithmetic backend | A Goldilocks module inside `incomplete-rexl`, scalar first, SIMD later (Phase 7). No external field crate. |
| Existing rings | They keep building and passing their tests. The field is selected at compile time from the ring spec, so the current suite stays a regression oracle. |
| Challenge field inside `R_p` | The diagonal of the homogenized slots, with homogenization constants chosen so that the diagonal is `Z_p[tau]` (see Finding 5). |
| Projection batches | `n_bat = 2` at `p` (the note, Section 4.2). |

## Findings that go beyond the note

1. **Modulus width.** The arithmetic assumes `q < 2^50`, or at least two spare bits in a `u64`:
   - `build.rs` rejects `q >= 2^50`.
   - The IFMA 52-bit kernels and float-Barrett (Dekker) products need `q < 2^50`.
   - HEXL's lazy `[0, 4q)` butterflies and `add_uint_mod` overflow.
   - `HALF_WAY_MOD_Q = floor(2^64 / 4q) * q` becomes 0 (or panics in dev builds).
   - `MOD_Q as i64` is negative for `q > 2^63`.

   Goldilocks gets its own reduction, using `2^64 = 2^32 - 1 mod p`.
2. **Slot NTT.** In `Y = X^3`, `Y^{2^k} - Y^{2^{k-1}} + 1 = (Y^{2^{k-1}} - w6)(Y^{2^{k-1}} - w6^5)` with `w6` a primitive sixth root of unity. The transform of each strided part is therefore one trinomial butterfly layer followed by two twisted power-of-two NTTs of length `2^{k-1}` (the NTTRU construction). No radix-3 layer is needed. The current strided layout (part `r` holds `c_{3s+r}`) is already the note's decomposition `x_0(Y) + X x_1(Y) + X^2 x_2(Y)`.
3. **`constant_term_from_incomplete_ntt` computes `Tr(x)/phi = x_0 + x_n/2` in this ring** *(checked)*. Every protocol use of it (the norm check on `<w, conj w>` and the `Tr(r_i) = 0` checks) needs exactly `Tr/phi` (the note, Corollaries 3.4 and 3.5). Rename it `trace_over_degree`, keep it, and fix its unreduced `u64` sum.
4. **The fine projection's dual lift is wrong for this ring.** The code lifts `J_batched` with `conj(j)`, which relies on `ct(conj(a) b) = <cf a, cf b>`. That identity fails here *(checked)*.
   - The correct lift is `j_hat = conj(M j)`. `M` mixes each coefficient pair `(i, i+n)` with `(4/3) [[1, -1/2], [-1/2, 1]]`, so that `Tr(j_hat x)/phi = <j, x>` *(checked)*.
   - Equivalently, the trace-dual basis is `b_i^v = (4 / (3 phi)) (X^{-i} - X^{-(i +- n)} / 2)`.
   - Call sites: `project_fine.rs:142, 327`, the AVX reversal at `project_fine.rs:335-375`, `loader.rs:212`, `loader_verifier.rs:158`.
5. **Choice of the challenge field.** Challenges are sampled on the diagonal of the homogenized slots (`hash.rs` `sample_ring_element_ntt_slots_same_*`, `arithmetic.rs` `field_to_ring_element_into`).
   - The `eta_j` fall into two cubic-residue classes, `j = 1` and `j = 2 mod 3`, of equal size *(checked)*. Half the slots therefore need the inverted homogenization `X -> c Y^{-1}`, which swaps parts 1 and 2, as `INVERTED_SLOTS` does for degree 4.
   - For each slot, exactly one of the three cube-root constants `c_j` maps `tau_j` to a common target *(checked)*. With that choice the diagonal is `Z_p[tau]`, and `conj(r) = r` for every challenge.
6. **The challenges are fixed weight, not uniform ternary.** The note's non-invertibility estimate (Claim 4.3) assumes uniform ternary coefficients, but `short_challenge.rs` samples `TAU` entries equal to `+-1`. `tau` must be chosen for invertibility of differences, not only for the size of the challenge set. Estimates:

   | phi | tau | log2 of the set size | log2 Pr[difference non-invertible], heuristic range |
   |---|---|---|---|
   | 96 | 40 | 130.5 | [-141, -117] |
   | 96 | 48 | 140.4 | [-150, -132] |
   | 192 | 26 | 132.3 | [-117, -83] (too weak) |
   | 192 | 32 | 153.1 | [-131, -103] |
   | 192 | 48 | 199.9 | [-166, -156] |

   The range comes from how much probability mass the non-zero lattice points carry, the same heuristic as in the note.
7. **Operator norm.** `short_challenge.rs` evaluates at the odd powers of a `2N`-th root with a power-of-two FFT. For this ring it must instead evaluate at `zeta_m^j` with `gcd(j, m) = 1`, one `j` per conjugate pair, using a phase table of length `m`. The norm bound then becomes `gamma_C = sqrt(3) * bound` (the note, Lemma 3.8).

## Phases

### Phase 0: baseline and guardrails

- Record the test-suite runtime and the outcomes on the default ring. Record the proof size of `p-26` as a golden value.
- Turn the scripts in `goldilocks-note/port-checks/` into reference-vector generators: slot products, conjugation, trace, dual basis, homogenization and Goldilocks scalar operations. Have them emit vectors that Rust tests can load.

### Phase 1: 64-bit modular arithmetic

- **`incomplete-rexl`.**
  - A Goldilocks scalar layer: `u128` product reduction, add/sub with carry handling, inverse without `i64`.
  - `eltwise_{add,sub,mult,fma,reduce}` paths for `p`, dispatched when `modulus == p`.
  - Fix the overflows in `number_theory.rs`: `add_uint_mod`, `sub_uint_mod`, `inverse_mod`, and the `q < 2^62` asserts.
- **Main crate.** Make every helper correct for any `q < 2^64`:
  - Remove every `MOD_Q as i64`: `decomposition.rs:142`, `arithmetic.rs:101, 329, 338`, `wire.rs:685`, and the tests.
  - Rework the consumers of `HALF_WAY_MOD_Q` (`project_fine.rs`, `helpers.rs`, `projection_matrix.rs`). They should accumulate exactly in `i64`/`i128` over the short inputs and reduce once.
  - Make Fiat-Shamir sampling uniform by rejection: `hash.rs:104, 160-192`.
  - Fix the remaining overflows:
    - the unreduced sum at `verifier.rs:732`;
    - `constant_term_from_incomplete_ntt`;
    - the radix weights `1u64 << (bl * i)` in `builder_verifier.rs:207`;
    - `wire.rs` (`SYMBOLS = 56`, the 6-bit model bounds).

### Phase 2: the ring

- **`build.rs`.** Add a ring spec field for the cyclotomic family and emit a field/ring `cfg`. For this family emit `DEGREE = 3 * 2^k`, `SLOT_DEGREE = 3` and `NUM_SLOTS = 2^k`, and drop the power-of-two and `2N >> v2(q-1)` rules.
- **NTT.** Build it from Finding 2.
- **Slot arithmetic.** A cubic slot product (6 multiplications with Karatsuba, plus 2 by `eta_j`) and slot dot products.
- **Homogenization and the challenge field.**
  - Homogenization with inverted slots and the `tau`-aligned cube roots (Finding 5).
  - Degree-3 multiplication and inverse for `FieldExtension`. Today the `else` branch silently computes a degree-2 product.
- **Conjugation and trace.**
  - Conjugation reference with `X^{-1} = X^{n-1} - X^{2n-1}`. The NTT-domain transform derivation carries over.
  - The naive multiplication oracle.
  - `trace_over_degree`.
- **Slot-degree dispatch.** Make every `match SLOT_DEGREE { 2 => .., _ => degree 4 }` and every `SLOT_DEGREE.ilog2()` transpose explicit: `helpers.rs:552, 574`, `project_fine.rs:353`.

### Phase 3: fold challenges

- Compute the operator norm in the canonical embedding with a length-`m` phase table (Finding 7).
- Calibrate `tau` from both the size of the challenge set and the invertibility estimate (Finding 6), and set `op_norm_bound`.
- Use `gamma_C = sqrt(3) * bound` in the norm schedule and in `debug_hardness.rs`.

### Phase 4: protocol

- **Fine projection.**
  - Use the `conj(M j)` lift.
  - Set `n_rp = 384` and fix the shape checks: `builder.rs:486-548`, `verifier.rs:564`.
  - Replace the `e_1 = DEGREE.ilog2()` tensor layers by a uniform `e_1 in Z_p^phi`. The verifier computes it in `O(phi)`, and the error per row is `(1 + log(.))/p`.
  - Use `n_bat = 2`, and relax the `NOF_BATCHES == 2` assert.
- **Norm check.**
  - Verifier bound: `Tr(v)/phi <= (3/2) beta^2`. The extracted bound becomes `sqrt(3) beta`.
  - Add an explicit guard `3 r' beta'^2 < p/2`. Nothing in the code enforces it today.
- **SNARK layer.** The norm claims (`snark/mod.rs:487`, `examples/claims.rs:75`) now measure the canonical norm divided by `phi`. Update the docs and `test_norm_claim_roundtrip`.

### Phase 5: parameters

- Add `rings/goldilocks_k5.toml`, `rings/goldilocks_k6.toml` and `src/instantiation_goldilocks.rs`, with new digit sizes and ranks.
- Regenerate the `NB_*` tables with `--features calibration`.
- Run the estimator with `q = p` and the `sqrt(3)` extraction factor.

### Phase 6: Goldilocks evaluation layer (the note, Section 5)

- Pack each Goldilocks coefficient as 3 digits of 22 bits.
- Implement the functional `L_{z_lo}`, so that an `F_{p^3}` evaluation claim costs one ring element and one left-right constraint.
- Convert to the outer protocol's model `F_p[Z]/(Z^3 - 2)`.

### Phase 7: performance

- AVX-512, AVX2 and NEON Goldilocks kernels for the NTT, the slot product and the dot products.
- Re-bench against `benches/comparison_ckks.md`.

## Testing

- **Reference vectors.** Check every ring primitive against the Python vectors.
- **Round trips.** Run the round trips in `executor.rs` and `full_chain_verifies` on both families.
- **Negative tests.**
  - A wrong fine-projection lift must be rejected.
  - An over-norm witness must be rejected.
- **Invariants.**
  - `conj(r) == r` holds for challenges.
  - `Tr(j_hat x)/phi == <j, x>` holds on random `x`.

## Items to fold back into the note

- Finding 6: the estimate for fixed-weight challenges.
- Finding 5: `conj(r) = r` once the diagonal is `Z_p[tau]`.
- Finding 4: the explicit trace-dual basis.
- Finding 2: the NTT needs no radix-3 layer.

## Progress

### Phase 0: baseline (done)

Measured on `main` at `9023459`, on an Apple Silicon (arm64, NEON, no AVX-512) machine with `rustc 1.96.0-nightly (2026-03-23)`, default ring (`q = 2^50 - 2687`, `N = 128`).

| Command | Outcome | Wall time |
|---|---|---|
| `incomplete-rexl`: `cargo +nightly test --release` | 7 unit + 27 integration tests pass | 6 s with the build |
| `cargo +nightly test --release --no-fail-fast` | lib: 174 pass, 1 fail; bin (`src/main.rs`): 7 pass; doctests: 1 fail | 253 s (bin tests 251 s) |
| `cargo +nightly test --lib --no-fail-fast` (debug assertions on) | 174 pass, 1 fail | 38 s with the build |
| `ROKOKO_RING=rings/n128_d4.toml cargo +nightly test --release --lib` | 174 pass, 1 fail | 33 s with the build |
| `cargo +nightly run --release -- p-26` | verifies | prover 1.8-2.6 s, verifier 15-21 ms |

Golden values for `p-26`: `Total proof size: 100.3199462890625 KB`, `Wire proof size: 112.0703125 KB`. Both are deterministic across runs.

Two tests already fail on `main`:

- `common::short_challenge::tests::challenge_stream_fingerprint`, on both rings: the stream hashes to `710f97b1…` against the pinned `5670da5c…`. Nothing in this port touches the short-challenge sampler.
- The doctest at `protocol::sumchecks::builder::selected_input_pieces` (line 46): the block is prose, not Rust (`expected one of ! or ::, found =`).

Most unit tests compare with `debug_assert*`, which `--release` compiles out. Only the debug-profile run checks them.

**Reference vectors.** `goldilocks-note/port-checks/gen_vectors.py` (`python3 -I`, deterministic) writes one text file per set into `incomplete-rexl/tests/vectors/`. Each line is `op operand... expected`, in hex, and the Rust tests `include_str!` the files. The sets are entries of a table in the script, so Phase 2 adds its ring sets there. `goldilocks_scalar.txt` holds 4324 vectors:

- add, sub and mul over all pairs of 21 boundary residues (`0`, `2^32 +- 1`, `2^63 +- 1`, `p/2`, `p - 2^32 +- 1`, `p - 1`, …) and 11 random ones;
- mul of unreduced `u64` operands;
- `reduce64` over `[p, 2^64)`;
- `reduce128` over every `(hi, lo)` pair of boundary words, plus `2^96 +- 1`, `2^128 - 1`, `(p - 1)^2` and random values;
- inv and pow.

### Phase 1: 64-bit modular arithmetic (done)

**`incomplete-rexl`.**

- New module `goldilocks` (`GOLDILOCKS_P` is re-exported):
  - `reduce128` through `2^64 = 2^32 - 1` and `2^96 = -1`;
  - add and sub through the carry and the borrow;
  - `mul`, `pow`, `neg`, and `inv` by Fermat;
  - eltwise loops.
- `number_theory`:
  - `is_wide(q) = q >= 2^62`.
  - `add_uint_mod` and `sub_uint_mod` are correct for any `q < 2^64`. They use the carry and the borrow, at no extra cost.
  - `inverse_mod` runs extended Euclid in `i128` and dispatches `p` to Fermat. A non-invertible input panics.
  - `barrett_reduce_128` (and so `multiply_mod` and `pow_mod`) dispatches `p` to `goldilocks` and other wide moduli to `u128 %`.
  - The `q < 2^62` asserts are moved behind that dispatch.
- `eltwise_{add,sub,mult,fma,reduce}_mod` dispatch `p` to `goldilocks`, and other wide moduli to scalar exact loops, before any AVX-512/IFMA/float path. `eltwise_add_mod_assign` sends every wide modulus, `p` included, to `add_uint_mod`. The existing paths are untouched for `q < 2^62`, apart from `eltwise_sub_mod_native` (wrapping arithmetic, same instructions) and `twice_modulus` (no longer overflows).
- `Ntt::with_root` asserts `q < 2^62`, because its lazy `[0, 4q)` butterflies would overflow silently. The Goldilocks NTT is Phase 2.
- Tests in `tests/goldilocks.rs`:
  - every reference vector, through both `goldilocks::*` and the generic API at `p`;
  - 200k random residues (biased towards `p - 1` and `2^32`) against `u128 %`;
  - the primes `2^64 - 59`, `2^63 - 25` and `2^62 + 135` through the generic `u128` paths;
  - every eltwise op against the scalar ops;
  - the NTT guard.

**Main crate.** Each helper is correct for any odd `q < 2^64`. The modulus-generic pieces take `q` as a parameter and are tested at `p` as well as at `MOD_Q`.

- `arithmetic.rs`:
  - New `reduce_signed`, `reduce_signed_wide` and `signed_mod_q` replace every `rem_euclid(MOD_Q as i64)`, `x % (MOD_Q as i64)` and `(MOD_Q as i64 + e) as u64`, in code and in tests.
  - New `LAZY_REDUCTION = q < 2^50`. `HALF_WAY_MOD_Q` no longer overflows; it is 0, and unused, once `4q >= 2^64`.
- The biased-accumulator kernels (`HALF_WAY_MOD_Q`) keep their fast paths under `LAZY_REDUCTION`. Otherwise they sum exactly in `i128` and reduce once, through `ProjectionMatrix::signed_column_sums_mod` and `signed_row_sum_mod`:
  - `compute_j_batched` and `compute_j_batched_collectively` go to `compute_j_batched_exact`;
  - `project_coefficients` goes to `project_coefficients_with::<true>`;
  - `projection_flatter_1_times_matrix` and its `_ref` go to `projection_flatter_1_times_matrix_exact`;
  - the Four-Russians kernel is now documented as wrapping mod `2^64` and as lazy-only.

  Tests check the exact paths against the lazy ones at `MOD_Q`, and the two primitives against entry-by-entry modular accumulation at `MOD_Q` and at `p`.
- `hash.rs`: uniform sampling by rejection. A `u64` word at or above `2^64 - (2^64 mod q)` is replaced by the next word of the same XOF stream (or the next `sample_u64`), in `sample_u64_mod_q`, `sample_ring_element_into` and `sample_field_element_into`. Accepted words reduce exactly as before, and the bound is `p` itself at Goldilocks.
- `decomposition.rs`:
  - The shift `k` is computed in `u128`. While it fits a `u64` and `q < 2^62`, it is kept unreduced, as before: the p-26 windows (55-56 bits) give `k > q`, and reducing `k` changes the digits and the proof size to 101.78 KB. Past 64 bits, or for a wide `q`, it is reduced mod `q`.
  - Digits past bit 63 are `-b/2`.
  - `carried_digit` works in `i128` and takes `q`.
  - New test `digits_recompose_at_goldilocks`, with windows of 18 to 66 bits.
- `ring_arithmetic.rs`:
  - `constant_term_from_incomplete_ntt` sums in `u128`. It is not renamed yet; the rename to `trace_over_degree` is Phase 2.
  - The naive multiplication oracle adds in `u128`.
  - `degree4_mul` takes `slot_mul_mod`, a schoolbook product reduced at every step, outside `LAZY_REDUCTION`. A test checks that it matches at `MOD_Q` and is a ring product for `D = 2, 3, 4` at `p`.
- `verifier.rs`: `expected_ct` sums in `u128`.
- `builder_verifier.rs`: the radix weights are `pow_mod(2, base_log * i)`, as on the prover side.
- `flat.rs`: `MOD_Q / 2 + 1` replaces `(MOD_Q + 1) / 2`.
- `wire.rs`:
  - `SYMBOLS` is 56 below `2^55` and 64 above, so octaves up to 63 fit and the 6-bit model bounds hold.
  - The raw path no longer uses `MOD_Q as i64`.
  - The cell coding is factored into `cell_field` and `cell_residue(q)`, and a new test round-trips Goldilocks residues near `+-p/2`.
- `snark/lowering.rs`: `expand_combination` already reduced at every step through `eltwise_fma_mod`. The stale "u128 lazy accumulation, cap 32 components" comment and its assert are gone.

**Results after Phase 1.** Every failure below is one of the two that already fail on `main`.

| Command | Result |
|---|---|
| `incomplete-rexl`, `--release` and debug | 9 unit + 5 Goldilocks + 27 integration pass |
| `cargo +nightly test --release --no-fail-fast` | lib: 182 pass, 1 fail; bin: 7 pass in 259 s; doctests: 1 fail |
| `cargo +nightly test --lib` (debug) | 182 pass, 1 fail |
| `n128_d4` lib (`--release`) | 182 pass, 1 fail |
| `p-26`, 3 interleaved runs each of the `main` and branch binaries | verifies; 100.3199462890625 KB and 112.0703125 KB, as on `main`; prover 1.77-2.56 s (`main`) against 1.79-1.96 s (branch); verifier about 15 ms on both |

The x86 code was neither run nor type-checked: no installed toolchain has the `x86_64-apple-darwin` standard library, so `cargo check --target x86_64-apple-darwin` fails with E0463. An earlier version of this section claimed the check passed; that could not be reproduced. The new dispatch sits before the `#[cfg(target_arch = "x86_64")]` blocks and the diff has no arch-specific edits, so breakage is unlikely, but it is unverified.

**Behaviour changes for the existing rings.**

- Fiat-Shamir sampling now rejects. A word is rejected with probability `2^-38.6` (default ring) or `2^-41.2` (`n128_d4`), so transcripts and proofs are unchanged unless a rejection happens. None happened in the runs above: the p-26 proof is the same size, and no pinned value changed.
- Shifts `>= log2 q` in the verifier's radix weights are now reduced mod `q`. They used to be unreduced constants.
- `expand_combination` no longer caps all-scalar combinations at 32 components.

**Deferred, and notes for Phase 2.**

- `build.rs` still rejects `q >= 2^50`, so at `p` only the modulus-parameterised helpers are tested. The `LAZY_REDUCTION` branches in `project_fine.rs` and `helpers.rs`, and the `SYMBOLS = 64` model, run end to end only once a Goldilocks ring builds.
- The NTT (`ntt.rs`) refuses wide moduli. The NTTRU-style transform of Finding 2 can reuse `goldilocks::{add, sub, mul}`, with no lazy ranges. `fused_slot4_mult_native` and `fused_incomplete_ntt_mult_native` go through `multiply_mod`, `add_uint_mod` and `sub_uint_mod`, so they are already correct at `p`. `SlotRing`, the IFMA tables and the float kernels must stay gated off (`ifma_tables` and the `< 2^50` checks already do this).
- `slot_mul_mod::<3>` is a correct but slow reference for the cubic slot product, and `field_extension_schoolbook` in the tests is another.
- `decompose`: for `q < 2^62` and windows past `log2 q`, the digits depend on the shifting add subtracting `q` once from an unreduced `k` (see `big_shift`). At `p`, `k` is reduced.
- `HALF_WAY_MOD_Q` and `LAZY_REDUCTION` stay as they are. A faster wide-modulus kernel (signed `i64` over the short witness digits in `project_coefficients`, or split 32-bit halves) is Phase 7.
- The pre-existing failures (`challenge_stream_fingerprint`, the `builder.rs` doctest) are untouched.
- `RingElement::constant` and `RingElement::all` store their value unreduced on purpose: the lazy kernels use them as raw-word containers (`HALF_WAY_MOD_Q_RING_CF`, the decomposition shift). Reducing there changes those biases and breaks the default ring. Values of `q` or more can therefore reach the NTT, so the Goldilocks NTT must accept any `u64` input (reduce on load), as the wide add and sub now do.

### Review fixes after Phase 1

A read-only review of the Phase 0-1 diff found no high- or medium-severity bug. It reproduced one low-severity hazard: the wide-modulus add and sub (`goldilocks::{add, sub}`, and `add_uint_mod`/`sub_uint_mod` for `q >= 2^62`) returned wrong residues for operands in `[q, 2^64)` in release builds, where only a `debug_assert` guarded them. HEXL's adds tolerate operands up to `2^64 - q` and callers rely on it, for example `conjugate_in_place_ref` writing `q - 0 = q`.

- `goldilocks::{add, sub, neg}` reduce their operands first, with one comparison each.
- `add_uint_mod` and `sub_uint_mod` reduce their operands when `q` is wide, through `canonical_wide`. Below `2^62` they are unchanged.
- The wide branch of `eltwise_sub_mod` goes through `sub_uint_mod` instead of `eltwise_sub_mod_native`, which assumes canonical operands.
- `conjugate_in_place_ref` writes `(q - c) mod q`.
- New test `wide_adds_and_subs_reduce_non_canonical_operands`, at `p`, `2^64 - 59`, `2^63 - 25` and `2^62 + 135`. Both operands range over `q`, values near `2^64`, values in `[q, 2^64)` and residues, in every pairing. Removing the reduction from either `goldilocks::add` or the generic wide path makes it fail.

Results: `incomplete-rexl` 9 unit + 6 Goldilocks + 27 integration pass in release and debug; main-crate lib in debug, 182 pass and 1 fail (`challenge_stream_fingerprint`, as on `main`); `p-26` verifies with `100.3199462890625 KB` and `112.0703125 KB`.
