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

### Phase 2: the ring (done)

**Ring spec and `build.rs`.**

- New optional field `family`: `"power_of_two"` (the default, `Z_q[X]/(X^N + 1)`) or `"trinomial"` (`Z_p[X]/(X^{2n} - X^n + 1)`, conductor `9 * 2^k`). The existing specs do not set it and validate exactly as before.
- For `"trinomial"`, `build.rs` requires `degree = 3 * 2^k` with `k` in `{5, 6}` and `mod_q = p`, skips the power-of-two, `q < 2^50` and `2N >> v2(q-1)` rules, and emits `SLOT_DEGREE = 3`, `NUM_SLOTS = 2^k` and `HALF_DEGREE = n = phi/2`.
- `ring.rs` gains `enum RingFamily` and `const RING_FAMILY`, and `build.rs` sets `cfg(rokoko_family = "power_of_two" | "trinomial")`. The code branches on the constant inside functions and on the `cfg` to pick types.
- `rings/goldilocks_k5.toml` (`tau = 48`) and `rings/goldilocks_k6.toml` (`tau = 32`), both with `op_norm_bound = 1000.0`. Both values are marked provisional and uncalibrated (Phase 3).
- The binary (`src/main.rs`) refuses to build for these rings with a `compile_error!` that names Phase 5. The library builds for both.

**`incomplete-rexl`: `trinomial::TrinomialSlotRing`.**

- Same interface as `SlotRing` (`new`, `mult`, `ntt_forward`, `ntt_inverse`, `ntt_inverse_into`, `dot`, `dot_many`), plus `slot_roots` and `slot_root_exponents`. The main crate picks the type with `cfg(rokoko_family)`, so the family is explicit and `SlotRing` is untouched; its doc now says it is the `X^N + 1` family.
- The transform of each strided part is Finding 2: `theta = 7^{(p-1)/3L}` (`L = 2^k`, `h = L/2`), a first layer `a_lo + w a_hi`, `a_lo + a_hi - w a_hi` with `w = theta^h` and `w^5 = 1 - w`, then Cooley-Tukey butterflies that split `Y^{2m} - theta^E` into `Y^m -+ theta^{E/2}`. The inverse runs Gentleman-Sande and undoes the first layer with `(w - w^5)^{-1} h^{-1}`. Everything is scalar `goldilocks::{add, sub, mul}`, which reduce any `u64` input.
- Slot order: slot `j` is position `j` of each part after the forward transform (the butterfly tree's order), with root `eta_j = theta^{e_j}`. `e_0 = 1`, so `alpha = FIELD_SHIFT_FACTOR = eta_0 = theta`.
- `trinomial::cubic_mul`: the product modulo `X^3 - eta` by Karatsuba, 6 products plus 2 by `eta`. `dot` and `dot_many` sum the five `eta`-free terms per slot and multiply by `eta_j` once per output.

**Main crate (`ring_arithmetic.rs`).**

- `SLOT_RING` is `SlotRing` or `TrinomialSlotRing` by family. `SHIFT_FACTORS` is, for the trinomial family, part 0 of the slot transform of `Y = X^3`, that is `eta_j` in slot order; the power-of-two derivation is unchanged.
- `naive_polynomial_multiplication` reduces through `X^{2n} = X^n - 1` from the top down. `conjugate_in_place_ref` goes through `conjugate_coefficients`, which uses the note's eq. (negpow) for the trinomial family.
- `derive_conjugation_transform` is reused unchanged: in slot `j`, `X^{-1} = eta_j^{-1} X^2`, so every basis vector of part `r` lands in exactly one slot of part `(3 - r) mod 3`. Its `debug_assert` on that holds at `k = 5` and `k = 6`.
- Homogenization (Finding 5), `degree3_homogenization`. With `eta_i = alpha^{e_i}`:
  - slots with `e_i = 1 mod 3` use `X -> c Y`, `c^3 = eta_i/alpha`;
  - slots with `e_i = 2 mod 3` use `X -> d Y^{-1} = d Y^2/alpha`, `d^3 = eta_i alpha`, which swaps parts 1 and 2 (`INVERTED_SLOTS`, exactly `NUM_SLOTS/2` slots);
  - of the three cube roots, the one kept is the unique one that sends `tau`'s image to its image in slot 0, so the diagonal is `Z_p[tau]` and `conj(r) = r` for every element of it. Slot 0 keeps the identity.
- Every slot-degree dispatch is explicit (`2`, `3`, `4`, otherwise `unreachable!`): the homogenization swaps, `INVERTED_SLOTS`, `get_roots_of_unity_trans`, `inverse`, the homogenized product and `FieldExtension`'s `Mul` and `MulAssign`. The degree-2 and degree-4 code is unchanged.
- `FieldExtension` at degree 3: `degree3_mul` (Karatsuba through `cubic_mul`, Goldilocks only, checked by a `const` assertion) and `inverse_degree3` (adjugate over the norm, one batch inversion).
- `constant_term_from_incomplete_ntt` is renamed `trace_over_degree`. The computation is unchanged. Its doc says it is the constant term only for `X^N + 1` and `x_0 + x_n/2` for the trinomial family. The call sites are listed under "Notes for Phase 4".
- `subfield_generator()` builds `tau = X^h + X^{2h} - X^{4h}`.
- Lazy-only kernels are gated. A `const` assertion in `helpers.rs` and another in `project_fine.rs` state that `LAZY_REDUCTION` implies slots of degree 2 or 4 and `X^N + 1`. They cover the `SLOT_DEGREE.ilog2()` transposes, `COARSE_GROUP = 16 / SLOT_DEGREE` and the negacyclic reversal of `reduce_conjugate_to_incomplete_ntt`, which also asserts it at run time. Goldilocks is past `LAZY_REDUCTION`, so these are unreachable at `p`. The `[u64; 4]` helper in `ring_to_field_combiner.rs` takes the first `SLOT_DEGREE` entries and is correct at degree 3.

**Tests.**

- `incomplete-rexl/tests/trinomial.rs`, 8 tests at `k = 5` and `k = 6`:
  - the slot roots are the `L` distinct primitive `3L`-th roots, `eta_j = theta^{e_j}`;
  - the forward transform evaluates each part at every `eta_j`;
  - round trips, with inputs in `[p, 2^64)` read modulo `p`;
  - `cubic_mul` against a schoolbook product, including unreduced operands;
  - the slot product per slot, including aliasing;
  - the reference vectors (`mul`, `eval`, and `tau^3 - 3 tau + 1 = 0`);
  - `dot` and `dot_many` against sums of products, with and without accumulation;
  - a non-Goldilocks modulus is rejected.
- Main crate, 2 new tests on every ring: `test_slot_product_matches_slot_mul_mod` and `test_dot_products_match_sums_of_products`.
- Main crate, 6 new tests under `cfg(rokoko_family = "trinomial")` in `ring_arithmetic::tests::trinomial`:
  - `ring_vectors`: every vector through the naive oracle, the NTT product, the homogenized product, both conjugations, `trace_over_degree`, `subfield_generator` and `SHIFT_FACTORS`;
  - `shift_factors_are_the_primitive_roots`;
  - `tau_is_diagonal_after_homogenization`, with `T^3 - 3T + 1 = 0`;
  - `conjugation_fixes_the_challenge_field`, for random `f`, and a random ring element is not fixed;
  - `the_diagonal_is_spanned_by_powers_of_tau`;
  - `incomplete_ntt_reduces_on_load`, through `RingElement::all(u64::MAX)` and `constant(p + 5)`.
- Adapted tests. Their behaviour on the existing rings is unchanged.
  - `test_constant_term_from_incomplete_ntt` became `test_trace_over_degree`.
  - `test_norm_squared_via_conjugation` expects `sum x_i^2 + sum_{i<n} x_i x_{i+n}` for the trinomial family (the note, Corollary 3.5).
  - `decomposition::test_random_mod_q` takes its digit count from the bit length of `q`, `ceil(bits(q) / 13)`: 4 for the 50-bit rings, as before, and 5 at `p`.
  - `hash::rejected_words_are_replaced_in_order` runs its second half at a 50-bit modulus when `MOD_Q = p`, whose rejection bound is `p` itself.
  - `project_fine::reduce_conjugate_matches_ntt_then_conjugate` returns early outside `LAZY_REDUCTION`: it tests a lazy-only kernel on inputs biased by `HALF_WAY_MOD_Q`, which is 0 at `p`.
- Mutation checks: each mutation was applied, the tests run, then the code restored.
  - `incomplete-rexl`: the first layer made negacyclic, `eta` dropped from `cubic_mul`, a term dropped from the lazy dot, a wrong inverse scale, and no reduction of inputs at or above `p`. Each makes 1 to 4 of the trinomial tests fail.
  - Main crate at `k = 5`: a cube root not aligned on `tau`, a sign of the conjugation, the trinomial reduction of the naive product, the degree-3 adjugate, `alpha^2` in `degree3_mul`, and `SHIFT_FACTORS` taken from `Y^2`. Each makes the targeted tests fail; the `tau`-alignment mutation fails exactly the homogenization, `tau` and challenge-field tests.

**Reference vectors.** `gen_vectors.py` has two new sets, `trinomial_k5` and `trinomial_k6`, 36 lines each, all in the coefficient domain:

- `tau`;
- 12 `mul`, including products that hit both reductions, `tau^2` and `tau^3`;
- 10 `conj`, including `X^{n-1}`, `X^n`, `X^{n+1}` and `tau`;
- 7 `trace` (`Tr/phi`);
- 5 `eval`, giving `x mod (X^3 - eta)` for explicit `eta = theta^e`, matched to a slot by value, so independent of the slot order.

The generator is still deterministic: a second run into another directory gives byte-identical files, `goldilocks_scalar.txt` included.

**Results.**

| Command | Result |
|---|---|
| `incomplete-rexl`, `cargo +nightly test --release` and debug | 9 unit + 6 Goldilocks + 27 integration + 8 trinomial pass |
| `cargo +nightly test --lib` (debug), default ring | 184 pass, 1 fail (`challenge_stream_fingerprint`, as on `main`) |
| `ROKOKO_RING=rings/n128_d4.toml cargo +nightly test --release --lib` | 184 pass, 1 fail (the same) |
| `cargo +nightly test --release --no-fail-fast`, default ring | lib: 184 pass, 1 fail (`challenge_stream_fingerprint`); bin: 7 pass in 221 s; doctests: 1 fail (`builder.rs`, as on `main`); 241 s in all |
| `cargo +nightly run --release -- p-26` | `100.3199462890625 KB` and `112.0703125 KB`; 3 interleaved runs against a `HEAD` build: prover 1.91-2.14 s (`HEAD`) and 1.94-2.10 s (branch), verifier 15.1-22.4 ms and 15.4-16.2 ms |
| `ROKOKO_RING=rings/goldilocks_k5.toml cargo +nightly test --lib` (debug) | 179 pass, 12 fail |
| `ROKOKO_RING=rings/goldilocks_k6.toml cargo +nightly test --lib` (debug) | 179 pass, 12 fail (the same 12) |

At `k = 5` and `k = 6`, every ring-layer test passes: `common::ring_arithmetic` (32, of which 6 are trinomial-only), `common::arithmetic` (4), `common::hash` (12) and `common::decomposition` (11). `common::sumcheck_element` and `common::norms` have no unit tests of their own; they are exercised through the sumcheck and protocol tests, which pass apart from the 12 below.

**Failing tests at `p`.** The same 12 fail at `k = 5` and `k = 6`:

| Test | Cause | Phase |
|---|---|---|
| `short_challenge::tests::fft_op_norm_matches_direct_on_fixed_vector`, `samples_have_correct_weight_and_op_norm`, `parametrized_samples_have_correct_weight_and_op_norm`, `sparse_op_norm_matches_dense_fft` | the radix-2 FFT of length `DEGREE` indexes past its array (`short_challenge.rs:83`): 96 is not a power of two | 3 (Finding 7) |
| `short_challenge::tests::challenge_stream_fingerprint` | already fails on every ring | none, pre-existing |
| `project_fine::tests::test_batch_projection`, `test_const_term_relation_to_prove`, `exact_projection_paths_match_the_lazy_kernels` | `projection_height / DEGREE` with a height of 256 and `DEGREE = 96` truncates, and `project_coefficients` indexes past its subwitness | 4 (`n_rp = 384` and the shape checks); the projection identities also need the `conj(M j)` lift (Finding 4) |
| `executor::tests::three_openings_round_trip`, `three_decomposition_chunks_round_trip` | the same projection-shape overrun (`project_fine.rs:552`) | 4, then 3 (challenges) and 5 (parameters) |
| `snark::lowering::tests::test_conj_segment_ct_zero_roundtrip` | the binariness claim `sum w conj(w) - w conj(J)`, `J` the all-ones element, is expected to have a zero constant term for binary `w`, which relies on `ct(a conj b) = <cf a, cf b>`, false for this ring | 4 (SNARK layer, Finding 4) |
| `snark::tests::test_norm_claim_roundtrip` | it expects `ct(<w, conj w>)` to be the coefficient norm; here it is `Tr/phi`, the canonical norm over `phi` | 4 (SNARK norm claims) |

`common::decomposition::test_random_mod_q` is flaky on `main` already: it draws from the unseeded thread RNG (no `init_common`), and its final `inf_norm < 2^12` check fails in about 5% of runs (4 of 40 runs of an unmodified `HEAD` build; 15 of 300 runs of the branch's default-ring test binary and 23 of 300 at `k = 5`). One debug run on each ring hit it. It is not caused by this phase and is left as is.

No test is ignored or `cfg`-gated to hide a failure. The only `cfg`-gated tests are the six new trinomial-only ones.

**Deferred, and notes for Phase 3.**

- `short_challenge.rs` still assumes `X^N + 1`: the radix-2 FFT, `PHASE_LEN = 2N` with `PHASE_MASK`, and the evaluation at odd powers of a `2N`-th root. It compiles at `p`. Finding 7 replaces it with a length-`m` phase table over the `zeta_m^j`, `gcd(j, m) = 1`.
- `tau` and `op_norm_bound` in the Goldilocks specs are placeholders (Finding 6).
- The fixed-weight sampler draws positions as bytes below `DEGREE`, which is fine for 96 and 192.

**Notes for Phase 4.**

- `trace_over_degree` call sites. None of them needs the constant term rather than `Tr/phi`:
  - `sumchecks/runner_verifier.rs`: the three norm claims (`norm_claim`, `most_inner_norm_claim`, `projection_norm_claim`), all of the form `<w, conj w>`, where `Tr/phi` is the right functional (Corollary 3.5);
  - `sumchecks/runner_verifier.rs`: the `Tr(r_i) = 0` check on `constant_term_claims`. Phase 4 must make sure the prover's `r_i` satisfy `Tr = 0` rather than `ct = 0`;
  - `intermediate_sumchecks/runner_verifier.rs`: the intermediate norm claim;
  - `parties/verifier.rs:399` and `:741`, the fine-projection constant-term consistency. It checks `Tr/phi(batched image) = sum c_1 . projection_image_ct`, which holds at `p` only with the `conj(M j)` lift (Finding 4). With today's `conj(j)` lift (`compute_j_batched_exact`) it fails.
- Tests and SNARK code still read the constant term `.v[0]` of conjugate products directly: `project_fine.rs` tests at about lines 1114 and 1370, `snark/mod.rs` (`test_norm_claim_roundtrip`) and `snark/lowering.rs` (binariness). They rely on `ct(conj(a) b) = <cf a, cf b>`, which fails here.
- Power-of-two shapes in the protocol: `projection_height / DEGREE`, the `DEGREE.ilog2()` layers in `project_fine.rs:1318` and `params.rs:110, 131`, and the `e_1` tensor layers.
- `compute_j_batched_exact` lifts with `conj(j)`; Phase 4 replaces it with `conj(M j)`.

**Notes for Phase 7.** `TrinomialSlotRing` is scalar. `dot_many` loops over the outputs outermost, with `5 x 128` accumulators on the stack. A NEON or AVX version can vectorise across slots: each part is contiguous and the butterflies are uniform per layer.

**Unverified.** The x86 paths were neither compiled nor run: the machine has no `x86_64` standard library. The new gates are outside `#[cfg(target_arch = "x86_64")]` blocks; the only edit inside one is a doc comment on `COARSE_GROUP`. Performance at `p` was not measured.

### Review fixes after Phase 2

A read-only review of the Phase 2 diff found no high- or medium-severity bug.

- `build.rs` stripped `_` from every spec value before matching `family`, so an explicit `family = "power_of_two"` (the documented default) failed the build. Only numeric fields are stripped now.
- Corrected the digit-count comment in `decomposition::test_random_mod_q` and the line numbers cited above (`project_fine.rs:552`, `verifier.rs:399` and `:741`).
- `decomposition::test_random_mod_q` is flaky on `c2fecbd` too (unseeded RNG; the reviewer saw 8 failures in 200 runs). Left as is.
