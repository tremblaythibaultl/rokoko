# Porting RoKoko to Goldilocks

This plan adds one ring to RoKoko, next to the default ring (d2) and `n128_d4` (d4): the Goldilocks prime `p = 2^64 - 2^32 + 1` with the cyclotomic ring of conductor `m = 288`,

```
R_p = Z_p[X] / (X^96 - X^48 + 1) = Z_p[X] / Phi_288(X),   N = phi = 96,   n = 48
```

The ring splits into 32 cubic slots `F_p[X]/(X^3 - eta_j)` (`eta_j` the primitive 96th roots of unity, the powers `4^k`, `gcd(k, 96) = 1`), and contains the subfield `F_{p^3} = Z_p[tau]` with `tau = X^16 + X^32 - X^64`.

Two documents drive the plan:

- **The note**, `goldilocks-note/rokoko-goldilocks.tex`. It treats the family `X^{2n} - X^n + 1`, `n = 3 * 2^{k-1}`, of conductor `9 * 2^k`; this ring is its `k = 5`. Citations use the numbering of the compiled note.
- **The proposal**, the maintainer's `goldilocks-288-proposal.pdf` ("Proposal: rokoko over Goldilocks, conductor 288", outside the repo, next to the note). It fixes the scope, the order of the work (a stack of 7 PRs) and several design choices. Section "Alignment with the proposal" below lists where this plan follows it and where it does not.

The facts marked *(checked)* were verified numerically by the scripts in `goldilocks-note/port-checks/`.

## Scope

From the proposal, accepted by the user:

- **One ring, conductor 288.** `Phi_576` (`k = 6`, `N = 192`) is not upstreamed: a sent ring element costs 1.50 KB against 0.75 KB, and the degree-256 power-of-two ring already ended at about 150 KB of wire size against 112 KB. The backend in `incomplete-rexl` stays generic in `k`; the main crate accepts only `N = 96`.
- **PCS parameter sets only**, `p-26` then `p-28`. SNARK mode (`--features snark`) and the exact-norm (EN) parameter sets are out of scope. Building SNARK mode for the trinomial family is a compile error.
- **Every PR keeps the d2 and d4 proofs byte-identical** to `main` (`9023459`). This is checked by comparing proof hashes at `p-26` and `p-28` on both rings, not only proof sizes.
- **Selected through `ROKOKO_RING`** like the other rings, with its parameters in its own instantiation module.

## Decisions

| Question | Decision |
|---|---|
| Ring | Conductor 288, `N = 96` only (see Scope). |
| Fine projection rows `n_rp` | **Provisional:** `n_rp = 384`, an image of exactly 4 ring elements, with the JL constants re-derived. The proposal keeps 256 rows in 3 ring elements with 32 zero coefficients; that shape does not tile the power-of-two placements (Finding 9). Settle with the maintainer before PR 4. |
| Arithmetic backend | A Goldilocks module inside `incomplete-rexl`, no external field crate. Scalar first; AVX-512 IFMA later (proposal, Section 2.5). No NEON: aarch64 keeps the scalar fallback. |
| Existing rings | Unchanged, byte for byte. The family is selected at compile time from the ring spec, so the existing suite stays a regression oracle. |
| Challenge field inside `R_p` | The diagonal of the homogenized slots, with the cube roots aligned on `tau` so that the diagonal is `Z_p[tau]` and `conj(r) = r` (Finding 5). The proposal's `alpha = 4` with cube roots `x^e` gives another copy of `F_{p^3}`, not fixed by conjugation; both are correct, and this plan keeps the aligned one. |
| Projection batches | `n_bat = 2` at `p` (the note, Section 5.2). |
| Fold challenges | Fixed weight `tau = 40`, rejection on `||sigma(c)||_inf <= 12.4`, so `gamma_C = sqrt(3) * 12.4 = 21.48` (the proposal's operating point, Section 2.4; Finding 6). |
| Norm check | `Tr(v)/phi <= (3/2) beta^2`, extracted bound `sqrt(3) beta`, and the no-wrap guard `(3/2) r' beta'^2 < p/2` (Finding 8; the proposal, Section 2.3). |
| Process | The proposal's 7 stacked PRs. The phase commits made so far are re-split into PRs 1-4. |

## Findings that go beyond the note

1. **Modulus width.** The arithmetic assumes `q < 2^50`, or at least two spare bits in a `u64`:
   - `build.rs` rejects `q >= 2^50`.
   - The IFMA 52-bit kernels and float-Barrett (Dekker) products need `q < 2^50`.
   - HEXL's lazy `[0, 4q)` butterflies and `add_uint_mod` overflow.
   - `HALF_WAY_MOD_Q = floor(2^64 / 4q) * q` becomes 0 (or panics in dev builds).
   - `MOD_Q as i64` is negative for `q > 2^63`.

   Goldilocks gets its own reduction, using `2^64 = 2^32 - 1` and `2^96 = -1 mod p`.
2. **Slot NTT.** `X^96 - X^48 + 1 = (X^48 + w)(X^48 + w^2)` with `w = 2^32 - 1` a primitive cube root of unity, then radix-2 layers down to the cubics. No radix-3 layer is needed. The code applies the same factorisation to each strided part in `Y = X^3` (the NTTRU construction): one trinomial layer, then two twisted power-of-two NTTs of length 16. Every slot root and twiddle is a power of 2 (`theta = 7^((p-1)/96) = 2^154`, *checked*), so a later kernel can multiply by them with shifts.
3. **`constant_term_from_incomplete_ntt` computes `Tr(x)/phi = x_0 + x_48/2` in this ring** *(checked)*. Every protocol use of it (the norm check on `<w, conj w>` and the `Tr(r_i) = 0` checks) needs exactly `Tr/phi` (the note, Corollaries 4.3 and 4.4). It is renamed `trace_over_degree` (the proposal keeps the old name with the new meaning), and its unreduced `u64` sum is fixed.
4. **The fine projection's dual lift is wrong for this ring.** The code lifts `J_batched` with `conj(j)`, which relies on `ct(conj(a) b) = <cf a, cf b>`. That identity fails here *(checked)*: `ct(a conj(b)) = <a, b> + sum_{i<48} a_i b_{i+48}` (also in the proposal, Section 2.1).
   - The correct lift is `j_hat = conj(M j)`. `M` mixes each coefficient pair `(i, i+48)` with `(4/3) [[1, -1/2], [-1/2, 1]]`, so that `Tr(j_hat x)/phi = <j, x>` *(checked)*.
   - Equivalently, the trace-dual basis is `b_i^v = (4 / (3 phi)) (X^{-i} - X^{-(i +- 48)} / 2)`. The proposal states the requirement ("J is lifted through the trace-dual basis", Section 2.3) but its PR list and code inventory do not include the change.
5. **Choice of the challenge field.** Challenges are sampled on the diagonal of the homogenized slots (`hash.rs` `sample_ring_element_ntt_slots_same_*`, `arithmetic.rs` `field_to_ring_element_into`).
   - The `eta_j` fall into two cubic-residue classes of 16 slots each *(checked)*. Half the slots therefore need the inverted homogenization `X -> c Y^{-1}`, which swaps parts 1 and 2, as `INVERTED_SLOTS` does for degree 4. The proposal has the same two classes.
   - For each slot, exactly one of the three cube-root constants maps `tau` to a common target *(checked)*. With that choice the diagonal is `Z_p[tau]`, and `conj(r) = r` for every field challenge.
6. **The challenges are fixed weight, not uniform ternary.** The note's non-invertibility estimate (Claim 5.4) assumes uniform ternary coefficients, but `short_challenge.rs` samples `TAU` entries equal to `+-1`, so `tau` must be chosen for invertibility of differences as well as for the size of the challenge set. The proposal says only "checked, fine". Estimates at `phi = 96`:

   | tau | log2 of the set size | log2 Pr[difference non-invertible], heuristic range |
   |---|---|---|
   | 39 | 128.95 | [-139, -115] |
   | 40 | 130.47 | [-140.7, -117.1] |
   | 48 | 140.4 | [-150, -132] |

   The range comes from how much probability mass the non-zero lattice points carry, the same heuristic as in the note. With rejection at 12.4, `tau = 40` keeps `2^128.86` challenges for 3.04 attempts.
7. **Operator norm.** `short_challenge.rs` evaluated at the odd powers of a `2N`-th root with a power-of-two FFT. For this ring it evaluates at `zeta_288^j` with `gcd(j, 288) = 1`, one `j` per conjugate pair, using a phase table of length 288. The bound on the coefficient operator norm is then `gamma_C = sqrt(3) * bound` (the note, Lemma 4.7; the proposal, Section 2.4).
8. **The no-wrap guard is `(3/2) r' beta'^2 < p/2`, not `3 r' beta'^2 < p/2`.** The note's eq. (4) bounds every coefficient of `<w*, conj w*>`, which is more than the check needs. The verifier reads only `Tr(v)/phi`. For an integer vector, `I = Tr(<w*, conj w*>)/phi` is the sum over the entries `x` of `w*` of `||x||^2 + sum_{i<48} x_i x_{i+48}`, an integer (the note, Corollary 4.4) in `[||w*||^2/2, (3/2) ||w*||^2]` (Proposition 4.2), and `I = Tr(v)/phi mod p`. So `(3/2) r' beta'^2 < p/2` suffices: one bit better, and the proposal's "lemma to write" is Corollary 4.4 with Proposition 4.2. It fails for other conductors (the proposal's `f = 15`), which is why the paper keeps `f_hat`.
9. **The projection image must tile power-of-two shapes.** The image is chunked by `projection_height / DEGREE`, its height is a power of two (`project_fine.rs:495-533`), commitment placements must be power-of-two sized (`config_generator.rs:621`, and the "a slice must be dyadic" check in `commitment.rs:224`), and the consistency step multiplies one ring element `e`, built from `DEGREE.ilog2()` layers, by the scalar tensor `c_2 (x) c_0 (x) e_0` (`sumchecks/loader.rs` about 182-215). The proposal's 3-element image with a `2^8` tensor over positions 0-255 gives 3 different ring multipliers per chunk, which is not a power-of-two axis, so it almost certainly needs padding to 4 elements. At 4 elements, 384 JL rows cost the same image size as 256 and give stronger JL constants. If the padding coefficients ever carry zero batching weight, the lemma must cover them: they are unconstrained, and only the bound `||x||^2 <= 2 Tr/phi` keeps them harmless.
10. **Byte-identity on the existing rings holds with overwhelming probability, not by construction**, for three changes of PR 3 (Phase 1): Fiat-Shamir sampling rejects words at or above `2^64 - (2^64 mod q)` (probability `2^-38.6` per word on d2, `2^-41.2` on d4); verifier radix-weight shifts `>= log2 q` are reduced mod `q`; `expand_combination` no longer caps all-scalar combinations at 32 components. The proof-hash check is what establishes byte-identity for the PCS sets.

## The PR stack

The proposal's 7 stacked PRs (its Section 5). The phases this plan was written in map onto them as follows.

| PR | Content | Former phase | Status |
|---|---|---|---|
| 1. Ring spec | Explicit ring kind (`family = "trinomial"`), `build.rs` accepts `N = 96`, a non-power-of-two degree and `p > 2^63`; emitted `RingFamily`, `HALF_DEGREE`, `SLOT_DEGREE = 3`, `NUM_SLOTS = 32`. No behaviour change. | 2 (`build.rs`, `ring.rs`, `main.rs`) | done |
| 2. Goldilocks backend | `incomplete-rexl` only: field (`goldilocks.rs`), wide-modulus fixes in `number_theory.rs` and the eltwise dispatch, trinomial NTT, cubic slot multiply, dot products, scalar code, tests against the naive product mod `(Phi_288, p)` and the reference vectors. The main crate is untouched. | 1 and 2 (`incomplete-rexl`) | done, scalar |
| 3. Ring algebra | Conjugation and the trace readout; degree-3 `FieldExtension` with homogenisation; 64-bit wire format and rejection sampling in the transcript; the `MOD_Q as i64` / `MOD_Q * 4` audit; exact `i128` kernels outside `LAZY_REDUCTION`; ring-level identity tests. | 1 and 2 (main crate) | done |
| 4. Shapes | The challenge sampler at the 288th roots; the fine-projection image shape (`n_rp`, Finding 9), the `conj(M j)` lift (Finding 4), the placement of the `c_1` tensor and `e_1`. | 3, and 4 (projection) | sampler done |
| 5. Norm accounting | `gamma_C = sqrt(3) * 12.4` (done in `debug_hardness.rs`); the canonical exact norm `Tr(v)/phi <= (3/2) beta^2` and the extracted `sqrt(3) beta`; the guard `(3/2) r' beta'^2 < p/2` (Finding 8); `Tr(r_i) = 0` rather than `ct(r_i) = 0` on the prover side; the estimator with `q = p`, `gamma_C` and the `sqrt(3)` extraction. | 3 (`gamma_C`), 4 (norm check), 5 (estimator) | `gamma_C` done |
| 6. Instantiation | `rings/goldilocks_k5.toml` and its instantiation module with `p-26` and `p-28`; the `NB_*` tables from `--features calibration`; end-to-end verification; `debug-hardness`; numbers against d4. Lifts the `compile_error!` in `src/main.rs` for the plain chain. | 5 | spec file done |
| 7. Commitment kernel | Measure the direct Goldilocks dot product against a port of the VNNI CRT kernel (#113, removed in #157) on the `p-28` root shape and keep the faster. | new | to do |

After the stack, outside the proposal:

- **Goldilocks evaluation claims** (the note, Section 6; former Phase 6). An `F_{p^3}` evaluation point costs one ring element and one left-right constraint through the functional `L_{z_lo}`, with the outer model `F_p[Z]/(Z^3 - 2)`. Waiting on the maintainer: the proposal packs 64 field elements per ring element as 4 digits of 16 bits (2/3 fill), the note 3 digits of 22 bits at full fill (Question 2).
- **Performance** (former Phase 7): AVX-512 IFMA kernels for the NTT, the slot product and the dot products, shifts for the power-of-2 twiddles, Karatsuba dot products flushed every 2048 terms, vectorised sumcheck field code (the proposal, Section 2.5). No NEON. Re-bench against `benches/comparison_ckks.md`.

### Code sites still to change (PRs 4 and 5)

Merged from this plan's notes and the proposal's inventory (its Section 2.7). The proposal's line numbers are on `main` (`9023459`); the others are on this branch.

- **`projection_height / DEGREE`** (16 sites, rounds 256/96 down to 2), on `main`: `project_fine.rs:36, 154, 467-484, 802, 1017`; `sumchecks/builder.rs:486, 540`; `sumchecks/builder_verifier.rs:791, 847`; `intermediate_sumchecks/builder.rs:90`; `intermediate_sumchecks/builder_verifier.rs:96`; `verifier.rs:350, 371, 714, 720`. On this branch the executor tests overrun at `project_fine.rs:552`.
- **The `c_1` tensor split by `DEGREE.ilog2()`**, on `main`: `sumchecks/loader.rs:177-212`, `sumchecks/loader_verifier.rs:148-156`, `project_fine.rs:755-764`. Also `project_fine.rs:1318` and `params.rs:110, 131` on this branch. With `e_1` uniform in `Z_p^phi`, the verifier computes it in `O(phi)` and the error per row is `(1 + log(.))/p`.
- **The dual lift** (Finding 4): `compute_j_batched_exact` and the call sites `project_fine.rs:142, 327`, the AVX-512 reversal at `project_fine.rs:335-375`, `loader.rs:212` (`e.conjugate_in_place()`), `loader_verifier.rs:158`.
- **`trace_over_degree` call sites**, none of which needs the constant term rather than `Tr/phi`: the three norm claims in `sumchecks/runner_verifier.rs`, the `Tr(r_i) = 0` check on `constant_term_claims` there, the intermediate norm claim in `intermediate_sumchecks/runner_verifier.rs`, and the fine-projection consistency in `parties/verifier.rs:399, 741`. The last holds at `p` only with the `conj(M j)` lift.
- **Shape checks:** `builder.rs:486-548`, `verifier.rs:564`, and the `NOF_BATCHES == 2` assert.
- **Norm accounting:** the verifier's exact-norm bound, the new no-wrap guard, `estimator.rs:41-52` (on `main`: plain SIS with `n = rank * 96`, `q = p`, coefficient norm).
- **Tests reading `.v[0]` of conjugate products** (`project_fine.rs` tests at about lines 1114 and 1370) rely on `ct(conj(a) b) = <cf a, cf b>` and must read `Tr/phi` or use the lift.

## Alignment with the proposal

Same in both: the ring, its slots and the NTT strategy (no radix-3 layer); the trace readout `Tr/N = a_0 + a_48/2`; the norm check `Tr(v)/N <= 1.5 beta^2` and the `sqrt(3)` extraction; the evaluation set of the operator norm and `gamma_C = sqrt(3) * bound`; the two homogenization classes; a dedicated backend; selection through `ROKOKO_RING`; the PR order.

Adopted from the proposal: conductor 288 only; `tau = 40` and bound 12.4; the guard constant 1.5; the PCS-only scope; byte-identity checked per PR; the PR stack; no NEON; the commitment-kernel PR; its code inventory.

Kept from this plan, to raise with the maintainer:

1. **Projection shape** (Finding 9). How do 3-element image chunks fit the power-of-two placements and the single-`e` consistency step? This plan keeps `n_rp = 384` (4 elements) provisionally.
2. **Packing and evaluation claims.** Why 4 digits of 16 bits at 2/3 fill rather than 3 of 22 bits, or 6 of 11 bits (same `2^25` ring elements for `2^29` values, full fill)? The proposal's own estimate charges 1.5x for the empty coefficients. And how are `F_{p^3}` evaluation points handled? The note's Section 6 has a mechanism (`L_{z_lo}`).
3. **The dual lift and `Tr(r_i) = 0`** are missing from the proposal's PR list (Finding 4, and the `trace_over_degree` sites above).
4. **Calibration of the challenge bound.** Under the rule that reproduces the existing rings (smallest bound keeping `2^128` after rejection), `tau = 40` would give 11.8 at 5.44 attempts; 12.4 trades 4.6% of `gamma_C` for about 3 attempts. Which rule does the proposal use? This plan adopts 12.4.
5. **Challenge-field constants** (Finding 5): this plan keeps the `tau`-aligned cube roots, which make `conj(r) = r`.
6. **VNNI kernel margin.** If coefficients of `A w` reach `2^94`, a signed CRT needs a product above `2^95`: 8 primes, not 7. The listed "16-bit primes" are at most 14 bits.
7. **Smaller corrections.** `||x||_inf <= (4/pi) ||sigma(x)||_inf` holds but the exact constant is `2/sqrt(3)`; the PCS throughput estimate uses a 70 ns kernel where the measured one gives about 38.5 s instead of 34.7 s; the proof-size discussion omits the larger `F_{p^3}` sumcheck messages (the note, Section 6).

## Testing

- **Reference vectors.** Every ring primitive against the Python vectors (`gen_vectors.py`).
- **Byte-identity.** For every PR, the proof hashes of `p-26` and `p-28` on d2 and d4 equal those of `main`.
- **Round trips.** The round trips in `executor.rs` and `full_chain_verifies` on both families.
- **Negative tests.**
  - A wrong fine-projection lift must be rejected.
  - An over-norm witness must be rejected.
- **Invariants.**
  - `conj(r) == r` holds for field challenges.
  - `Tr(j_hat x)/phi == <j, x>` holds on random `x`.
  - Accepted fold challenges have coefficient operator norm at most `gamma_C`.

## Items to fold back into the note

- Finding 8: the guard `(3/2) r' beta'^2 < p/2`, replacing eq. (4), with its proof from Corollary 4.4 and Proposition 4.2.
- Finding 6: the estimate for fixed-weight challenges, and the chosen `tau = 40`.
- Finding 5: `conj(r) = r` once the diagonal is `Z_p[tau]`.
- Finding 4: the explicit trace-dual basis.
- Finding 2: the NTT needs no radix-3 layer.
- The measured `||c||_op / ||sigma(c)||_inf` range for fixed-weight challenges (1.08-1.66, mean 1.35).

## Progress

The log below is in the order the work was done (Phases 0-3). Phases 0-2 covered `k = 5` and `k = 6`; only `k = 5` is upstreamed (see Scope).

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

- `common::short_challenge::tests::challenge_stream_fingerprint`, on both rings, against the pinned `5670da5c…`: the stream hashes to `710f97b1…` on the default ring and to `422675347b…` on `n128_d4` (corrected after the Phase 3 review). Nothing in this port touches the short-challenge sampler.
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
  - `test_norm_squared_via_conjugation` expects `sum x_i^2 + sum_{i<n} x_i x_{i+n}` for the trinomial family (the note, Corollary 4.4).
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
  - `sumchecks/runner_verifier.rs`: the three norm claims (`norm_claim`, `most_inner_norm_claim`, `projection_norm_claim`), all of the form `<w, conj w>`, where `Tr/phi` is the right functional (the note, Corollary 4.4);
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

### Phase 3: fold challenges (done)

**Operator norm (`short_challenge.rs`, Finding 7).**

- The module evaluates challenges at `zeta_m^j`, `gcd(j, m) = 1`, `j < m/2`, one `j` per conjugate pair, with a phase table `T[t] = exp(2 i pi t / m)` of length `m`, the conductor. New constants `CONDUCTOR` (`2N`, or `3 phi = 9 * 2^k`) and `PHASE_LEN = CONDUCTOR`, and a public `EVAL_EXPONENTS`, the `phi/2` exponents found by a `gcd` filter.
- Trinomial paths, each selected by `RING_FAMILY` inside the existing functions:
  - `op_norm_sq_sparse` (the sampler's hot path) and `op_norm_sq_sparse_scalar` go to `op_norm_sq_sparse_trinomial`. The exponents prime to `m` below `m/2` are `6t + 1` and `6t + 5` for `t < N/4`, so a nonzero at position `p` steps through two phase indices by `6p` mod `m`, with one conditional subtraction each. This is `tau * phi/2` additions per attempt, as for `X^N + 1`.
  - `op_norm` (the dense reference) evaluates at every `EVAL_EXPONENTS` entry through the table, with the index `(j k) mod m` computed directly rather than stepped.
  - `op_norm_direct` uses `cos` and `sin` at all `phi` exponents `j < m` that are odd and not multiples of 3, without the table or the `gcd` filter.
- The power-of-two path does not change. The phase angle keeps its old expression `pi t / N`, so the table is bit-identical, and the AVX-512 kernel and `PHASE_ROWS` are textually unchanged. `op_norm_sq_sparse` returns to the trinomial path before the `#[cfg(avx512f)]` dispatch, so trinomial builds never reach the kernel's `PHASE_MASK` indexing. The fingerprint stream is still `710f97b1…` on the default ring, and `422675347b…` on `n128_d4`, the same value as a `HEAD` build in a temporary worktree. `p-26` is unchanged.
- New `GAMMA_C`, the bound on the coefficient-embedding operator norm of an accepted challenge: `T_OP_NORM_BOUND` for `X^N + 1`, where `||c||_op = ||sigma(c)||_inf`, and `sqrt(3) * T_OP_NORM_BOUND` for the trinomial family (the note, Lemma 4.7).

**`gamma_C` consumers.** `T_OP_NORM_BOUND` is used only by `debug_hardness.rs`, in its two live extracted-witness bounds (`check_sumcheck_round`, `check_simple_round`) and in one commented-out copy. All three now use `GAMMA_C`. The other users are the sampler itself and the `n128_d4` instantiation's spec assert. Nothing else consumes the challenge norm: the `NB_*` norm schedule is measured, not derived from it. The module builds with `--features debug-hardness` on the default ring and at `k = 5`. The other `sqrt(3)`, the extracted norm-check bound `sqrt(3) beta` (the note, Section 5.3, "Norm check"), is a separate Phase 4 item and is not in `debug_hardness.rs` yet.

**Calibration of `tau` and `op_norm_bound`.**

How the existing rings were calibrated. Commit `fe7f4dd` ("Give the degree-4 rings a 2^128 fold challenge space") applies this rule: take the smallest `tau` with `C(N, tau) 2^tau >= 2^128` before rejection, then the smallest bound, in steps of 0.1, that keeps `2^128` challenges after rejection. The rule was reconstructed from the commit's values, since the repository has no calibration script. Here `log2 |C| + log2 Pr[accept] >= 128`. The procedure below reproduces both rings:

| ring | `tau` | bound | attempts | after rejection |
|---|---|---|---|---|
| `n128_d4` (spec) | 31 | 11.3 | 3.01 | `2^128.03` |
| `n128_d4` (reproduced) | 31 | 11.3 | 3.01 | `2^128.03` |
| `n256_d4` (spec) | 23 | 9.8 | 7.55 | `2^128.16` |
| `n256_d4` (reproduced) | 23 | 9.8 | 7.51 | `2^128.17` |

The default ring (`tau = 22`, bound 9.8, `2^103`) predates this rule.

Invertibility, the second constraint (Finding 6). `goldilocks-note/port-checks/fixed_weight_invertibility.py` now also scans `tau`. It reports the smallest `tau` with `|C| >= 2^128` and the smallest `tau` whose pessimistic estimate (`P1 = P0`) is at most `2^-100`, the note's target:

| phi | smallest `tau` for `2^128` | smallest `tau` for `Pr[non-invertible] <= 2^-100` |
|---|---|---|
| 96 | 39 (`2^128.95`) | 33 (`2^-100.41`) |
| 192 | 25 (`2^128.59`) | 32 (`2^-103.37`); 31 gives `2^-99.95` |

The op-norm distributions come from 10^6 attempts per `tau`, drawn through the sampler's own Fisher-Yates stream and evaluated on the sparse path. The code is `goldilocks-note/port-checks/op_norm_calibration.rs`, a test module to append to `short_challenge.rs`; its outputs are `op_norm_calibration_{k5,k6,n128,n256}.txt` next to it.

The chosen values:

- **`k = 5`: `tau = 39`, `op_norm_bound = 13.0`.** The set size binds. This gives `2^128.95` challenges before rejection, an acceptance rate of 0.527 (1.90 attempts) and `2^128.03` after rejection. `Pr[non-invertible]` lies in `[2^-139.3, 2^-115.0]`, and `gamma_C = 22.52`.
- **`k = 6`: `tau = 32`, `op_norm_bound = 11.2`.** Invertibility binds. The set (`2^153.11`) is then so large that the 128-bit rule would ask for a `2^-25` quantile, about `3.6 * 10^7` attempts. The bound is therefore the smallest one with at most 8 expected attempts, the precedent of `n256_d4` (7.55). That is an acceptance rate of 0.132 (7.57 attempts) and `2^150.19` after rejection. `Pr[non-invertible]` lies in `[2^-131.2, 2^-103.4]`, and `gamma_C = 19.40`.

The distribution of `||sigma(c)||_inf` before rejection (10^6 attempts):

| ring | `tau` | min | 1% | 10% | 50% | mean | 90% | 99% | max |
|---|---|---|---|---|---|---|---|---|---|
| `k = 5` | 39 | 8.25 | 10.16 | 11.21 | 12.89 | 13.10 | 15.25 | 17.77 | 25.46 |
| `k = 6` | 32 | 8.49 | 10.15 | 11.04 | 12.48 | 12.65 | 14.48 | 16.62 | 23.40 |

The coefficient operator norm `||c||_op` (power iteration on `M_c^T M_c`, 300 attempts per `tau`) is between 1.08 and 1.66 times `||sigma(c)||_inf`, with a mean of 1.35, at every `tau` tried for `k = 5` and `k = 6`. It always stays below `sqrt(3)`, and matches the note's 1.2-1.6 for uniform ternary challenges. On the power-of-two rings the ratio is 1.000, as expected.

Alternatives, recorded for the user. The `k = 5` rows follow the 128-bit rule, under which the bound falls as `tau` grows, at the cost of more attempts. The `k = 6` rows are the neighbouring bounds at `tau = 32` under the 8-attempt cap:

| `k` | `tau` | bound | attempts |
|---|---|---|---|
| 5 | 40 | 11.8 | 5.44 |
| 5 | 41 | 11.3 | 13.7 |
| 5 | 42 | 11.0 | 32 |
| 5 | 48 | 10.2 | about 5100 |
| 6 | 32 | 11.0 | 10.8 |
| 6 | 32 | 11.5 | 4.88 |

So `k = 5` at `tau = 40` would lower `gamma_C` by 9% (to 20.4) for 2.9 times the sampling cost. The plan's provisional `tau = 48` for `k = 5` would need about 5100 attempts to keep `2^128`.

**Challenge samplers at `p`.** The fold challenges and the field challenges are distinct samplers:

- Fold challenges: `sample_low_op_norm_ring_*` call `sample_short_challenge_into`, which draws fixed-weight ternary challenges and writes `-1` as `MOD_Q - 1`. They are not in the challenge field.
- Field challenges: `sample_ring_element_ntt_slots_*` sample a `FieldExtension` (3 residues mod `p`, uniform by rejection since Phase 1) and embed it on the homogenized diagonal `Z_p[tau]`.

The new test `hash::field_challenges_are_fixed_by_conjugation_and_fold_challenges_are_not` (trinomial only) runs both samplers through a transcript. The field challenges satisfy `conj(r) = r` (the plan's testing invariant). The fold challenges are ternary of weight `TAU` and are not fixed by conjugation. The invariant is specific to the trinomial family's `tau`-aligned diagonal, which is why the test is `cfg`-gated: it is new, and no existing test is hidden.

**Tests.**

- The four `short_challenge` tests that failed at `p` pass unchanged. Their power-of-two assumptions were inside the functions they call, not in the tests: `fft_op_norm_matches_direct_on_fixed_vector`, `samples_have_correct_weight_and_op_norm`, `parametrized_samples_have_correct_weight_and_op_norm` (weight 23, bound 8.357, which still rejects at `phi = 96` and `192`) and `sparse_op_norm_matches_dense_fft`.
- New tests, on every ring:
  - `evaluation_points_are_the_roots_of_the_modulus`: the ring's modulus vanishes (`< 1e-10`) at `zeta_m^j` for every `j` in `EVAL_EXPONENTS` and for its conjugate. That is `phi` roots of a monic polynomial of degree `phi`, so the modulus is `Phi_m` and these are all its roots. It does not vanish at any other `m`-th root of unity.
  - `canonical_embedding_is_multiplicative`: `(ab)(zeta^j) = a(zeta^j) b(zeta^j)` at every evaluation point, for the ring's own product of centered fixed-weight challenges.
  - `challenges_have_operator_norm_at_most_gamma_c`: for 16 sampled challenges, the power-iteration estimate of `||c||_op` (multiplication matrix built from ring products) is at most `GAMMA_C` and at most `sqrt(3) ||sigma(c)||_inf`. For the trinomial family the largest ratio `||c||_op / ||sigma(c)||_inf` exceeds 1.1, so the factor is needed. For `X^N + 1` it is at most 1 and `GAMMA_C = T_OP_NORM_BOUND`.
- Mutation checks at `k = 5`. Each mutation was applied, the `short_challenge` tests run, and the code restored (byte-compared against a copy). `challenge_stream_fingerprint` fails in every run, as it does without a mutation.
  - Phase table built for `m = N` instead of `3N`: fails `fft_op_norm_matches_direct_on_fixed_vector` and `challenges_have_operator_norm_at_most_gamma_c`.
  - Coprimality filter reduced to odd `j`: fails 7 tests, including `evaluation_points_are_the_roots_of_the_modulus`, `canonical_embedding_is_multiplicative` and `sparse_op_norm_matches_dense_fft`.
  - `sqrt(3)` dropped from `GAMMA_C`: fails `challenges_have_operator_norm_at_most_gamma_c`.
  - Sparse step `4p` instead of `6p`: fails 4 tests, including `sparse_op_norm_matches_dense_fft` and `samples_have_correct_weight_and_op_norm`.
  - Second index started at `7p` instead of `5p`: fails 3 tests, including `sparse_op_norm_matches_dense_fft`.

**`challenge_stream_fingerprint`.** The cause of this pre-existing failure is now established. `EXPECTED_FINGERPRINT` (`5670da5c…`) was pinned in `251db0d` with `T_OP_NORM_BOUND = 10.0`, and `4ee89b9` lowered the bound to 9.8 without re-pinning it. A temporary probe on the default ring hashed the stream of `sample_fixed_weight_challenge::<22>` at bound 10.0 and got exactly `5670da5c…`; at 9.8 it got `710f97b1…`, the value the test reports. The sampler algorithm has therefore not changed since it was pinned. The test is left untouched for the user to decide. Options:

- re-pin to the current stream;
- make the test sample with explicit parameters (`W = 22`, bound 10.0), which reproduces the original pin on `N = 128`.

Either way the fingerprint depends on `N`, `TAU` and the bound, so one constant cannot hold on every ring.

**Results.**

| Command | Result |
|---|---|
| `incomplete-rexl`, `cargo +nightly test` | 9 unit + 6 Goldilocks + 27 integration + 8 trinomial pass |
| `cargo +nightly test --lib` (debug), default ring | 187 pass, 1 fail (`challenge_stream_fingerprint`) |
| `ROKOKO_RING=rings/n128_d4.toml cargo +nightly test --release --lib` | 187 pass, 1 fail (the same) |
| `cargo +nightly run --release -- p-26` | `100.3199462890625 KB` and `112.0703125 KB`; repetition rate 2.18 as before |
| `ROKOKO_RING=rings/goldilocks_k5.toml cargo +nightly test --lib` (debug) | 187 pass, 8 fail |
| `ROKOKO_RING=rings/goldilocks_k6.toml cargo +nightly test --lib` (debug) | 187 pass, 8 fail (the same 8) |
| `cargo +nightly build --lib --features debug-hardness`, default and `k = 5` | builds |

The 3 new generic tests account for 184 to 187 on the existing rings. At `p` they and the new hash test take the count from 191 to 195. The 8 failures at `p` are `challenge_stream_fingerprint` (pre-existing, see above) and the 7 Phase 4 tests of the Phase 2 table:

- `project_fine`: `test_batch_projection`, `test_const_term_relation_to_prove`, `exact_projection_paths_match_the_lazy_kernels`;
- `executor`: `three_openings_round_trip`, `three_decomposition_chunks_round_trip`, still the projection-shape overrun at `project_fine.rs:552`;
- `snark`: `test_conj_segment_ct_zero_roundtrip`, `test_norm_claim_roundtrip`.

`decomposition::test_random_mod_q` did not flake in these runs.

**Deferred, and notes for Phases 4 and 5.**

- Phase 4: the norm-check factor (`Tr(v)/phi <= (3/2) beta^2`, extracted `sqrt(3) beta`) and the guard `3 r' beta'^2 < p/2` are separate from `gamma_C` and still to do. `debug_hardness.rs` now carries the challenge-side `sqrt(3)`, but not the norm-check one.
- Phase 5: the estimator must use `gamma_C = sqrt(3) * op_norm_bound` (22.5 at `k = 5`, 19.4 at `k = 6`), not the spec's `op_norm_bound`. If the parameters turn out sensitive to `gamma_C`, the alternatives above (`k = 5`, `tau = 40`, bound 11.8) are available.
- The Goldilocks specs fix `tau` and the bound; any later change also changes the fold-challenge transcripts at `p`.
- `benches/bench.rs` `op_norm_sq_sparse` uses 22 hard-coded positions up to 127, which panic at `N = 96`. It is a bench, not a test, and is left as is.
- Items to fold back into the note:
  - the fixed-weight `tau` per `k`;
  - the measured `||c||_op / ||sigma(c)||_inf` range for fixed-weight challenges (1.08-1.66, mean 1.35).

**Unverified.**

- The invertibility numbers are the plan's heuristic range (Finding 6), not a proof, and do not account for the op-norm rejection, which the note argues is negligible.
- The calibration rule is reconstructed from `fe7f4dd`'s values. The attempt cap of 8 for `k = 6` follows the `n256_d4` precedent (confirmed by the user, see below).
- The x86 paths were neither compiled nor run. The edit sits before the `#[cfg(avx512f)]` block, which is textually unchanged.
- Sampler performance at `p` was not benchmarked.

### Review fixes after Phase 3

A read-only review of the Phase 3 diff found no high- or medium-severity bug. It re-derived the evaluation set, the `sqrt(3)` factor and the calibration independently, and re-ran every command of the table above with the same counts.

- The Phase 0 note on `challenge_stream_fingerprint` gave `710f97b1…` for both rings; on `n128_d4` the stream hashes to `422675347b…`.
- The alternatives table's header now says that its `k = 6` rows are neighbouring bounds at `tau = 32`, not values of the 128-bit rule. The attempt count the rule would need at `k = 6` is about `3.6 * 10^7`, not `3 * 10^7`.
- In `debug_hardness.rs`, the `gamma_C` sentence moved after the justification of the inequality it qualifies.
- `cargo build --features debug-hardness` at `k = 5` stops on the binary's intended `compile_error!`; the `--lib` build is the one that applies, as the table says.

Decisions taken by the user:

- `k = 5` keeps `tau = 39` (bound 13.0). Revisit in Phase 5 against `tau = 40` (bound 11.8, 5.44 attempts) once the estimator shows the effect of `gamma_C`.
- `k = 6` keeps the 8-attempt cap of the `n256_d4` precedent (`tau = 32`, bound 11.2).
- `challenge_stream_fingerprint` stays failing, as on `main`; it is not part of the port.

Left as is: the bounds-checked indexing in `op_norm_sq_sparse_trinomial` (not benchmarked; Phase 7), and the test name `fft_op_norm_matches_direct_on_fixed_vector`, which no longer uses an FFT on the trinomial family.

### Alignment with the proposal (after Phase 3)

The plan above was rewritten around the maintainer's proposal, with the user's decisions: conductor 288 only, the guard constant 1.5, `n_rp = 384` kept provisional, `tau = 40`, SNARK mode out of scope, and a proof-hash check with the commits re-split into the proposal's PR stack. These supersede the Phase 3 decision to keep `tau = 39`.

**Changes.**

- `build.rs` accepts the trinomial family only at degree 96 (`TRINOMIAL_DEGREE`). `rings/goldilocks_k6.toml` is deleted, and the main crate's ring-vector test drops its `k = 6` arm. `incomplete-rexl` is unchanged: `TrinomialSlotRing` stays generic in `k`, and `tests/trinomial.rs` still checks `k = 5` and `k = 6` against both vector files.
- `rings/goldilocks_k5.toml`: `tau = 40`, `op_norm_bound = 12.4`.
  - `C(96, 40) * 2^40 = 2^130.47` challenges before rejection.
  - The calibration row at 12.4 (`10^6` attempts) gives an acceptance rate of 0.3284, so 3.04 attempts, and `2^128.86` challenges after rejection. Measured directly on the sampler: 3.053 attempts over `2 * 10^5` challenges.
  - `gamma_C = sqrt(3) * 12.4 = 21.477`.
  - `Pr[non-invertible difference]` in `[2^-140.72, 2^-117.14]`.
  - `||c||_op / ||sigma(c)||_inf` reaches at most 1.613 in `challenges_have_operator_norm_at_most_gamma_c`; the 300-sample calibration row gives 1.131 to 1.649, mean 1.354.
- SNARK mode is compiled out for the trinomial family. `pub mod snark` is under `cfg(not(rokoko_family = "trinomial"))`, and `--features snark` is a `compile_error!` there. Nothing outside SNARK mode used the module, apart from three SNARK-only pieces that are gated with it: `executor::execute_snark`, `wire::initial_{to,from}_bytes`, and `examples/claims.rs`, which is a `compile_error!` at `p`. `Role::WitnessEval` is kept, so the wire header has the same 16 models in every family. The default rings build and test exactly as before, with and without `--features snark`.
- **Proof-hash check**, `scripts/proof-hash.sh`.
  - `ROKOKO_PROOF_OUT=<path>` makes `executor::run` write the proof's wire bytes once they are serialised. This is the only change to the executor, kept as `scripts/proof-hash.patch`.
  - The script builds the base (by default the merge base with `origin/main`, `9023459`) in a temporary worktree with that patch applied, and nothing else changed. It builds the working tree too, each ring into its own target directory.
  - It runs each set twice per side, compares SHA-256 hashes, and exits 0 only if they all agree.
  - Usage: `./scripts/proof-hash.sh [--base=REV] [--rings=...] [--sets=p-26,p-28] [--runs=2] [--dir=PATH]`. About 8 minutes with the builds.

**Proof hashes against `9023459`.** Every run on both sides gave the same bytes, so proof generation is deterministic and the port leaves d2 and d4 byte-identical at the PCS sets.

| ring | set | SHA-256 (both sides) | bytes |
|---|---|---|---|
| d2 | `p-26` | `9d25e7ac77bc4d31…` | 114760 |
| d2 | `p-28` | `1492186a59a2f8e2…` | 114824 |
| d4 | `p-26` | `81483ff62903915d…` | 132165 |
| d4 | `p-28` | `9029df17a356e192…` | 132189 |

On the three changes of Finding 10, probed on these four runs:
- no Fiat-Shamir word was rejected;
- the verifier's radix weights (`load_combiner_evaluation_data`) are reached only by intermediate rounds, which these chains do not run;
- `expand_combination` is SNARK-only.

The hook hashes the round proof's wire bytes. The commitment enters them through the transcript, but is not compared byte for byte.

**Results.**

| Command | Result |
|---|---|
| `incomplete-rexl`, `cargo +nightly test` | 9 unit + 6 Goldilocks + 27 integration + 8 trinomial pass |
| `cargo +nightly test --lib` (debug), default ring | 187 pass, 1 fail (`challenge_stream_fingerprint`, as on `main`) |
| `ROKOKO_RING=rings/n128_d4.toml cargo +nightly test --release --lib` | 187 pass, 1 fail (the same) |
| `ROKOKO_RING=rings/goldilocks_k5.toml cargo +nightly test --lib` (debug) | 165 pass, 6 fail |
| `cargo +nightly run --release -- p-26` | `100.3199462890625 KB` and `112.0703125 KB` |
| `cargo +nightly build`, with and without `--features snark`, default ring | builds |
| `--features snark` and `cargo build --examples` at `k = 5` | stop on the intended `compile_error!` |

At `p` the count drops from 195 to 171 because the 24 `snark` tests no longer exist there; they still run on d2 and d4. Among them are the two that failed at `p` (`test_conj_segment_ct_zero_roundtrip`, `test_norm_claim_roundtrip`). The 6 failures are `challenge_stream_fingerprint` and the 5 PR 4 tests: `project_fine` `test_batch_projection`, `test_const_term_relation_to_prove` and `exact_projection_paths_match_the_lazy_kernels`, and `executor` `three_openings_round_trip` and `three_decomposition_chunks_round_trip`.

**Unverified.** The x86 paths, as before; none of these edits is arch-specific. The full `cargo test --release` with the binary's tests was not re-run.
