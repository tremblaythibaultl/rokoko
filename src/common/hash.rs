use crate::common::arithmetic::field_to_ring_element_into;
use crate::common::config::{DEGREE, MOD_Q};
use crate::common::ring_arithmetic::*;
use crate::common::short_challenge::sample_short_challenge_into;
use crate::common::sumcheck_element::SumcheckElement;
use crate::hexl::bindings::eltwise_reduce_mod;
use blake3::Hasher;

/// Incremental transcript hash suitable for Fiat–Shamir style challenges.
/// The transcript absorbs ring elements and can deterministically derive
/// pseudorandom bytes for challenges.
#[derive(Clone, Debug)]
pub struct HashWrapper {
    transcript: Hasher,
    sample_counter: u64,
}

impl Default for HashWrapper {
    fn default() -> Self {
        Self::new()
    }
}

static SEED: &[u8] = b"Smutno mi, Boze";

/// `2^64 - (2^64 mod q)`, the largest multiple of an odd `q > 1` that a `u64` draw can stay below.
/// A draw at or past it is rejected, and an accepted one reduces to a uniform residue. For
/// Goldilocks this is `p` itself (rejection `2^-32`). Below `2^50` the rejection rate is below
/// `2^-14` (`2^-38.6` for the default ring) and every accepted draw reduces exactly as the plain
/// `x mod q` did, so the transcripts of the existing rings only change on a rejection.
pub const fn rejection_bound(q: u64) -> u64 {
    0u64.wrapping_sub((u64::MAX % q + 1) % q)
}

const SAMPLE_BOUND: u64 = rejection_bound(MOD_Q);
const _: () = assert!(MOD_Q % 2 == 1 && MOD_Q > 1);

/// Replaces every word of `words` at or past [`rejection_bound`]`(q)` by the next accepted word of
/// `next`, in order: uniform words in, uniform words below a multiple of `q` out.
pub fn reject_past_bound(words: &mut [u64], q: u64, mut next: impl FnMut() -> u64) {
    let bound = rejection_bound(q);
    for word in words.iter_mut() {
        while *word >= bound {
            *word = next();
        }
    }
}

impl HashWrapper {
    #[cfg(feature = "debug-fs")]
    fn dbg_fs(&self, tag: &str, n: usize) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static C: AtomicUsize = AtomicUsize::new(0);
        eprintln!("FS[{}] {} n={}", C.fetch_add(1, Ordering::Relaxed), tag, n);
    }
    #[cfg(not(feature = "debug-fs"))]
    #[inline(always)]
    fn dbg_fs(&self, _tag: &str, _n: usize) {}

    pub fn new() -> Self {
        let mut transcript = Hasher::new();
        transcript.update(SEED);
        Self {
            transcript,
            sample_counter: 0,
        }
    }

    /// Absorb a ring element into the transcript.
    pub fn update_with_ring_element(&mut self, element: &RingElement) {
        self.dbg_fs("update_with_ring_element", 1);
        // hack: treat the u64 slice as raw bytes (native endianness)
        let ptr = element.v.as_ptr() as *const u8;
        let len = element.v.len() * std::mem::size_of::<u64>();
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        self.transcript.update(bytes);
    }

    pub fn update_with_ring_element_slice(&mut self, elements: &[RingElement]) {
        self.dbg_fs("update_with_ring_element_slice", elements.len());
        for element in elements {
            self.update_with_ring_element(element);
        }
    }

    pub fn update_with_field_extension_element(&mut self, element: &FieldExtension) {
        self.dbg_fs("update_with_field_extension_element", 1);
        // hack: treat the coeffs slice as raw bytes (native endianness)
        let ptr = element.coeffs.as_ptr() as *const u8;
        let len = element.coeffs.len() * std::mem::size_of::<u64>();
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        self.transcript.update(bytes);
    }

    pub fn update_with_field_extension_slice(&mut self, elements: &[FieldExtension]) {
        self.dbg_fs("update_with_field_extension_slice", elements.len());
        for element in elements {
            self.update_with_field_extension_element(element);
        }
    }

    pub fn update_with_bytes(&mut self, bytes: &[u8]) {
        self.dbg_fs("update_with_bytes", bytes.len());
        self.transcript.update(bytes);
    }

    pub fn update_with_u64(&mut self, value: u64) {
        self.dbg_fs("update_with_u64", 1);
        self.transcript.update(&value.to_le_bytes());
    }

    /// Derive `len` bytes of pseudorandomness from the current transcript.
    pub fn sample_bytes(&mut self, len: usize) -> Vec<u8> {
        self.dbg_fs("sample_bytes", len);
        let mut out = vec![0u8; len];
        self.fill_from_xof(b"bytes", &mut out);
        out
    }

    /// Convenience helper to sample a u64 challenge.
    pub fn sample_u64(&mut self) -> u64 {
        self.dbg_fs("sample_u64", 1);
        let mut buf = [0u8; 8];
        self.fill_from_xof(b"u64", &mut buf);
        u64::from_le_bytes(buf)
    }

    /// A uniform residue: [`Self::sample_u64`] until a draw lands below [`rejection_bound`].
    pub fn sample_u64_mod_q(&mut self) -> u64 {
        self.dbg_fs("sample_u64_mod_q", 1);
        loop {
            let x = self.sample_u64();
            if x < SAMPLE_BOUND {
                return x % MOD_Q;
            }
        }
    }

    pub fn fill_from_xof(&mut self, label: &[u8], out: &mut [u8]) {
        self.xof(label).fill(out);
    }

    fn xof(&mut self, label: &[u8]) -> blake3::OutputReader {
        let mut state = self.transcript.clone();
        state.update(&self.sample_counter.to_le_bytes());
        state.update(label);
        self.sample_counter += 1;

        state.finalize_xof()
    }

    /// Uniform residues mod q into `out`, from one XOF stream: the words are read in order (native
    /// endianness), and a word past [`rejection_bound`] is replaced by the next word of the same
    /// stream, so without rejections the output is the plain reduction of the first `out.len()`
    /// words.
    fn fill_residues_from_xof(&mut self, label: &[u8], out: &mut [u64]) {
        let mut xof = self.xof(label);
        xof.fill(unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, std::mem::size_of_val(out))
        });
        reject_past_bound(out, MOD_Q, || {
            let mut word = [0u8; 8];
            xof.fill(&mut word);
            u64::from_ne_bytes(word)
        });
        unsafe { eltwise_reduce_mod(out.as_mut_ptr(), out.as_mut_ptr(), out.len() as u64, MOD_Q) };
    }

    #[allow(dead_code)]
    fn sample_binary_ring_element(&mut self) -> RingElement {
        let mut buf = [0u8; DEGREE / 8];
        self.fill_from_xof(b"binary-ring-element", &mut buf);
        let mut element = RingElement::new(Representation::Coefficients);
        for i in 0..DEGREE / 8 {
            let b = buf[i];
            for j in 0..8 {
                element.v[i * 8 + j] = ((b >> j) & 1) as u64;
            }
        }
        element
    }
    #[allow(dead_code)]
    fn sample_biased_ternary_ring_element(&mut self) -> RingElement {
        let el1 = self.sample_binary_ring_element();
        let el2 = self.sample_binary_ring_element();
        let mut result = RingElement::new(Representation::Coefficients);
        subtraction(&mut result, &el1, &el2);
        result
    }

    fn sample_biased_ternary_ring_element_into(&mut self, output: &mut RingElement) {
        let mut buf = [0u8; DEGREE / 4];
        self.fill_from_xof(b"biased-ternary-ring-element", &mut buf);
        output.to_representation(Representation::Coefficients);
        for i in 0..DEGREE / 4 {
            let b = buf[i];
            for j in 0..4 {
                let bits = (b >> (j * 2)) & 0b11;
                let coeff = match bits {
                    0b01 => 1,
                    0b10 => MOD_Q - 1,
                    _ => 0,
                };
                output.v[i * 4 + j] = coeff;
            }
        }
        output.to_representation(Representation::IncompleteNTT);
    }

    pub fn sample_ring_element_into(&mut self, output: &mut RingElement) {
        self.dbg_fs("sample_ring_element_into", 1);
        self.fill_residues_from_xof(b"ring-element", &mut output.v);
    }

    pub fn sample_field_element_into(&mut self, output: &mut FieldExtension) {
        self.dbg_fs("sample_field_element_into", 1);
        self.fill_residues_from_xof(b"field-element", &mut output.coeffs);
    }
    pub fn sample_ring_element_vec_into(&mut self, output: &mut [RingElement]) {
        self.dbg_fs("sample_ring_element_vec_into", output.len());
        for element in output.iter_mut() {
            self.sample_ring_element_into(element);
        }
    }

    pub fn sample_ring_element_ntt_slots_same_vec_into(&mut self, output: &mut [RingElement]) {
        self.dbg_fs("sample_ring_element_ntt_slots_same_vec_into", output.len());
        let mut f = FieldExtension::zero();
        for element in output.iter_mut() {
            self.sample_field_element_into(&mut f);
            field_to_ring_element_into(&mut *element, &f);
            element.from_homogenized_field_extensions_to_incomplete_ntt();
        }
    }

    pub fn sample_ring_element_ntt_slots_into(&mut self, output: &mut RingElement) {
        self.dbg_fs("sample_ring_element_ntt_slots_into", 1);
        let mut f = FieldExtension::zero();
        self.sample_field_element_into(&mut f);
        field_to_ring_element_into(output, &f);
        output.from_homogenized_field_extensions_to_incomplete_ntt();
    }

    pub fn sample_biased_ternary_ring_element_vec_into(&mut self, output: &mut [RingElement]) {
        self.dbg_fs("sample_biased_ternary_ring_element_vec_into", output.len());
        for element in output.iter_mut() {
            self.sample_biased_ternary_ring_element_into(element);
        }
    }
    pub fn sample_low_op_norm_ring_into(&mut self, output: &mut RingElement) {
        self.dbg_fs("sample_low_op_norm_ring_into", 1);
        sample_short_challenge_into(self, output);
        // self.sample_biased_ternary_ring_element_into(output);
    }

    pub fn sample_low_op_norm_ring_vec_into(&mut self, output: &mut [RingElement]) {
        self.dbg_fs("sample_low_op_norm_ring_vec_into", output.len());
        for element in output.iter_mut() {
            self.sample_low_op_norm_ring_into(element);
        }
    }

    #[allow(dead_code)]
    fn sample_ternary_ring_element(&mut self) -> RingElement {
        // we need 2 bits per coefficient to sample uniformly from {-1, 0, 1}
        // 0b11 is discarded so we take (two times) more bytes to account for the rejections
        let mut buf = [0u8; DEGREE / 2];
        self.fill_from_xof(b"ternary-ring-element", &mut buf);
        let mut element = RingElement::new(Representation::Coefficients);
        let mut coeff_index = 0;
        for i in 0..DEGREE / 2 {
            let b = buf[i];
            for j in 0..4 {
                if coeff_index >= DEGREE {
                    break;
                }
                let bits = (b >> (j * 2)) & 0b11;
                if bits == 0b11 {
                    continue; // reject
                }
                let coeff = match bits {
                    0b00 => 0,
                    0b01 => 1,
                    0b10 => MOD_Q - 1,
                    _ => unreachable!(),
                };
                element.v[coeff_index] = coeff;
                coeff_index += 1;
            }
        }
        if coeff_index < DEGREE {
            // we didn't fill all coefficients due to rejections, sample recursively
            // practically this will almost never recurse
            return self.sample_ternary_ring_element();
        }
        element
    }
}

#[cfg(test)]
mod tests {
    use crate::common::config::MOD_Q;

    use super::*;

    #[test]
    fn deterministic_given_same_transcript() {
        let mut t1 = HashWrapper::new();
        let mut t2 = HashWrapper::new();

        let mut e = RingElement::new(Representation::Coefficients);
        e.v[0] = 42;
        e.v[5] = 7;

        t1.update_with_ring_element(&e);
        t2.update_with_ring_element(&e);

        let c1 = t1.sample_bytes(32);
        let c2 = t2.sample_bytes(32);
        debug_assert_eq!(c1, c2);
    }

    #[test]
    fn deterministic_given_different_transcript_order() {
        let mut t1 = HashWrapper::new();
        let mut t2 = HashWrapper::new();

        let mut e1 = RingElement::new(Representation::Coefficients);
        e1.v[0] = 1;
        let mut e2 = RingElement::new(Representation::Coefficients);
        e2.v[0] = 2;

        t1.update_with_ring_element(&e1);
        t1.update_with_ring_element(&e2);

        t2.update_with_ring_element(&e2);
        t2.update_with_ring_element(&e1);

        let c1 = t1.sample_bytes(32);
        let c2 = t2.sample_bytes(32);
        debug_assert_ne!(c1, c2);
    }

    #[test]
    fn same_transcript_sampled_twice_yield_different_outputs() {
        let mut t1 = HashWrapper::new();

        let challenge1 = t1.sample_biased_ternary_ring_element();
        let challenge2 = t1.sample_biased_ternary_ring_element();
        debug_assert_ne!(challenge1, challenge2);
    }

    #[test]
    fn transcript_update_changes_output() {
        let mut t1 = HashWrapper::new();
        let mut t2 = HashWrapper::new();

        let mut e = RingElement::new(Representation::Coefficients);
        e.v[0] = 1;
        t1.update_with_ring_element(&e);

        let challenge1 = t1.sample_bytes(16);
        let challenge2 = t2.sample_bytes(16);
        debug_assert_ne!(challenge1, challenge2);
    }

    #[test]
    fn counter_produces_fresh_challenges() {
        let mut transcript = HashWrapper::new();
        let first = transcript.sample_bytes(8);
        let second = transcript.sample_bytes(8);
        debug_assert_ne!(first, second);
    }

    #[test]
    fn sample_binary_ring_element_works() {
        let mut transcript = HashWrapper::new();
        let element = transcript.sample_binary_ring_element();
        for &coeff in element.v.iter() {
            debug_assert!(coeff == 0 || coeff == 1);
        }
    }

    #[test]
    fn sample_biased_ternary_ring_element_works() {
        let mut transcript = HashWrapper::new();
        let element = transcript.sample_biased_ternary_ring_element();
        for &coeff in element.v.iter() {
            debug_assert!(coeff == MOD_Q - 1 || coeff == 0 || coeff == 1);
        }
    }

    #[test]
    fn sample_ternary_ring_element_works() {
        let mut transcript = HashWrapper::new();
        let element = transcript.sample_ternary_ring_element();
        for &coeff in element.v.iter() {
            debug_assert!(coeff == MOD_Q - 1 || coeff == 0 || coeff == 1);
        }
    }

    #[test]
    fn sample_ring_element_works() {
        let mut transcript = HashWrapper::new();
        let mut element = RingElement::new(Representation::IncompleteNTT);
        transcript.sample_ring_element_into(&mut element);
        let mut found_nonzero = false;
        for &coeff in element.v.iter() {
            if coeff != 0 {
                found_nonzero = true;
                break;
            }
        }
        debug_assert!(found_nonzero);
    }

    #[test]
    fn rejection_bound_is_the_largest_multiple_of_q() {
        const P: u64 = incomplete_rexl::GOLDILOCKS_P;
        assert_eq!(rejection_bound(P), P);
        for q in [MOD_Q, 3, (1 << 63) - 25, u64::MAX - 58, P] {
            let bound = rejection_bound(q) as u128;
            assert_eq!(bound % q as u128, 0);
            assert!((1u128 << 64) - bound < q as u128);
        }
    }

    #[test]
    fn rejected_words_are_replaced_in_order() {
        const P: u64 = incomplete_rexl::GOLDILOCKS_P;
        let mut words = [0, P - 1, P, u64::MAX, 5, P + 7];
        let mut stream = [u64::MAX, 11, P, 12, 13].into_iter();
        reject_past_bound(&mut words, P, || stream.next().unwrap());
        assert_eq!(words, [0, P - 1, 11, 12, 5, 13]);

        let bound = rejection_bound(MOD_Q);
        let mut words = [0, MOD_Q, bound - 1, bound, u64::MAX];
        let mut stream = [bound + 1, 7, 8].into_iter();
        reject_past_bound(&mut words, MOD_Q, || stream.next().unwrap());
        assert_eq!(words, [0, MOD_Q, bound - 1, 7, 8]);
    }

    /// Below the bound, uniform sampling is the plain reduction of the XOF words it always was.
    #[test]
    fn accepted_ring_samples_are_the_plain_reduction() {
        let bound = rejection_bound(MOD_Q);
        let mut checked = 0;
        for seed in 0..16u64 {
            let mut transcript = HashWrapper::new();
            transcript.update_with_u64(seed);
            let mut plain = transcript.clone();
            let mut words = [0u64; DEGREE];
            plain.fill_from_xof(b"ring-element", unsafe {
                std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, 8 * DEGREE)
            });
            let mut element = RingElement::new(Representation::IncompleteNTT);
            transcript.sample_ring_element_into(&mut element);
            assert!(element.v.iter().all(|&c| c < MOD_Q));
            if words.iter().all(|&w| w < bound) {
                assert!(element.v.iter().zip(&words).all(|(&c, &w)| c == w % MOD_Q));
                checked += 1;
            }
            assert_eq!(transcript.sample_u64(), plain.sample_u64());
        }
        assert!(checked > 0);
    }
}
