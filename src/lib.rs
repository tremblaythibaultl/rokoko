#![feature(coerce_unsized, unsize)]
#![allow(static_mut_refs)] // keep to suppress warning

// `build.rs` accepts the trinomial family (`family = "trinomial"`, `Z_p[X]/(X^96 - X^48 + 1)` over
// Goldilocks) and emits its constants, but the library does not implement that ring yet: its
// degree-3 slots, conjugation, trace and 64-bit modulus. The modules are compiled out for it so
// that the build stops on this one error.
#[cfg(rokoko_family = "trinomial")]
compile_error!(
    "the trinomial (Goldilocks) ring family is not implemented by the library yet; \
     build with a power-of-two ring spec (`family = \"power_of_two\"`, the default)"
);

#[cfg(not(rokoko_family = "trinomial"))]
pub mod common;
#[cfg(not(rokoko_family = "trinomial"))]
pub mod hexl;
#[cfg(not(rokoko_family = "trinomial"))]
pub mod protocol;
#[cfg(not(rokoko_family = "trinomial"))]
pub mod tracing;
