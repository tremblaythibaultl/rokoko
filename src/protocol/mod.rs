pub mod boundary;
pub mod commitment;
pub mod crs;
// pub mod execution_simple;
pub mod config;
pub mod config_generator;
pub mod evaluation_point_sampler;
pub mod fold;
pub mod intermediate_sumchecks;
pub mod open;
pub mod params;
pub mod parties;
pub mod project_coarse;
pub mod project_fine;
// SNARK mode and its exact-norm chains are out of scope for the trinomial (Goldilocks) family:
// only the PCS chains are ported. Nothing outside SNARK mode uses this module.
#[cfg(not(rokoko_family = "trinomial"))]
pub mod snark;
#[cfg(all(feature = "snark", rokoko_family = "trinomial"))]
compile_error!("SNARK mode (`--features snark`) is out of scope for the trinomial (Goldilocks) family");
pub mod sumcheck;
pub mod sumcheck_utils;
pub mod sumchecks;
pub mod wire;
