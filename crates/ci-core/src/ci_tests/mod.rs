//! Concrete conditional-independence tests.
//!
//! Five discrete power-divergence tests ([`ChiSquared`], [`LogLikelihood`],
//! [`CressieRead`], [`FreemanTukey`], [`ModifiedLikelihood`]) share the
//! λ-parameterized discrete back-end via [`discrete_common`]; two continuous
//! tests ([`PearsonCorrelation`], [`PearsonEquivalence`]) cover the
//! partial-correlation path.

mod discrete_common;

pub mod chi_squared;
pub mod cressie_read;
pub mod freeman_tukey;
pub mod log_likelihood;
pub mod modified_likelihood;
pub mod pearson_correlation;
pub mod pearson_equivalence;

pub use chi_squared::ChiSquared;
pub use cressie_read::CressieRead;
pub use freeman_tukey::FreemanTukey;
pub use log_likelihood::LogLikelihood;
pub use modified_likelihood::ModifiedLikelihood;
pub use pearson_correlation::PearsonCorrelation;
pub use pearson_equivalence::PearsonEquivalence;
