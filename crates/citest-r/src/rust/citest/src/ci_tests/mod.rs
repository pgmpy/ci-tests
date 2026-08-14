//! Concrete conditional-independence tests.
//!
//! Five discrete power-divergence tests ([`ChiSquared`], [`LogLikelihood`],
//! [`CressieRead`], [`FreemanTukey`], [`ModifiedLikelihood`]) differ only in
//! their `λ` and are declared together by one macro; three
//! continuous tests ([`PearsonCorrelation`], [`FisherZ`],
//! [`PearsonEquivalence`]) share the partial-correlation path.

mod discrete_common;
mod fisher_z;
mod pearson_correlation;
mod pearson_equivalence;
mod power_divergence;

pub use fisher_z::FisherZ;
pub use pearson_correlation::PearsonCorrelation;
pub use pearson_equivalence::PearsonEquivalence;
pub use power_divergence::{
    ChiSquared, CressieRead, FreemanTukey, LogLikelihood, ModifiedLikelihood,
};
