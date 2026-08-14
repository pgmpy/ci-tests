//! Lazy Gaussian sufficient statistics for the continuous tests.
//!
//! On the first continuous query a [`GramCache`] is built over **all**
//! continuous columns of the [`Dataset`]: per-column means plus the centered
//! cross-product matrix `S` (`S[i][j] = Σ_r (x_ri − m_i)(x_rj − m_j)`). Every
//! (partial) correlation then reduces to a gather + Cholesky solve on the
//! `(|Z|+2)²` submatrix — O(|Z|³) per query, flat in the number of rows — the
//! same sufficient-statistic trick as pcalg's `gaussCItest`, but automatic.
//!
//! Centered cross-products are algebraically identical to residual regression
//! on `[1, Z]` (the intercept is the centering), so this is the standard
//! partial-correlation computation expressed in sufficient statistics rather
//! than a distinct approximation of it.

use crate::dataset::Dataset;
use crate::error::CiError;

/// Maximum number of continuous columns for which the matrix is built:
/// `2048² × 8 B = 32 MiB`. This is a hard limit, not a switch to another
/// algorithm — a wider dataset makes the continuous tests return
/// [`crate::error::CiError::DegenerateData`]. The cap also bounds the one-shot
/// `O(n·p²)` build cost paid on the first continuous query, which is then
/// amortized across every subsequent query.
pub(crate) const MAX_GRAM_COLS: usize = 2048;

/// Means + centered cross-product matrix over the continuous columns.
#[derive(Debug, Clone)]
pub(crate) struct GramCache {
    /// Global column index -> dense continuous index (None for discrete).
    col_to_dense: Vec<Option<u32>>,
    /// Number of continuous columns (`p`).
    p: usize,
    /// `p × p` centered cross-products, row-major, symmetric.
    s: Vec<f64>,
}

impl GramCache {
    /// Build the cache, or `None` when there is nothing to cache (no rows, no
    /// continuous columns) or the dataset is too wide (`p > MAX_GRAM_COLS`).
    pub(crate) fn build(data: &Dataset) -> Option<Self> {
        let n = data.n_rows();
        let n_cols = data.n_cols();
        let mut col_to_dense = vec![None; n_cols];
        let mut continuous: Vec<&[f64]> = Vec::new();
        for (col, slot) in col_to_dense.iter_mut().enumerate() {
            if let Some(values) = data.continuous_values(col) {
                *slot = Some(u32::try_from(continuous.len()).expect("p bounded by MAX_GRAM_COLS"));
                continuous.push(values);
            }
        }
        let p = continuous.len();
        if n == 0 || p == 0 || p > MAX_GRAM_COLS {
            return None;
        }

        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;
        let means: Vec<f64> = continuous
            .iter()
            .map(|col| col.iter().sum::<f64>() / n_f)
            .collect();

        // `s[i][j]` only ever reads columns `i` and `j`, so a non-finite value
        // in some other column cannot affect a query on disjoint columns.
        let mut s = vec![0.0; p * p];
        for i in 0..p {
            let xi = continuous[i];
            let mi = means[i];
            for j in i..p {
                let xj = continuous[j];
                let mj = means[j];
                let acc: f64 = xi
                    .iter()
                    .zip(xj)
                    .map(|(&vi, &vj)| (vi - mi) * (vj - mj))
                    .sum();
                s[i * p + j] = acc;
                s[j * p + i] = acc;
            }
        }
        Some(Self { col_to_dense, p, s })
    }

    /// Dense continuous index of global column `col`, or an error mirroring
    /// [`Dataset::continuous`] when the column is discrete.
    fn dense(&self, col: usize) -> Result<usize, CiError> {
        match self.col_to_dense.get(col) {
            Some(Some(d)) => Ok(*d as usize),
            Some(None) => Err(CiError::WrongColumnKind(format!(
                "column {col} is discrete but a continuous column was required"
            ))),
            None => Err(CiError::UnknownColumn(format!("column index {col}"))),
        }
    }

    #[inline]
    fn s_at(&self, i: usize, j: usize) -> f64 {
        self.s[i * self.p + j]
    }

    /// Centered (co)variances of `x` and `y` given `z`:
    /// `(s_xx, s_yy, a, b, c)` where `a = S_xx·Z`, `b = S_yy·Z`, `c = S_xy·Z`
    /// are the Schur complements after eliminating `Z` (with empty `z`,
    /// `a = s_xx`, `b = s_yy`, `c = s_xy`).
    ///
    /// # Errors
    ///
    /// [`CiError::WrongColumnKind`] for a discrete column, and
    /// [`CiError::Numeric`] when `S_zz` is not positive definite
    /// (rank-deficient / collinear conditioning set).
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        reason = "x, y, z, k, l, u, v, a, b, c are the standard linear-algebra / CI variable names"
    )]
    pub(crate) fn schur_xy_given_z(
        &self,
        x: usize,
        y: usize,
        z: &[usize],
    ) -> Result<(f64, f64, f64, f64, f64), CiError> {
        let dx = self.dense(x)?;
        let dy = self.dense(y)?;
        let dz: Vec<usize> = z
            .iter()
            .map(|&zi| self.dense(zi))
            .collect::<Result<_, _>>()?;

        let s_xx = self.s_at(dx, dx);
        let s_yy = self.s_at(dy, dy);
        let s_xy = self.s_at(dx, dy);
        let k = dz.len();
        if k == 0 {
            return Ok((s_xx, s_yy, s_xx, s_yy, s_xy));
        }

        let rank_deficient =
            || CiError::Numeric("rank-deficient design matrix in partial correlation".to_string());

        // In-place Cholesky of the k×k S_zz gather (lower triangle).
        let mut l = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                l[i * k + j] = self.s_at(dz[i], dz[j]);
            }
        }
        for j in 0..k {
            let mut diag = l[j * k + j];
            for t in 0..j {
                diag -= l[j * k + t] * l[j * k + t];
            }
            // Deliberately an *absolute* (not relative) threshold: it errors
            // only for exact-or-rounding-singular S_zz. A relative threshold
            // would reject merely near-singular Z, for which a large finite r
            // is the correct answer. Downstream, the relative
            // `VARIANCE_REL_EPS` Schur check is the degeneracy gate.
            if diag <= 0.0 {
                return Err(rank_deficient());
            }
            let pivot = diag.sqrt();
            l[j * k + j] = pivot;
            for i in (j + 1)..k {
                let mut v = l[i * k + j];
                for t in 0..j {
                    v -= l[i * k + t] * l[j * k + t];
                }
                l[i * k + j] = v / pivot;
            }
        }

        // Forward-solve L·u = S_zx and L·v = S_zy.
        let mut u = vec![0.0; k];
        let mut v = vec![0.0; k];
        for i in 0..k {
            let mut ui = self.s_at(dz[i], dx);
            let mut vi = self.s_at(dz[i], dy);
            for t in 0..i {
                ui -= l[i * k + t] * u[t];
                vi -= l[i * k + t] * v[t];
            }
            u[i] = ui / l[i * k + i];
            v[i] = vi / l[i * k + i];
        }

        let a = s_xx - u.iter().map(|w| w * w).sum::<f64>();
        let b = s_yy - v.iter().map(|w| w * w).sum::<f64>();
        let c = s_xy - u.iter().zip(&v).map(|(ui, vi)| ui * vi).sum::<f64>();
        Ok((s_xx, s_yy, a, b, c))
    }
}
