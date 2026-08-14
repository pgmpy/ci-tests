//! Discrete contingency-table machinery operating on factorized `Dataset` codes.
//!
//! This is the discrete back-end for the power-divergence family of
//! conditional-independence tests (see [`crate::ci_tests`]): it builds `(X, Y)`
//! count tables from integer codes, applies the power-divergence statistic for a
//! given parameter `λ` (with optional Yates' continuity correction on 2×2
//! tables), and aggregates over the strata defined by the conditioning set `Z`.
//!
//! The statistic matches `scipy.stats.chi2_contingency(table, lambda_=λ,
//! correction=yates)`, which is how the golden fixture was generated. For a table
//! with observed counts `O`, marginals `R_i`, `C_j`, total `N` and expected
//! `E_ij = R_i * C_j / N`:
//!
//! - `λ = 0`   (log-likelihood / G-test): `2 * Σ O·ln(O/E)`.
//! - `λ = −1`  (modified log-likelihood): `2 * Σ E·ln(E/O)`.
//! - otherwise (incl. `λ = 1`, `2/3`, `−1/2`):
//!   `(2 / (λ·(λ+1))) · Σ ( O^(λ+1)/E^λ − O )`.
//!
//! The `O^(λ+1)/E^λ − O` form is used (rather than `O·((O/E)^λ − 1)`) because it
//! is `O = 0`-safe for every `λ > −1`: at `O = 0` the term is `0` instead of the
//! `0·∞ = NaN` the naive form would produce for `λ < 0`.
//!
//! ## Conditional stratification strategy
//!
//! Stratification is split into a build phase and a consume phase so that the
//! grouping work can be cached on the [`crate::dataset::Dataset`]. The build
//! phase ([`build_strata_partition`]) produces a [`StrataPartition`] — a
//! counting-sort permutation of the row indices by their Z-code combination —
//! and the consume phase ([`power_divergence_conditional`]) sweeps the
//! partition, tabulating each stratum in turn. When the mixed-radix product
//! `∏ k_zi` overflows `usize` (very many / very-high-cardinality Z columns)
//! the builder falls back to a [`HashMap`]-based grouping internally; the API
//! is unchanged.

/// The parameter `λ` selecting a member of the power-divergence family.
const LAMBDA_LOG_LIKELIHOOD: f64 = 0.0;
/// The parameter `λ` for the modified log-likelihood (Neyman) statistic.
const LAMBDA_MODIFIED: f64 = -1.0;

/// A dense `rows × cols` contingency table of counts, stored row-major.
pub(crate) struct ContingencyTable {
    rows: usize,
    cols: usize,
    counts: Vec<f64>,
}

impl ContingencyTable {
    /// Allocate a zeroed `rows × cols` table.
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            counts: vec![0.0; rows * cols],
        }
    }

    #[inline]
    fn add(&mut self, r: usize, c: usize) {
        self.counts[r * self.cols + c] += 1.0;
    }

    /// Zero the table for reuse across strata.
    fn reset(&mut self) {
        self.counts.fill(0.0);
    }

    /// Iterate over the rows of the table as count slices.
    fn rows(&self) -> impl Iterator<Item = &[f64]> {
        self.counts.chunks_exact(self.cols)
    }
}

/// Per-table result: `(statistic, degrees_of_freedom)`.
type TableStat = (f64, usize);

/// One cell's contribution to the power-divergence statistic, given the
/// (possibly Yates-corrected) observed count `observed` and expected `expected`.
///
/// `expected` is guaranteed `> 0` by the caller (active marginals and total are
/// positive). Returns `f64::INFINITY` for the `λ = −1`, `O = 0`, `E > 0` cell,
/// which propagates to a `+∞` statistic and hence `p = 0`.
#[inline]
fn power_divergence_term(observed: f64, expected: f64, lambda: f64) -> f64 {
    if lambda == LAMBDA_LOG_LIKELIHOOD {
        // 2 * O * ln(O / E), with O*ln(O) -> 0 at O = 0.
        if observed == 0.0 {
            0.0
        } else {
            2.0 * observed * (observed / expected).ln()
        }
    } else if lambda == LAMBDA_MODIFIED {
        // 2 * E * ln(E / O); at O = 0 (E > 0) this is +inf.
        if observed == 0.0 {
            f64::INFINITY
        } else {
            2.0 * expected * (expected / observed).ln()
        }
    } else {
        // (2 / (λ(λ+1))) * ( O^(λ+1)/E^λ − O ).
        // O = 0-safe for λ > −1: the bracket is 0 at O = 0.
        let bracket = observed.powf(lambda + 1.0) / expected.powf(lambda) - observed;
        (2.0 / (lambda * (lambda + 1.0))) * bracket
    }
}

/// Power-divergence statistic and dof for one table, applying Yates' continuity
/// correction on 2×2 tables when `yates` is set.
///
/// Returns `None` if the table is degenerate for testing: fewer than 2 active
/// rows/columns, or a zero total (the stratum should then be skipped). dof is
/// computed over the *active* sub-table (rows/columns with a non-zero marginal),
/// matching scipy/pgmpy which drop empty categories.
fn power_divergence_table(table: &ContingencyTable, lambda: f64, yates: bool) -> Option<TableStat> {
    // Marginals over the global shape.
    let mut row_sums = vec![0.0; table.rows];
    let mut col_sums = vec![0.0; table.cols];
    for (row, row_sum) in table.rows().zip(row_sums.iter_mut()) {
        for (&v, col_sum) in row.iter().zip(col_sums.iter_mut()) {
            *row_sum += v;
            *col_sum += v;
        }
    }
    let total: f64 = row_sums.iter().sum();
    if total == 0.0 {
        return None;
    }

    // Active rows/cols are those with a non-zero marginal.
    let active_rows = row_sums.iter().filter(|&&s| s > 0.0).count();
    let active_cols = col_sums.iter().filter(|&&s| s > 0.0).count();
    if active_rows < 2 || active_cols < 2 {
        return None;
    }

    let dof = (active_rows - 1) * (active_cols - 1);
    // Yates' correction applies to the active 2×2 table (dof == 1), for all λ.
    let use_yates = yates && dof == 1;

    let mut statistic = 0.0;
    for (row, &row_sum) in table.rows().zip(row_sums.iter()) {
        if row_sum == 0.0 {
            continue;
        }
        for (&observed, &col_sum) in row.iter().zip(col_sums.iter()) {
            if col_sum == 0.0 {
                continue;
            }
            let expected = row_sum * col_sum / total;
            // Active marginals are > 0 and total > 0, so expected > 0.
            let corrected = if use_yates {
                // Yates: shrink O toward E by clamping (O − E) to [−0.5, 0.5].
                observed - (observed - expected).clamp(-0.5, 0.5)
            } else {
                observed
            };
            statistic += power_divergence_term(corrected, expected, lambda);
        }
    }

    Some((statistic, dof))
}

/// Rows grouped by the observed combinations of a conditioning set `Z`:
/// `rows[starts[s] .. starts[s + 1]]` are the row indices of stratum `s`
/// (original row order within each stratum). Built once per distinct `Z` and
/// cached on the [`crate::dataset::Dataset`].
#[derive(Debug)]
pub(crate) struct StrataPartition {
    pub(crate) starts: Vec<u32>,
    pub(crate) rows: Vec<u32>,
}

impl StrataPartition {
    /// Number of observed strata.
    fn n_strata(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// Approximate heap size, for the cache budget.
    pub(crate) fn approx_bytes(&self) -> usize {
        4 * (self.starts.len() + self.rows.len())
    }
}

/// Result of the discrete power-divergence test.
pub(crate) struct DiscreteOutcome {
    pub statistic: f64,
    pub dof: usize,
}

/// Unconditional `(X, Y)` power-divergence statistic for parameter `lambda`,
/// with Yates' correction when `yates` is set. Returns a zero statistic with
/// `dof == 0` when the table is degenerate (single active row/column).
pub(crate) fn power_divergence_unconditional(
    x_codes: &[usize],
    y_codes: &[usize],
    kx: usize,
    ky: usize,
    lambda: f64,
    yates: bool,
) -> DiscreteOutcome {
    let mut table = ContingencyTable::zeros(kx, ky);
    for (&xc, &yc) in x_codes.iter().zip(y_codes) {
        table.add(xc, yc);
    }
    let (statistic, dof) = power_divergence_table(&table, lambda, yates).unwrap_or((0.0, 0));
    DiscreteOutcome { statistic, dof }
}

/// When the folded space (`∏ k_zi`) fits within this many times the row count
/// plus slack, use a flat `u32::MAX`-sentinel remap array instead of a `HashMap`.
const DENSE_REMAP_SLACK: usize = 4096;

/// Mixed-radix strides over the Z cardinalities, and the total `∏ k_zi`.
/// `None` if the product overflows `usize` (caller falls back to hashing).
fn fold_strides(z_columns: &[(&[usize], usize)]) -> Option<(Vec<usize>, usize)> {
    let mut strides = Vec::with_capacity(z_columns.len());
    let mut acc: usize = 1;
    for &(_, card) in z_columns {
        strides.push(acc);
        // card >= 1 whenever there are rows; max(1) keeps n == 0 safe.
        acc = acc.checked_mul(card.max(1))?;
    }
    Some((strides, acc))
}

/// Map each key to a first-seen dense id in `[0, n_strata)`, pushing the ids
/// onto `dense_ids`; returns the number of distinct keys (`n_strata`).
fn densify_hashed<K: Eq + std::hash::Hash>(
    keys: impl IntoIterator<Item = K>,
    dense_ids: &mut Vec<u32>,
) -> usize {
    let mut remap: std::collections::HashMap<K, u32> = std::collections::HashMap::new();
    for key in keys {
        let next = u32::try_from(remap.len()).expect("strata bounded by row count");
        dense_ids.push(*remap.entry(key).or_insert(next));
    }
    remap.len()
}

/// Group rows by the combination of the `Z` columns' codes. Strata are
/// identified by a mixed-radix fold (no per-row allocation); if `∏ k_zi`
/// overflows `usize` (only reachable with very many / very-high-cardinality Z
/// columns), falls back to hashing the per-row code tuple. Rows are then
/// grouped with a counting sort.
pub(crate) fn build_strata_partition(z_columns: &[(&[usize], usize)], n: usize) -> StrataPartition {
    let mut dense_ids: Vec<u32> = Vec::with_capacity(n);
    let n_strata: usize;

    if let Some((strides, total)) = fold_strides(z_columns) {
        // Pass 1: folded stratum id per row.
        let folded: Vec<usize> = (0..n)
            .map(|row| {
                z_columns
                    .iter()
                    .zip(&strides)
                    .map(|((codes, _), stride)| codes[row] * stride)
                    .sum()
            })
            .collect();
        // Pass 2: densify (flat remap when the folded space is small).
        if total <= 4 * n + DENSE_REMAP_SLACK {
            let mut remap = vec![u32::MAX; total];
            let mut next = 0u32;
            for &f in &folded {
                if remap[f] == u32::MAX {
                    remap[f] = next;
                    next += 1;
                }
                dense_ids.push(remap[f]);
            }
            n_strata = next as usize;
        } else {
            n_strata = densify_hashed(folded.iter().copied(), &mut dense_ids);
        }
    } else {
        // Radix overflow: hash the per-row code tuple to first-seen dense ids.
        let keys = (0..n).map(|row| {
            z_columns
                .iter()
                .map(|(codes, _)| codes[row])
                .collect::<Vec<usize>>()
        });
        n_strata = densify_hashed(keys, &mut dense_ids);
    }

    // Pass 3: counting-sort row indices by dense stratum id.
    let mut starts = vec![0u32; n_strata + 1];
    for &d in &dense_ids {
        starts[d as usize + 1] += 1;
    }
    for s in 0..n_strata {
        starts[s + 1] += starts[s];
    }
    let mut cursor = starts.clone();
    let mut rows = vec![0u32; n];
    for (row, &d) in dense_ids.iter().enumerate() {
        let slot = cursor[d as usize] as usize;
        rows[slot] = u32::try_from(row).expect("row count fits in u32");
        cursor[d as usize] += 1;
    }

    StrataPartition { starts, rows }
}

/// Conditional power-divergence statistic for parameter `lambda`: tabulate
/// each stratum of `partition` into one reused table over the *global* X/Y
/// cardinalities, skip degenerate strata, and sum the statistic and dof.
/// Yates' correction is applied per active 2×2 stratum when `yates` is set.
pub(crate) fn power_divergence_conditional(
    x_codes: &[usize],
    y_codes: &[usize],
    kx: usize,
    ky: usize,
    partition: &StrataPartition,
    lambda: f64,
    yates: bool,
) -> DiscreteOutcome {
    let mut table = ContingencyTable::zeros(kx, ky);
    let mut statistic = 0.0;
    let mut dof = 0;
    for s in 0..partition.n_strata() {
        table.reset();
        let lo = partition.starts[s] as usize;
        let hi = partition.starts[s + 1] as usize;
        for &row in &partition.rows[lo..hi] {
            table.add(x_codes[row as usize], y_codes[row as usize]);
        }
        if let Some((stat, table_dof)) = power_divergence_table(&table, lambda, yates) {
            statistic += stat;
            dof += table_dof;
        }
    }
    DiscreteOutcome { statistic, dof }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAMBDA_PEARSON: f64 = 1.0;
    const LAMBDA_CRESSIE_READ: f64 = 2.0 / 3.0;
    const LAMBDA_FREEMAN_TUKEY: f64 = -0.5;

    fn table(rows: usize, cols: usize, counts: &[f64]) -> ContingencyTable {
        ContingencyTable {
            rows,
            cols,
            counts: counts.to_vec(),
        }
    }

    #[test]
    fn yates_2x2_matches_scipy_pearson() {
        // Table [[10, 0], [0, 10]] with Yates: each |O-E|=5 shrinks by 0.5.
        // E = 5 everywhere; corrected diff = 4.5; stat = 4*(4.5^2/5) = 16.2.
        let x = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let out = power_divergence_unconditional(&x, &y, 2, 2, LAMBDA_PEARSON, true);
        assert!((out.statistic - 16.2).abs() < 1e-9, "got {}", out.statistic);
        assert_eq!(out.dof, 1);
    }

    #[test]
    fn yates_can_be_disabled() {
        // Same table without Yates: stat = 4*(5^2/5) = 20.
        let x = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let out = power_divergence_unconditional(&x, &y, 2, 2, LAMBDA_PEARSON, false);
        assert!((out.statistic - 20.0).abs() < 1e-9, "got {}", out.statistic);
        assert_eq!(out.dof, 1);
    }

    #[test]
    fn independent_table_zero_statistic() {
        // Perfectly balanced 2x2 -> chi-square 0 (after Yates, still ~0).
        let x = vec![0, 0, 1, 1];
        let y = vec![0, 1, 0, 1];
        let out = power_divergence_unconditional(&x, &y, 2, 2, LAMBDA_PEARSON, true);
        assert!(out.statistic.abs() < 1e-12);
        assert_eq!(out.dof, 1);
    }

    #[test]
    fn single_category_degenerate() {
        let x = vec![0, 0, 0, 0];
        let y = vec![0, 1, 0, 1];
        let out = power_divergence_unconditional(&x, &y, 1, 2, LAMBDA_PEARSON, true);
        assert_eq!(out.dof, 0);
        assert!(out.statistic.abs() < 1e-12);
    }

    #[test]
    fn modified_likelihood_zero_cell_is_infinite() {
        // λ = −1 with a structural zero where E > 0 -> +∞ statistic.
        // 3×3 (dof != 1 so no Yates) with a zero in an active row/col.
        let t = table(3, 3, &[5.0, 5.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
        let (stat, dof) = power_divergence_table(&t, LAMBDA_MODIFIED, true).unwrap();
        assert!(stat.is_infinite() && stat > 0.0, "got {stat}");
        assert_eq!(dof, 4);
    }

    #[test]
    fn zero_cell_finite_for_lambda_gt_minus_one() {
        // For λ > −1 a zero observed cell contributes a finite (0) term, never NaN.
        let t = table(3, 3, &[5.0, 5.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
        for lambda in [
            LAMBDA_PEARSON,
            LAMBDA_LOG_LIKELIHOOD,
            LAMBDA_CRESSIE_READ,
            LAMBDA_FREEMAN_TUKEY,
        ] {
            let (stat, _) = power_divergence_table(&t, lambda, true).unwrap();
            assert!(stat.is_finite(), "lambda {lambda} gave {stat}");
        }
    }

    /// Deterministic LCG so the parity tests need no rand dependency.
    fn lcg_codes(seed: &mut u64, n: usize, card: usize) -> Vec<usize> {
        (0..n)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((*seed >> 33) as usize) % card
            })
            .collect()
    }

    /// Independent naive reference: group rows by the Z-code tuple with a
    /// `HashMap`, tabulate each group, and sum statistic/dof.
    fn naive_conditional(
        x: &[usize],
        y: &[usize],
        kx: usize,
        ky: usize,
        z: &[(&[usize], usize)],
        lambda: f64,
        yates: bool,
    ) -> DiscreteOutcome {
        let n = x.len();
        let mut strata: std::collections::HashMap<Vec<usize>, Vec<usize>> =
            std::collections::HashMap::new();
        for i in 0..n {
            let key: Vec<usize> = z.iter().map(|(codes, _)| codes[i]).collect();
            strata.entry(key).or_default().push(i);
        }
        let mut statistic = 0.0;
        let mut dof = 0;
        for idx in strata.values() {
            let mut table = ContingencyTable::zeros(kx, ky);
            for &i in idx {
                table.add(x[i], y[i]);
            }
            if let Some((stat, table_dof)) = power_divergence_table(&table, lambda, yates) {
                statistic += stat;
                dof += table_dof;
            }
        }
        DiscreteOutcome { statistic, dof }
    }

    fn assert_outcomes_match(fast: &DiscreteOutcome, slow: &DiscreteOutcome, label: &str) {
        assert_eq!(fast.dof, slow.dof, "dof {label}");
        let scale = fast.statistic.abs().max(slow.statistic.abs()).max(1.0);
        assert!(
            (fast.statistic - slow.statistic).abs() <= 1e-9 * scale
                || (fast.statistic.is_infinite() && slow.statistic.is_infinite()),
            "stat {label}: {} vs {}",
            fast.statistic,
            slow.statistic
        );
    }

    #[test]
    fn partition_matches_naive_grouping() {
        let mut seed = 0x00C0_FFEE_u64;
        for (n, kx, ky, z_cards) in [
            (200, 2, 2, vec![2]),
            (500, 3, 4, vec![2, 3]),
            (350, 2, 3, vec![3, 2, 4]),
            (64, 4, 4, vec![5]),
            (30, 2, 2, vec![100, 100]), // exercises the HashMap densify arm
        ] {
            let x = lcg_codes(&mut seed, n, kx);
            let y = lcg_codes(&mut seed, n, ky);
            let z: Vec<(Vec<usize>, usize)> = z_cards
                .iter()
                .map(|&c| (lcg_codes(&mut seed, n, c), c))
                .collect();
            let z_ref: Vec<(&[usize], usize)> = z.iter().map(|(v, c)| (v.as_slice(), *c)).collect();
            let partition = build_strata_partition(&z_ref, n);
            for lambda in [1.0, 0.0, -1.0, 2.0 / 3.0, -0.5] {
                let fast = power_divergence_conditional(&x, &y, kx, ky, &partition, lambda, true);
                let slow = naive_conditional(&x, &y, kx, ky, &z_ref, lambda, true);
                assert_outcomes_match(&fast, &slow, &format!("λ={lambda} n={n}"));
            }
        }
    }

    #[test]
    fn overflow_path_matches_naive() {
        // 64 conditioning columns of cardinality 2 -> 2^64 overflows usize,
        // forcing the in-builder hashed grouping. 8 rows.
        let n = 8;
        let x: Vec<usize> = (0..n).map(|i| i % 2).collect();
        let y: Vec<usize> = (0..n).map(|i| (i / 2) % 2).collect();
        let z_cols: Vec<Vec<usize>> = (0..64)
            .map(|c| (0..n).map(|i| (i >> (c % 3)) % 2).collect())
            .collect();
        let z_ref: Vec<(&[usize], usize)> = z_cols.iter().map(|v| (v.as_slice(), 2)).collect();
        let partition = build_strata_partition(&z_ref, n);
        let fast = power_divergence_conditional(&x, &y, 2, 2, &partition, 1.0, true);
        let slow = naive_conditional(&x, &y, 2, 2, &z_ref, 1.0, true);
        assert_outcomes_match(&fast, &slow, "overflow");
    }
}
