//! Data-bound container shared by every conditional-independence test.
//!
//! A [`Dataset`] is built once from named columns. Discrete columns are
//! *factorized* up front into contiguous integer codes (`0..cardinality`) so
//! the discrete tests can operate on cheap `usize` codes instead of repeatedly
//! hashing floats. Continuous columns keep their raw `f64` values.
//! NaN values are rejected at construction ([`CiError::MissingData`]); choose
//! and apply a missing-data convention (drop / impute) before binding.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::discrete::{build_strata_partition, StrataPartition};
use crate::error::CiError;
use crate::gram::GramCache;

/// Total byte budget for cached stratum partitions (see
/// [`Dataset::strata_partition`]): partitions are cached until the budget is
/// reached; further conditioning sets compute on the fly without caching
/// (monotone — no eviction).
pub(crate) const MAX_STRATA_CACHE_BYTES: usize = 256 << 20;

/// Keyed cache of stratum partitions plus its current byte footprint.
#[derive(Debug, Default)]
pub(crate) struct StrataCacheInner {
    pub(crate) map: HashMap<Vec<usize>, Arc<StrataPartition>>,
    bytes: usize,
}

impl StrataCacheInner {
    /// Insert `partition` under `key` iff it fits in `budget`; returns whether
    /// it was cached. Factored out so tests can drive a tiny budget.
    ///
    /// The caller must ensure `key` is absent (the cache's double-checked
    /// lookup guarantees this); inserting an existing key would double-count
    /// `bytes`.
    pub(crate) fn insert_within_budget(
        &mut self,
        key: Vec<usize>,
        partition: &Arc<StrataPartition>,
        budget: usize,
    ) -> bool {
        debug_assert!(
            !self.map.contains_key(&key),
            "insert_within_budget requires an absent key"
        );
        let size = partition.approx_bytes();
        if self.bytes + size > budget {
            return false;
        }
        self.bytes += size;
        self.map.insert(key, Arc::clone(partition));
        true
    }
}

/// Whether a column holds categorical (discrete) or numeric (continuous) data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnKind {
    /// Categorical data; values are factorized into integer codes.
    Discrete,
    /// Numeric data; values are kept as `f64`.
    Continuous,
}

/// Stored representation of a single column.
#[derive(Debug, Clone)]
pub(crate) enum Column {
    /// Factorized discrete column: per-row codes plus the number of distinct
    /// categories.
    Discrete {
        codes: Vec<usize>,
        cardinality: usize,
    },
    /// Raw continuous column.
    Continuous { values: Vec<f64> },
}

/// A named, typed, immutable table of columns.
///
/// Build one with [`Dataset::from_columns`]; tests then refer to columns by
/// their integer index (see [`Dataset::index_of`]).
#[derive(Debug)]
pub struct Dataset {
    name_to_index: HashMap<String, usize>,
    columns: Vec<Column>,
    n_rows: usize,
    /// Lazily built Gaussian sufficient-statistic cache (means + centered
    /// cross-product matrix). Populated on first access via [`Dataset::gram`].
    gram: OnceLock<Option<GramCache>>,
    /// Lazily built, budget-bounded cache of stratum partitions keyed by the
    /// sorted conditioning-set indices (see [`Dataset::strata_partition`]).
    strata_cache: RwLock<StrataCacheInner>,
}

/// Canonicalize a float for factorization so that `-0.0` and `0.0` share a
/// code. NaN never reaches this function: `from_columns` rejects NaN columns
/// up front (strict missing-data policy).
fn canonical_bits(v: f64) -> u64 {
    if v == 0.0 {
        // Collapse -0.0 and 0.0.
        0.0_f64.to_bits()
    } else {
        v.to_bits()
    }
}

impl Dataset {
    /// Build a dataset from `(name, kind, values)` triples.
    ///
    /// Discrete columns are factorized into contiguous codes (`0..cardinality`)
    /// in first-seen order; `-0.0`/`0.0` are grouped together.
    /// Continuous columns store their raw values.
    ///
    /// # Errors
    ///
    /// Returns [`CiError::DimensionMismatch`] if the columns do not all share a
    /// common length, or if two columns share a name (names must be unique).
    /// Returns [`CiError::MissingData`] if a discrete column contains NaN, or a
    /// continuous column contains any non-finite value (NaN or ±inf).
    pub fn from_columns(cols: Vec<(String, ColumnKind, Vec<f64>)>) -> Result<Self, CiError> {
        let n_rows = cols.first().map_or(0, |(_, _, v)| v.len());
        for (name, _, values) in &cols {
            if values.len() != n_rows {
                return Err(CiError::DimensionMismatch(format!(
                    "column `{name}` has length {} but expected {n_rows}",
                    values.len(),
                )));
            }
        }

        let mut name_to_index = HashMap::with_capacity(cols.len());
        let mut columns = Vec::with_capacity(cols.len());

        for (idx, (name, kind, values)) in cols.into_iter().enumerate() {
            // Discrete columns reject NaN (missing); continuous columns reject any
            // non-finite value (NaN or ±inf), which would corrupt the Gram path.
            let bad = match kind {
                ColumnKind::Continuous => values.iter().position(|v| !v.is_finite()),
                ColumnKind::Discrete => values.iter().position(|v| v.is_nan()),
            };
            if let Some(row) = bad {
                return Err(CiError::MissingData(format!(
                    "column `{name}` contains NaN/missing values (first at row {row}); \
                     remove or impute before building the Dataset"
                )));
            }
            let column = match kind {
                ColumnKind::Continuous => Column::Continuous { values },
                ColumnKind::Discrete => {
                    let mut lookup: HashMap<u64, usize> = HashMap::new();
                    let mut codes = Vec::with_capacity(values.len());
                    for v in values {
                        let key = canonical_bits(v);
                        let next = lookup.len();
                        let code = *lookup.entry(key).or_insert(next);
                        codes.push(code);
                    }
                    Column::Discrete {
                        cardinality: lookup.len(),
                        codes,
                    }
                }
            };
            if name_to_index.contains_key(&name) {
                return Err(CiError::DimensionMismatch(format!(
                    "duplicate column name `{name}`; column names must be unique"
                )));
            }
            name_to_index.insert(name, idx);
            columns.push(column);
        }

        Ok(Self {
            name_to_index,
            columns,
            n_rows,
            gram: OnceLock::new(),
            strata_cache: RwLock::new(StrataCacheInner::default()),
        })
    }

    /// Number of rows (observations).
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    /// Index of the column named `name`, if present.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    /// Borrow the column at `index`, mapping an out-of-range index to
    /// [`CiError::UnknownColumn`].
    fn column(&self, index: usize) -> Result<&Column, CiError> {
        self.columns
            .get(index)
            .ok_or_else(|| CiError::UnknownColumn(format!("column index {index}")))
    }

    /// Read a discrete column as `(codes, cardinality)`.
    ///
    /// # Errors
    ///
    /// Returns [`CiError::UnknownColumn`] if `index` is out of range, or
    /// [`CiError::WrongColumnKind`] if the column is continuous.
    pub(crate) fn discrete(&self, index: usize) -> Result<(&[usize], usize), CiError> {
        match self.column(index)? {
            Column::Discrete { codes, cardinality } => Ok((codes, *cardinality)),
            Column::Continuous { .. } => Err(CiError::WrongColumnKind(format!(
                "column {index} is continuous but a discrete column was required"
            ))),
        }
    }

    /// The values of the column at `index` when it is continuous, or `None`
    /// when the index is out of range or the column is discrete. Callers that
    /// must distinguish the two failure modes (the continuous tests) go
    /// through [`crate::gram::GramCache`], whose lookup reports them as
    /// [`CiError::WrongColumnKind`] / [`CiError::UnknownColumn`].
    pub(crate) fn continuous_values(&self, index: usize) -> Option<&[f64]> {
        match self.columns.get(index)? {
            Column::Continuous { values } => Some(values),
            Column::Discrete { .. } => None,
        }
    }

    /// The lazily built Gaussian sufficient-statistic cache, or `None` when
    /// the dataset has no continuous columns / too many of them (see
    /// [`crate::gram::MAX_GRAM_COLS`]).
    ///
    /// `OnceLock::get_or_init` runs the build exactly once, blocking any
    /// concurrent callers until it completes; the stored cache is therefore
    /// built a single time and is deterministic.
    pub(crate) fn gram(&self) -> Option<&GramCache> {
        self.gram.get_or_init(|| GramCache::build(self)).as_ref()
    }

    /// The stratum partition for conditioning set `z`, cached per distinct
    /// (order-insensitive) set of column indices. On a miss the partition is
    /// built outside the lock and inserted only while the total cache stays
    /// within [`MAX_STRATA_CACHE_BYTES`]; once the budget is reached, further
    /// sets are computed per call without caching.
    ///
    /// # Errors
    ///
    /// Returns the usual column errors ([`CiError::WrongColumnKind`] /
    /// [`CiError::UnknownColumn`]) if any `z` column is not discrete.
    pub(crate) fn strata_partition(&self, z: &[usize]) -> Result<Arc<StrataPartition>, CiError> {
        let mut key: Vec<usize> = z.to_vec();
        key.sort_unstable();

        if let Some(hit) = self
            .strata_cache
            .read()
            .expect("strata cache lock poisoned")
            .map
            .get(&key)
        {
            return Ok(Arc::clone(hit));
        }

        // Build outside the lock; gather columns in sorted order so equal sets
        // yield byte-identical partitions.
        let z_columns: Vec<(&[usize], usize)> = key
            .iter()
            .map(|&zi| self.discrete(zi))
            .collect::<Result<_, _>>()?;
        let partition = Arc::new(build_strata_partition(&z_columns, self.n_rows));

        let mut cache = self
            .strata_cache
            .write()
            .expect("strata cache lock poisoned");
        if let Some(hit) = cache.map.get(&key) {
            return Ok(Arc::clone(hit)); // another thread won the race
        }
        cache.insert_within_budget(key, &partition, MAX_STRATA_CACHE_BYTES);
        Ok(partition)
    }
}

impl Clone for Dataset {
    /// Clones the column data; the lazy caches (Gram matrix, stratum
    /// partitions) start fresh in the clone and are rebuilt on demand.
    fn clone(&self) -> Self {
        Self {
            name_to_index: self.name_to_index.clone(),
            columns: self.columns.clone(),
            n_rows: self.n_rows,
            gram: OnceLock::new(),
            strata_cache: RwLock::new(StrataCacheInner::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorizes_discrete_first_seen() {
        let ds = Dataset::from_columns(vec![(
            "a".to_string(),
            ColumnKind::Discrete,
            vec![5.0, 5.0, 2.0, 2.0, 5.0],
        )])
        .unwrap();
        let (codes, card) = ds.discrete(0).unwrap();
        assert_eq!(card, 2);
        assert_eq!(codes, &[0, 0, 1, 1, 0]);
        assert_eq!(ds.n_rows(), 5);
    }

    #[test]
    fn groups_neg_zero() {
        let ds = Dataset::from_columns(vec![(
            "a".to_string(),
            ColumnKind::Discrete,
            vec![0.0, -0.0, 1.0],
        )])
        .unwrap();
        let (codes, card) = ds.discrete(0).unwrap();
        // 0.0 and -0.0 share a code; 1.0 is its own.
        assert_eq!(card, 2);
        assert_eq!(codes, &[0, 0, 1]);
    }

    #[test]
    fn nan_errors_in_any_column_kind() {
        for kind in [ColumnKind::Discrete, ColumnKind::Continuous] {
            let err =
                Dataset::from_columns(vec![("a".to_string(), kind, vec![1.0, f64::NAN, 2.0])])
                    .unwrap_err();
            assert!(
                matches!(err, CiError::MissingData(_)),
                "kind {kind:?}: {err}"
            );
            let msg = err.to_string();
            assert!(msg.contains("`a`"), "message should name the column: {msg}");
            assert!(msg.contains("row 1"), "message should give the row: {msg}");
        }
    }

    #[test]
    fn duplicate_column_names_error() {
        let err = Dataset::from_columns(vec![
            ("a".to_string(), ColumnKind::Continuous, vec![1.0, 2.0]),
            ("a".to_string(), ColumnKind::Continuous, vec![3.0, 4.0]),
        ])
        .unwrap_err();
        assert!(matches!(err, CiError::DimensionMismatch(_)), "{err}");
        assert!(err.to_string().contains("`a`"));
    }

    #[test]
    fn infinite_continuous_value_errors() {
        let err = Dataset::from_columns(vec![(
            "a".to_string(),
            ColumnKind::Continuous,
            vec![1.0, f64::INFINITY, 2.0],
        )])
        .unwrap_err();
        assert!(matches!(err, CiError::MissingData(_)), "{err}");
        assert!(err.to_string().contains("row 1"));
    }

    #[test]
    fn continuous_round_trips() {
        let ds = Dataset::from_columns(vec![(
            "x".to_string(),
            ColumnKind::Continuous,
            vec![1.5, 2.5, 3.5],
        )])
        .unwrap();
        assert_eq!(ds.continuous_values(0).unwrap(), &[1.5, 2.5, 3.5]);
        assert_eq!(ds.index_of("x"), Some(0));
        assert_eq!(ds.index_of("nope"), None);
    }

    #[test]
    fn mismatched_lengths_error() {
        let err = Dataset::from_columns(vec![
            ("a".to_string(), ColumnKind::Continuous, vec![1.0, 2.0]),
            ("b".to_string(), ColumnKind::Continuous, vec![1.0]),
        ])
        .unwrap_err();
        assert!(matches!(err, CiError::DimensionMismatch(_)));
    }

    #[test]
    fn wrong_kind_errors() {
        let ds = Dataset::from_columns(vec![(
            "a".to_string(),
            ColumnKind::Discrete,
            vec![1.0, 2.0],
        )])
        .unwrap();
        assert!(ds.continuous_values(0).is_none(), "discrete column");
        assert!(ds.continuous_values(9).is_none(), "out of range");
        assert!(matches!(ds.discrete(9), Err(CiError::UnknownColumn(_))));
    }

    #[test]
    fn strata_partition_is_cached_and_order_insensitive() {
        let ds = Dataset::from_columns(vec![
            (
                "x".into(),
                ColumnKind::Discrete,
                vec![1., 2., 1., 2., 1., 2.],
            ),
            (
                "z1".into(),
                ColumnKind::Discrete,
                vec![1., 1., 2., 2., 1., 2.],
            ),
            (
                "z2".into(),
                ColumnKind::Discrete,
                vec![2., 1., 2., 1., 1., 2.],
            ),
        ])
        .unwrap();
        let a = ds.strata_partition(&[1, 2]).unwrap();
        let b = ds.strata_partition(&[1, 2]).unwrap();
        let c = ds.strata_partition(&[2, 1]).unwrap();
        assert!(
            std::sync::Arc::ptr_eq(&a, &b),
            "repeat query must hit the cache"
        );
        assert!(std::sync::Arc::ptr_eq(&a, &c), "z order must not matter");
        // A continuous column in z surfaces the usual WrongColumnKind.
        let ds2 = Dataset::from_columns(vec![
            ("x".into(), ColumnKind::Discrete, vec![1., 2.]),
            ("c".into(), ColumnKind::Continuous, vec![0.1, 0.2]),
        ])
        .unwrap();
        assert!(matches!(
            ds2.strata_partition(&[1]),
            Err(CiError::WrongColumnKind(_))
        ));
    }

    #[test]
    fn clone_starts_with_fresh_caches() {
        let ds = Dataset::from_columns(vec![
            ("x".into(), ColumnKind::Discrete, vec![1., 2., 1., 2.]),
            ("z".into(), ColumnKind::Discrete, vec![1., 1., 2., 2.]),
        ])
        .unwrap();
        let before = ds.strata_partition(&[1]).unwrap();
        let cloned = ds.clone();
        let after = cloned.strata_partition(&[1]).unwrap();
        // Same content, but the clone rebuilt its own partition.
        assert_eq!(before.starts, after.starts);
        assert_eq!(before.rows, after.rows);
        assert!(!std::sync::Arc::ptr_eq(&before, &after));
    }

    #[test]
    fn strata_cache_respects_budget() {
        let p = std::sync::Arc::new(crate::discrete::StrataPartition {
            starts: vec![0, 2, 4],
            rows: vec![0, 1, 2, 3],
        });
        let mut inner = StrataCacheInner::default();
        // A zero budget rejects even the first insert (strict `>` boundary).
        assert!(!inner.insert_within_budget(vec![0], &p, 0));
        assert_eq!(inner.bytes, 0);
        // Budget fits exactly one copy (7 u32s = 28 bytes).
        assert!(inner.insert_within_budget(vec![1], &p, 28));
        assert!(!inner.insert_within_budget(vec![2], &p, 28), "over budget");
        assert!(inner.map.contains_key(&vec![1]));
        assert!(!inner.map.contains_key(&vec![2]));
    }
}
