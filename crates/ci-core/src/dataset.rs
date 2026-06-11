//! Data-bound container shared by every conditional-independence test.
//!
//! A [`Dataset`] is built once from named columns. Discrete columns are
//! *factorized* up front into contiguous integer codes (`0..cardinality`) so
//! the discrete tests can operate on cheap `usize` codes instead of repeatedly
//! hashing floats. Continuous columns keep their raw `f64` values.

use std::collections::HashMap;

use crate::error::CiError;

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
    Discrete { codes: Vec<usize>, cardinality: usize },
    /// Raw continuous column.
    Continuous { values: Vec<f64> },
}

/// A named, typed, immutable table of columns.
///
/// Build one with [`Dataset::from_columns`]; tests then refer to columns by
/// their integer index (see [`Dataset::index_of`]).
#[derive(Debug, Clone)]
pub struct Dataset {
    names: Vec<String>,
    name_to_index: HashMap<String, usize>,
    columns: Vec<Column>,
    n_rows: usize,
}

/// Canonicalize a float so that values which should be considered equal hash
/// and compare equal: every `NaN` maps to one canonical `NaN`, and `-0.0` maps
/// to `0.0`.
fn canonical_bits(v: f64) -> u64 {
    if v.is_nan() {
        // One canonical NaN bit pattern for all NaNs.
        f64::NAN.to_bits()
    } else if v == 0.0 {
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
    /// in first-seen order; `-0.0`/`0.0` and all `NaN`s are grouped together.
    /// Continuous columns store their raw values.
    ///
    /// # Errors
    ///
    /// Returns [`CiError::DimensionMismatch`] if the columns do not all share a
    /// common length.
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

        let mut names = Vec::with_capacity(cols.len());
        let mut name_to_index = HashMap::with_capacity(cols.len());
        let mut columns = Vec::with_capacity(cols.len());

        for (idx, (name, kind, values)) in cols.into_iter().enumerate() {
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
            name_to_index.insert(name.clone(), idx);
            names.push(name);
            columns.push(column);
        }

        Ok(Self {
            names,
            name_to_index,
            columns,
            n_rows,
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

    /// Name of the column at `index`, if any.
    #[must_use]
    pub fn name_of(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(String::as_str)
    }

    /// Index of the column named `name`, if present.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    /// Read a discrete column as `(codes, cardinality)`.
    ///
    /// # Errors
    ///
    /// Returns [`CiError::UnknownColumn`] if `index` is out of range, or
    /// [`CiError::WrongColumnKind`] if the column is continuous.
    pub(crate) fn discrete(&self, index: usize) -> Result<(&[usize], usize), CiError> {
        match self.columns.get(index) {
            Some(Column::Discrete { codes, cardinality }) => Ok((codes, *cardinality)),
            Some(Column::Continuous { .. }) => Err(CiError::WrongColumnKind(format!(
                "column {index} is continuous but a discrete column was required"
            ))),
            None => Err(CiError::UnknownColumn(format!("column index {index}"))),
        }
    }

    /// Read a continuous column's values.
    ///
    /// # Errors
    ///
    /// Returns [`CiError::UnknownColumn`] if `index` is out of range, or
    /// [`CiError::WrongColumnKind`] if the column is discrete.
    pub(crate) fn continuous(&self, index: usize) -> Result<&[f64], CiError> {
        match self.columns.get(index) {
            Some(Column::Continuous { values }) => Ok(values),
            Some(Column::Discrete { .. }) => Err(CiError::WrongColumnKind(format!(
                "column {index} is discrete but a continuous column was required"
            ))),
            None => Err(CiError::UnknownColumn(format!("column index {index}"))),
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
    fn groups_neg_zero_and_nan() {
        let ds = Dataset::from_columns(vec![(
            "a".to_string(),
            ColumnKind::Discrete,
            vec![0.0, -0.0, f64::NAN, f64::NAN, 1.0],
        )])
        .unwrap();
        let (codes, card) = ds.discrete(0).unwrap();
        // 0.0 and -0.0 share a code; both NaNs share a code; 1.0 is its own.
        assert_eq!(card, 3);
        assert_eq!(codes, &[0, 0, 1, 1, 2]);
    }

    #[test]
    fn continuous_round_trips() {
        let ds = Dataset::from_columns(vec![(
            "x".to_string(),
            ColumnKind::Continuous,
            vec![1.5, 2.5, 3.5],
        )])
        .unwrap();
        assert_eq!(ds.continuous(0).unwrap(), &[1.5, 2.5, 3.5]);
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
        assert!(matches!(
            ds.continuous(0),
            Err(CiError::WrongColumnKind(_))
        ));
        assert!(matches!(ds.discrete(9), Err(CiError::UnknownColumn(_))));
    }
}
