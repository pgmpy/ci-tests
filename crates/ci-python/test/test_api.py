"""Surface tests for the data-bound Python API (name/index handling, errors)."""

from __future__ import annotations

import numpy as np
import pytest

from ci_python import ChiSquared, CiError, Dataset, PearsonCorrelation, PearsonEquivalence


def _discrete_data() -> Dataset:
    rng = np.random.default_rng(0)
    return Dataset(
        {
            "A": ("discrete", rng.integers(0, 2, 200).astype(np.float64)),
            "B": ("discrete", rng.integers(0, 2, 200).astype(np.float64)),
            "C": ("discrete", rng.integers(0, 3, 200).astype(np.float64)),
        }
    )


def _continuous_data() -> Dataset:
    rng = np.random.default_rng(1)
    return Dataset(
        {
            "X": ("continuous", rng.standard_normal(200)),
            "Y": ("continuous", rng.standard_normal(200)),
            "Z": ("continuous", rng.standard_normal(200)),
        }
    )


def test_dataset_shape_and_index() -> None:
    data = _discrete_data()
    assert data.n_rows == 200
    assert data.n_cols == 3
    assert data.index_of("A") == 0
    assert data.index_of("C") == 2


def test_run_test_accepts_names_and_indices() -> None:
    data = _discrete_data()
    chi = ChiSquared(data)
    by_name = chi.run_test("A", "B", ["C"])
    by_index = chi.run_test(0, 1, [2])
    assert by_name.statistic == pytest.approx(by_index.statistic)
    assert by_name.p_value == pytest.approx(by_index.p_value)
    assert by_index.dof == by_name.dof


def test_result_fields_present_for_discrete() -> None:
    chi = ChiSquared(_discrete_data())
    res = chi.run_test("A", "B")
    assert isinstance(res.statistic, float)
    assert isinstance(res.p_value, float)
    assert isinstance(res.dof, int)


def test_z_defaults_to_empty() -> None:
    chi = ChiSquared(_discrete_data())
    assert chi.run_test("A", "B").p_value == pytest.approx(chi.run_test("A", "B", []).p_value)


def test_is_independent_uses_rule() -> None:
    chi = ChiSquared(_discrete_data())
    assert isinstance(chi.is_independent("A", "C", ["B"], significance_level=0.05), bool)


def test_pearson_equivalence_dof_is_none() -> None:
    eqv = PearsonEquivalence(_continuous_data(), delta_threshold=0.1)
    res = eqv.run_test("X", "Y", ["Z"])
    assert res.dof is None
    assert isinstance(res.p_value, float)


def test_pearson_correlation_takes_no_config() -> None:
    res = PearsonCorrelation(_continuous_data()).run_test("X", "Y")
    assert res.dof is None or isinstance(res.dof, int)


def test_meta_reports_rule_and_types() -> None:
    chi_meta = ChiSquared(_discrete_data()).meta()
    assert chi_meta["name"] == "chi_squared"
    assert chi_meta["data_types"] == ["discrete"]
    assert chi_meta["symmetric"] is True
    assert chi_meta["rule"] == "p_value_ge"

    eqv_meta = PearsonEquivalence(_continuous_data(), delta_threshold=0.1).meta()
    assert eqv_meta["rule"] == "p_value_lt"
    assert eqv_meta["data_types"] == ["continuous"]


def test_unknown_column_name_raises() -> None:
    chi = ChiSquared(_discrete_data())
    with pytest.raises(ValueError, match="unknown column"):
        chi.run_test("A", "nope")


def test_out_of_range_index_raises() -> None:
    chi = ChiSquared(_discrete_data())
    with pytest.raises(ValueError, match="out of range"):
        chi.run_test(0, 99)


def test_wrong_column_kind_raises_cierror() -> None:
    # A continuous column handed to the (discrete) chi-squared test.
    chi = ChiSquared(_continuous_data())
    with pytest.raises(CiError):
        chi.run_test("X", "Y")


def test_dataset_can_be_shared_across_tests() -> None:
    data = _continuous_data()
    a = PearsonCorrelation(data).run_test("X", "Y")
    b = PearsonEquivalence(data, delta_threshold=0.2).run_test("X", "Y")
    assert isinstance(a.p_value, float)
    assert isinstance(b.p_value, float)


def test_test_accepts_raw_mapping() -> None:
    # A test constructor may take a raw {name: (kind, values)} mapping directly.
    chi = ChiSquared(
        {
            "A": ("discrete", np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])),
            "B": ("discrete", np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])),
        }
    )
    assert isinstance(chi.run_test("A", "B").p_value, float)
