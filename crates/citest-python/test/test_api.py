"""Surface tests for the data-bound Python API (name/index handling, errors)."""

from __future__ import annotations

import pickle
import threading

import numpy as np
import pytest

import citest
from citest import (
    ChiSquared,
    CiError,
    Dataset,
    FisherZ,
    PearsonCorrelation,
    PearsonEquivalence,
    list_tests,
)


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


def test_effect_size_present_for_discrete_and_none_when_undefined() -> None:
    # Cramér's V is defined (a float) for a discrete test on 2+-level columns.
    res = ChiSquared(_discrete_data()).run_test("A", "B")
    assert isinstance(res.effect_size, float)
    # A single-level (constant) column has cardinality 1, so Cramér's V is
    # undefined and the core returns None for effect_size (statistic collapses
    # to 0, dof 0, p-value 1) rather than erroring.
    const_data = Dataset(
        {
            "A": ("discrete", np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])),
            "K": ("discrete", np.zeros(6)),
        }
    )
    res_none = ChiSquared(const_data).run_test("A", "K")
    assert res_none.effect_size is None


def test_negative_index_matches_name_resolution() -> None:
    # Python-style negative indices resolve identically to resolution by name:
    # -1 -> last column ("C"), -2 -> second-to-last ("B").
    chi = ChiSquared(_discrete_data())
    by_negative = chi.run_test(-1, -2)
    by_name = chi.run_test("C", "B")
    assert by_negative.statistic == pytest.approx(by_name.statistic)
    assert by_negative.p_value == pytest.approx(by_name.p_value)
    assert by_negative.dof == by_name.dof


def test_bare_string_z_raises_value_error() -> None:
    # A bare string as z must be rejected, not iterated character-by-character.
    chi = ChiSquared(_discrete_data())
    with pytest.raises(ValueError, match="not a single string"):
        chi.run_test("A", "B", "C")


def test_z_defaults_to_empty() -> None:
    chi = ChiSquared(_discrete_data())
    assert chi.run_test("A", "B").p_value == pytest.approx(chi.run_test("A", "B", []).p_value)


def test_is_independent_uses_rule() -> None:
    chi = ChiSquared(_discrete_data())
    assert isinstance(chi.is_independent("A", "C", ["B"], significance_level=0.05), bool)


@pytest.mark.parametrize("level", [float("nan"), float("inf"), float("-inf")])
def test_is_independent_rejects_non_finite_significance_level(level: float) -> None:
    chi = ChiSquared(_discrete_data())
    with pytest.raises(CiError, match="significance level must be finite"):
        chi.is_independent("A", "B", significance_level=level)


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


def test_fisher_z_concurrent_calls_are_safe() -> None:
    """fisher_z is exposed; concurrent run_test calls are consistent.

    Exercises the GIL-free path under thread contention and asserts results
    stay identical. (It cannot observe the GIL release itself; a wall-clock
    speedup assertion on a microsecond-scale call would be hopelessly flaky.)
    """
    data = _continuous_data()
    fz = FisherZ(data)
    res = fz.run_test("X", "Y", ["Z"])
    assert res.dof is None
    assert 0.0 <= res.p_value <= 1.0

    # Per-thread result lists: safe regardless of free-threaded CPython,
    # where concurrent list.append on a shared list is not guaranteed atomic.
    per_thread: list[list[float]] = [[] for _ in range(4)]

    def worker(bucket: list[float]) -> None:
        bucket.extend(fz.run_test("X", "Y", ["Z"]).p_value for _ in range(50))

    threads = [threading.Thread(target=worker, args=(b,)) for b in per_thread]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    results = [p for bucket in per_thread for p in bucket]
    assert len(results) == 200
    assert all(r == results[0] for r in results)


def test_invalid_queries_raise() -> None:
    chi = ChiSquared(_discrete_data())
    with pytest.raises(CiError, match="invalid query"):
        chi.run_test("A", "A")
    with pytest.raises(CiError, match="invalid query"):
        chi.run_test("A", "B", ["A"])
    with pytest.raises(CiError, match="invalid query"):
        chi.run_test("A", "B", ["C", "C"])


def test_nan_rejected_at_dataset_construction() -> None:
    with pytest.raises(CiError, match="missing data"):
        Dataset({"A": ("continuous", np.array([1.0, np.nan, 2.0]))})
    with pytest.raises(CiError, match="missing data"):
        Dataset({"A": ("discrete", np.array([1.0, np.nan, 2.0]))})


def test_constructor_accepts_dataframe_directly() -> None:
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "A": rng.integers(0, 2, 100),
            "B": rng.integers(0, 3, 100),
            "X": rng.standard_normal(100),
            "Y": rng.standard_normal(100),
        }
    )
    chi = ChiSquared(df)  # implicit: DataFrame -> Dataset inside the ctor
    via_dataset = ChiSquared(Dataset.from_pandas(df))
    a = chi.run_test("A", "B")
    b = via_dataset.run_test("A", "B")
    assert a.statistic == pytest.approx(b.statistic)
    assert a.p_value == pytest.approx(b.p_value)


def test_dataset_accepts_string_discrete_columns() -> None:
    data = Dataset(
        {
            "A": ("discrete", ["yes", "no", "yes", "no", "maybe", "yes"]),
            "B": ("discrete", np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])),
        }
    )
    res = ChiSquared(data).run_test("A", "B")
    assert 0.0 <= res.p_value <= 1.0


def test_continuous_string_values_rejected() -> None:
    with pytest.raises(ValueError, match="numeric"):
        Dataset({"X": ("continuous", ["a", "b", "c"])})


def test_from_pandas_object_and_category_columns() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "A": ["x", "y", "x", "y", "x", "y"],  # object -> discrete
            "B": pd.Categorical(["u", "v", "u", "v", "u", "v"]),  # category -> discrete
            "C": [0.1, 0.4, 0.2, 0.8, 0.5, 0.9],  # float -> continuous
        }
    )
    data = Dataset.from_pandas(df)
    assert data.n_cols == 3
    res = ChiSquared(data).run_test("A", "B")
    assert 0.0 <= res.p_value <= 1.0


def test_from_pandas_missing_categorical_errors() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"A": pd.Categorical(["u", None, "v"]), "B": [1.0, 2.0, 3.0]})
    with pytest.raises(CiError, match="missing data"):
        Dataset.from_pandas(df)
    df2 = pd.DataFrame({"A": ["u", None, "v"], "B": [1.0, 2.0, 3.0]})
    with pytest.raises(CiError, match="missing data"):
        Dataset.from_pandas(df2)


def test_from_pandas_extension_dtype_gives_friendly_error() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"A": pd.array([1, 2, 3], dtype="Int64"), "B": [0.1, 0.2, 0.3]})
    with pytest.raises(TypeError, match="cannot infer column kind"):
        Dataset.from_pandas(df)


def test_from_pandas_temporal_dtypes_rejected() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"A": pd.to_timedelta([1, 2, 3], unit="s"), "B": [0.1, 0.2, 0.3]})
    with pytest.raises(TypeError, match="cannot infer column kind"):
        Dataset.from_pandas(df)
    df2 = pd.DataFrame({"A": pd.to_datetime(["2024-01-01", "2024-01-02"]), "B": [0.1, 0.2]})
    with pytest.raises(TypeError, match="cannot infer column kind"):
        Dataset.from_pandas(df2)


def test_from_pandas_string_dtype_column() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "A": pd.array(["u", "v", "u", "w", "v", "u"], dtype="string"),
            "B": [0, 1, 0, 1, 0, 1],
        }
    )
    data = Dataset.from_pandas(df)
    res = ChiSquared(data).run_test("A", "B")
    assert 0.0 <= res.p_value <= 1.0


def test_every_failure_mode_is_a_cierror() -> None:
    """One exception type covers every way a query can fail.

    Column lookups used to raise a bare ValueError while engine faults raised
    CiError, so ``except CiError`` silently missed half the failure modes.
    """
    data = Dataset(
        {
            "A": ("discrete", np.array([0.0, 1.0, 0.0, 1.0])),
            "B": ("discrete", np.array([1.0, 1.0, 0.0, 0.0])),
            "X": ("continuous", np.array([0.1, 0.2, 0.3, 0.4])),
        }
    )
    test = ChiSquared(data)
    failures = [
        ("unknown column name", lambda: test.run_test("nope", "B")),
        ("out-of-range index", lambda: test.run_test(99, 1)),
        ("x equals y", lambda: test.run_test("A", "A")),
        ("wrong column kind", lambda: test.run_test("A", "X")),
        ("bad conditioning column", lambda: test.run_test("A", "B", ["X"])),
        ("bare string z", lambda: test.run_test("A", "B", "B")),
    ]
    assert issubclass(CiError, ValueError), "code written against the previous behaviour catches ValueError"
    for label, call in failures:
        try:
            call()
        except CiError:
            continue
        except Exception as exc:  # noqa: BLE001 - the point is that nothing else escapes
            pytest.fail(f"{label} raised {type(exc).__name__}, not CiError")
        pytest.fail(f"{label} did not raise at all")


def test_cierror_is_picklable() -> None:
    """A wrong __module__ made the exception unpicklable across processes."""
    assert CiError.__module__ == "citest._citest"
    payload = pickle.dumps(CiError("boom"))
    restored = pickle.loads(payload)  # noqa: S301 - round-tripping our own object
    assert isinstance(restored, CiError)


def test_registry_and_exported_classes_agree() -> None:
    """Every registered test is exported, and every export is registered.

    The core registry is the single source of truth. Without this check a ninth
    test could be added to the core and silently missing from this binding.
    """
    registered = {meta["name"] for meta in list_tests()}
    assert len(registered) == 8

    # "chi_squared" -> "ChiSquared"
    def to_class_name(stable: str) -> str:
        return "".join(part.capitalize() for part in stable.split("_"))

    for name in registered:
        cls = to_class_name(name)
        assert hasattr(citest, cls), f"{name} is registered but {cls} is not exported"
        assert cls in citest.__all__

    exported_tests = {name for name in citest.__all__ if name not in {"Dataset", "CiError", "CiResult", "list_tests"}}
    assert {to_class_name(n) for n in registered} == exported_tests


def test_list_tests_reports_each_decision_rule() -> None:
    by_name = {meta["name"]: meta for meta in list_tests()}
    assert by_name["chi_squared"]["rule"] == "p_value_ge"
    assert by_name["chi_squared"]["data_types"] == ["discrete"]
    # The equivalence test is the one with the inverted convention.
    assert by_name["pearson_equivalence"]["rule"] == "p_value_lt"
    assert by_name["fisher_z"]["data_types"] == ["continuous"]
