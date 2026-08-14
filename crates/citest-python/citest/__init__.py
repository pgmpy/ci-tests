"""Data-bound conditional-independence testing.

Build a :class:`Dataset` once from named, typed columns, then construct any of
the eight tests bound to that data and query ``run_test`` / ``is_independent``::

    import numpy as np
    from citest import Dataset, ChiSquared, PearsonEquivalence

    data = Dataset({
        "A": ("discrete", np.array([0, 1, 0, 1], dtype=float)),
        "B": ("discrete", np.array([1, 1, 0, 0], dtype=float)),
    })
    res = ChiSquared(data).run_test("A", "B")
    res.statistic, res.p_value, res.dof, res.effect_size

The per-test classes are data-bound: they take the :class:`Dataset` (or a raw
``{name: (kind, values)}`` mapping or a :class:`pandas.DataFrame`) plus their
own configuration in the constructor. ``x`` / ``y`` accept a column name or
integer index; ``z`` is a sequence of names/indices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from citest._citest import (
    ChiSquared,
    CiError,
    CiResult,
    CressieRead,
    FisherZ,
    FreemanTukey,
    LogLikelihood,
    ModifiedLikelihood,
    PearsonCorrelation,
    PearsonEquivalence,
)
from citest._citest import (
    Dataset as _Dataset,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import pandas as pd

__all__ = [
    "ChiSquared",
    "CiError",
    "CiResult",
    "CressieRead",
    "Dataset",
    "FisherZ",
    "FreemanTukey",
    "LogLikelihood",
    "ModifiedLikelihood",
    "PearsonCorrelation",
    "PearsonEquivalence",
]


def _is_numpy_numeric(dtype: Any) -> bool:  # noqa: ANN401
    """Return True iff *dtype* is a numpy numeric (number or bool) dtype.

    Delegates to :func:`_safe_issubdtype`, which returns ``False`` (rather than
    raising ``TypeError``) for pandas extension dtypes such as ``StringDtype``
    or ``ArrowDtype`` that numpy cannot interpret.
    """
    return _safe_issubdtype(dtype, np.number) or _safe_issubdtype(dtype, np.bool_)


def _safe_issubdtype(dtype: Any, base: Any) -> bool:  # noqa: ANN401
    """Return whether ``dtype`` is a subtype of ``base`` without raising.

    Pandas extension dtypes that NumPy cannot interpret return ``False``.
    """
    try:
        return bool(np.issubdtype(dtype, base))
    except TypeError:
        return False


def _infer_kind(dtype: Any) -> str:  # noqa: ANN401 - numpy/pandas dtype is opaque here
    """Infer a column kind from a numpy/pandas dtype.

    Integer, boolean, categorical, object and string dtypes map to
    ``"discrete"``; floating dtypes map to ``"continuous"``. Anything else
    raises ``TypeError``.
    """
    name = str(getattr(dtype, "name", dtype))
    if name in ("category", "string", "str"):
        return "discrete"
    if dtype == np.object_ or _safe_issubdtype(dtype, np.str_):
        return "discrete"
    # Temporal dtypes are neither categorical nor plain-numeric; reject them
    # explicitly (timedelta64 would otherwise pass the integer check below).
    if _safe_issubdtype(dtype, np.datetime64) or _safe_issubdtype(dtype, np.timedelta64):
        msg = f"cannot infer column kind for temporal dtype {dtype!r}; convert it explicitly first"
        raise TypeError(msg)
    if _safe_issubdtype(dtype, np.floating):
        return "continuous"
    if _safe_issubdtype(dtype, np.integer) or _safe_issubdtype(dtype, np.bool_):
        return "discrete"
    msg = f"cannot infer column kind for dtype {dtype!r}; pass an explicit mapping"
    raise TypeError(msg)


class Dataset(_Dataset):
    """A named, typed, immutable table shared by the test classes.

    See the native :class:`citest._citest.Dataset` for the primary
    constructor (a ``{name: (kind, values)}`` mapping). This subclass adds the
    pure-Python :meth:`from_pandas` convenience.
    """

    __slots__ = ()

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> Dataset:
        """Build a :class:`Dataset` from a :class:`pandas.DataFrame`.

        Column kinds are inferred from dtypes: integer / boolean / categorical
        / object / string columns become ``"discrete"`` and floating columns
        become ``"continuous"``. Categorical and object/string columns are
        factorized to integer codes; **missing values are mapped to NaN and
        rejected by the core** (strict missing-data policy — drop or impute
        first). ``pandas`` is imported lazily, so it is not a hard dependency
        of this package.

        .. note::
            Passing a :class:`pandas.DataFrame` directly to any test
            constructor (e.g. ``ChiSquared(df)``) is equivalent to
            ``ChiSquared(Dataset.from_pandas(df))`` — a fresh :class:`Dataset`
            is built per constructor call; to share data across tests, build
            a :class:`Dataset` explicitly.
        """
        import pandas as pd  # noqa: PLC0415 - optional dependency imported lazily

        columns: dict[str, tuple[str, Any]] = {}
        for name in df.columns:
            series = df[name]
            kind = _infer_kind(series.dtype)
            dtype_name = str(getattr(series.dtype, "name", ""))
            if dtype_name == "category":
                codes = np.asarray(series.cat.codes, dtype=np.float64)
                codes[codes == -1.0] = np.nan  # missing category -> NaN -> core error
                values = codes
            elif kind == "discrete" and not _is_numpy_numeric(series.dtype):
                raw, _ = pd.factorize(series)  # -1 marks missing
                codes = raw.astype(np.float64)
                codes[codes == -1.0] = np.nan
                values = codes
            else:
                values = np.asarray(series, dtype=np.float64)
            columns[str(name)] = (kind, values)
        return cls(columns)
