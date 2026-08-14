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
    list_tests,
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
    "list_tests",
]


def _infer_kind(dtype: Any) -> str:  # noqa: ANN401 - a numpy/pandas dtype is opaque here
    """Infer a column kind from a dtype, via pandas' own predicates.

    Boolean, integer, categorical, object and string dtypes map to
    ``"discrete"``, pandas nullable extension dtypes (``Int64``, ``boolean``)
    included; float dtypes (``Float64`` included) map to ``"continuous"``.
    Temporal dtypes are rejected explicitly; anything else (complex, interval,
    ...) raises ``TypeError``.
    """
    import pandas as pd  # noqa: PLC0415 - optional dependency imported lazily

    api = pd.api.types
    if api.is_datetime64_any_dtype(dtype) or api.is_timedelta64_dtype(dtype):
        msg = f"cannot infer column kind for temporal dtype {dtype!r}; convert it explicitly first"
        raise TypeError(msg)
    if isinstance(dtype, pd.CategoricalDtype) or api.is_bool_dtype(dtype) or api.is_integer_dtype(dtype):
        return "discrete"
    if api.is_float_dtype(dtype):
        return "continuous"
    if api.is_object_dtype(dtype) or api.is_string_dtype(dtype):
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
        / object / string columns become ``"discrete"`` and float columns
        become ``"continuous"``, pandas nullable extension dtypes (``Int64``,
        ``boolean``, ``Float64``) included. Categorical and object/string
        columns are factorized to integer codes; **missing values — NaN or
        ``pd.NA`` — are mapped to NaN and rejected by the core** (strict
        missing-data policy — drop or impute first). ``pandas`` is imported
        lazily, so it is not a hard dependency of this package.

        .. note::
            Passing a :class:`pandas.DataFrame` directly to any test
            constructor (e.g. ``ChiSquared(df)``) is equivalent to
            ``ChiSquared(Dataset.from_pandas(df))`` — a fresh :class:`Dataset`
            is built per constructor call; to share data across tests, build
            a :class:`Dataset` explicitly.
        """
        import pandas as pd  # noqa: PLC0415 - optional dependency imported lazily

        if df.columns.has_duplicates:
            duplicates = df.columns[df.columns.duplicated()].unique().tolist()
            raise CiError(f"duplicate column labels {duplicates!r}; column labels must be unique")

        columns: dict[str, tuple[str, Any]] = {}
        for name in df.columns:
            series = df[name]
            dtype = series.dtype
            kind = _infer_kind(dtype)
            if isinstance(dtype, pd.CategoricalDtype):
                codes = series.cat.codes.to_numpy(dtype=np.float64)
                codes[codes == -1.0] = np.nan  # missing category -> NaN -> core error
                values = codes
            elif kind == "discrete" and not (pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_integer_dtype(dtype)):
                raw, _ = pd.factorize(series)  # -1 marks missing
                codes = raw.astype(np.float64)
                codes[codes == -1.0] = np.nan
                values = codes
            else:
                # Plain and nullable numeric alike: NaN / pd.NA become NaN,
                # which the core rejects with its usual missing-data error.
                values = series.to_numpy(dtype=np.float64, na_value=np.nan)
            columns[str(name)] = (kind, values)
        return cls(columns)
