"""Data-bound conditional-independence testing.

Build a :class:`Dataset` once from named, typed columns, then construct any of
the seven tests bound to that data and query ``run_test`` / ``is_independent``::

    import numpy as np
    from ci_python import Dataset, ChiSquared, PearsonEquivalence

    data = Dataset({
        "A": ("discrete", np.array([0, 1, 0, 1], dtype=float)),
        "B": ("discrete", np.array([1, 1, 0, 0], dtype=float)),
    })
    res = ChiSquared(data).run_test("A", "B")
    res.statistic, res.p_value, res.dof, res.effect_size

The per-test classes are data-bound: they take the :class:`Dataset` (or a raw
``{name: (kind, values)}`` mapping) plus their own configuration in the
constructor. ``x`` / ``y`` accept a column name or integer index; ``z`` is a
sequence of names/indices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ci_python._ci_python import (
    ChiSquared,
    CiError,
    CiResult,
    CressieRead,
    Dataset as _Dataset,
    FreemanTukey,
    LogLikelihood,
    ModifiedLikelihood,
    PearsonCorrelation,
    PearsonEquivalence,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import pandas as pd

__all__ = [
    "ChiSquared",
    "CiError",
    "CiResult",
    "CressieRead",
    "Dataset",
    "FreemanTukey",
    "LogLikelihood",
    "ModifiedLikelihood",
    "PearsonCorrelation",
    "PearsonEquivalence",
]


def _infer_kind(dtype: Any) -> str:  # noqa: ANN401 - numpy dtype is opaque here
    """Infer a column kind from a numpy/pandas dtype.

    Integer, boolean and categorical dtypes map to ``"discrete"``; floating
    dtypes map to ``"continuous"``. Anything else raises ``TypeError``.
    """
    import numpy as np

    if str(getattr(dtype, "name", dtype)) == "category":
        return "discrete"
    if np.issubdtype(dtype, np.floating):
        return "continuous"
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        return "discrete"
    msg = f"cannot infer column kind for dtype {dtype!r}; pass an explicit mapping"
    raise TypeError(msg)


class Dataset(_Dataset):
    """A named, typed, immutable table shared by the test classes.

    See the native :class:`ci_python._ci_python.Dataset` for the primary
    constructor (a ``{name: (kind, values)}`` mapping). This subclass adds the
    pure-Python :meth:`from_pandas` convenience.
    """

    __slots__ = ()

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> Dataset:
        """Build a :class:`Dataset` from a :class:`pandas.DataFrame`.

        Column kinds are inferred from dtypes: integer / boolean / categorical
        columns become ``"discrete"`` and floating columns become
        ``"continuous"``. Categorical columns are encoded by their integer
        codes before factorization. ``pandas`` is imported lazily, so it is not
        a hard dependency of this package.
        """
        import numpy as np

        columns: dict[str, tuple[str, Any]] = {}
        for name in df.columns:
            series = df[name]
            kind = _infer_kind(series.dtype)
            if str(getattr(series.dtype, "name", "")) == "category":
                values = np.asarray(series.cat.codes, dtype=np.float64)
            else:
                values = np.asarray(series, dtype=np.float64)
            columns[str(name)] = (kind, values)
        return cls(columns)
