# Type stubs for the native `citest._citest` extension module.
# Hand-maintained to match `crates/citest-python/src/lib.rs`.
# ruff: noqa: D101, D102, D107, PYI021, ANN401

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

ColumnRef = str | int
ColumnSpec = tuple[str, Any]

class CiError(Exception): ...

class CiResult:
    @property
    def statistic(self) -> float | None: ...
    @property
    def p_value(self) -> float: ...
    @property
    def dof(self) -> int | None: ...
    @property
    def effect_size(self) -> float | None: ...

class Dataset:
    def __init__(self, columns: dict[str, ColumnSpec]) -> None: ...
    @property
    def n_rows(self) -> int: ...
    @property
    def n_cols(self) -> int: ...
    def index_of(self, name: str) -> int: ...

class _BaseTest:
    def run_test(
        self,
        x: ColumnRef,
        y: ColumnRef,
        z: Sequence[ColumnRef] | None = ...,
    ) -> CiResult: ...
    def is_independent(
        self,
        x: ColumnRef,
        y: ColumnRef,
        z: Sequence[ColumnRef] | None = ...,
        significance_level: float = ...,
    ) -> bool: ...
    def meta(self) -> dict[str, Any]: ...

class ChiSquared(_BaseTest):
    def __init__(self, data: Dataset | dict[str, ColumnSpec] | pd.DataFrame, yates: bool = ...) -> None: ...

class LogLikelihood(_BaseTest):
    def __init__(self, data: Dataset | dict[str, ColumnSpec] | pd.DataFrame, yates: bool = ...) -> None: ...

class CressieRead(_BaseTest):
    def __init__(self, data: Dataset | dict[str, ColumnSpec] | pd.DataFrame, yates: bool = ...) -> None: ...

class FisherZ(_BaseTest):
    def __init__(self, data: Dataset | dict[str, ColumnSpec] | pd.DataFrame) -> None: ...

class FreemanTukey(_BaseTest):
    def __init__(self, data: Dataset | dict[str, ColumnSpec] | pd.DataFrame, yates: bool = ...) -> None: ...

class ModifiedLikelihood(_BaseTest):
    def __init__(self, data: Dataset | dict[str, ColumnSpec] | pd.DataFrame, yates: bool = ...) -> None: ...

class PearsonCorrelation(_BaseTest):
    def __init__(self, data: Dataset | dict[str, ColumnSpec] | pd.DataFrame) -> None: ...

class PearsonEquivalence(_BaseTest):
    def __init__(
        self,
        data: Dataset | dict[str, ColumnSpec] | pd.DataFrame,
        delta_threshold: float = ...,
    ) -> None: ...
