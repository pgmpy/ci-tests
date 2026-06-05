# ci-python

Python bindings for the [Conditional Independence Testing](../../README.md) library. Wraps the Rust core via [PyO3](https://pyo3.rs) and accepts NumPy arrays directly.

## Available tests

| Class | Data type | Numeric output |
|---|---|---|
| `ChiSquared` | Discrete | `(p_value, statistic, dof)` |
| `CressieRead` | Discrete | `(p_value, statistic, dof)` |
| `FreemanTukey` | Discrete | `(p_value, statistic, dof)` |
| `LogLikelihood` | Discrete | `(p_value, statistic, dof)` |
| `ModifiedLikelihood` | Discrete | `(p_value, statistic, dof)` |
| `PearsonCorrelation` | Continuous | `(p_value, coefficient)` |
| `PearsonEquivalence` | Continuous | `(p_value, coefficient)` |

All tests support an optional conditioning matrix Z. Pass an empty matrix for unconditional tests.

## Requirements

- Python 3.10 to 3.14
- NumPy
- Rust (stable), installed via [rustup](https://rustup.rs)
- [maturin](https://www.maturin.rs) 1.11 or later

## Installation

From the repository root:

```bash
pip install maturin
maturin develop -m crates/ci-python/Cargo.toml
```

## Usage

### Numeric mode

Numeric mode returns the raw test statistic alongside the p-value. Construct a test with `boolean=False`:

```python
import numpy as np
from ci_python import ChiSquared

test = ChiSquared(boolean=False, significance_level=0.05)

x = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
y = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
z = np.empty((len(x), 0), dtype=np.float64)  # unconditional

p_value, statistic, dof = test.run_test(x, y, z)
print(f"p={p_value:.4f}, chi2={statistic:.4f}, df={dof}")
```

For continuous data, the return type is `(p_value, coefficient)` rather than a triple:

```python
from ci_python import PearsonCorrelation

test = PearsonCorrelation(boolean=False, significance_level=0.05)
p_value, coefficient = test.run_test(x, y, z)
```

### Boolean mode

Boolean mode returns a single `bool`: `True` if the null hypothesis of independence is not rejected, `False` if it is rejected. Construct the test with `boolean=True`:

```python
from ci_python import CressieRead

test = CressieRead(boolean=True, significance_level=0.05)
independent: bool = test.run_test(x, y, z)
```

### Conditional tests

Pass a conditioning matrix Z where each column is one conditioning variable. The matrix must have the same number of rows as x and y:

```python
import numpy as np
from ci_python import ChiSquared

x = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float64)
y = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float64)
z = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]], dtype=np.float64)

test = ChiSquared(boolean=False, significance_level=0.05)
p_value, statistic, dof = test.run_test(x, y, z)
```

To condition on multiple variables, stack them as columns:

```python
z = np.column_stack([z1, z2])  # shape (n, 2)
test.run_test(x, y, z)
```

## Running tests

```bash
pip install -e "crates/ci-python[test]"
pytest crates/ci-python
```

## License

Licensed under the [MIT license](../../LICENSE).
