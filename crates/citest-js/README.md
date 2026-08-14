# citest

WebAssembly bindings for the data-bound conditional-independence API.

## Usage

Build the Node.js package from the repository root:

```sh
wasm-pack build crates/citest-js --target nodejs
```

```js
import { Dataset, PearsonCorrelation } from "./pkg/citest_js.js";

const data = new Dataset({
  X: { kind: "continuous", values: [0.2, 0.8, 1.4, 2.1, 2.7] },
  Y: { kind: "continuous", values: [1.0, 1.7, 2.5, 3.0, 3.8] },
});
const result = new PearsonCorrelation(data).runTest("X", "Y", []);
```

Run the JavaScript checks from the package directory:

```sh
npm ci
npm test
```

For browser builds, the complete API, and contribution guidance, see the root
[README](../../README.md) and [CONTRIBUTING.md](../../CONTRIBUTING.md).
