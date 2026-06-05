wasm-pack bindings

## Usage

Install wasm-pack bindgen:

```sh
cargo install wasm-pack
cargo install wasm-bindgen-cli
```

After installing, move to ci-core/ci-js, and build the bindings:

```sh
# Build for Node.js
wasm-pack build --target nodejs

# Build for direct usage on the web
wasm-pack build --target web
```

### Run tests

Build the bindings for Node.js and run

```sh
npx vitest
```
