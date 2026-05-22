extendr bindings

## Usage

Install R, then open an R session:

```sh
R
```

Install `pak` (the modern R package manager that handles system dependencies automatically):

```r
install.packages("pak", repos = "https://cloud.r-project.org")
```

Install dev dependencies:

```r
pak::pak(c("devtools", "rextendr"))
```

### Regenerate bindings

```r
setwd("crates/cir")
rextendr::document()  # regenerates R wrappers and compiles Rust
```

### Load and test

```r
devtools::load_all()  # compiles Rust + loads the package
devtools::test()      # run all tests
```
