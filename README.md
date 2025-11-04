# GEDC Reasoning Co-Processor (RCP)

A certified theta-function verifier built from a Golden Exponential Dynamic Collapse (GEDC) core. This project provides a layer of interval arithmetic, JSON certificates and a simple task dispatcher around a user-supplied `gedc_core`. It is designed for rigorous numerical reasoning about theta-like series and simple analytic identities, with deterministic manifests for reproducibility.

This repository is a lightweight stand-alone implementation intended for research and experimentation. The accompanying `gedc_rcp.py` implements several primitives and a small dispatcher to drive them. A minimal `gedc_core.py` stub is provided so that the module can be imported out of the box – in practice you would replace it with your own core implementation.

## Features

### Interval-bound series
Compute partial sums of the series tr(x) = ∑_{n≥1} e^{-an²x} and its alternating version. The `bound` task returns an interval for the partial sum together with a rigorous tail bound. If the tail is below the requested precision `eps` the task is considered proved.

### Theta reciprocity scans
For a given exponent α and a list of positive grid points `t_grid`, the `scan` task certifies the inequality:

```
|t^α θ(t) - θ(1/t)| ≤ ε
```

where θ(t) = 1 + 2∑_{n≥1}(-1)^n e^{-an²t}. The output includes a certificate with the maximum and minimum residual intervals. If the maximal upper bound does not exceed `eps`, the scan is proved; if the minimal lower bound exceeds `eps` it is disproved; otherwise it is inconclusive.

### Identity verification
The `verify` task compares two user-supplied expressions on a small grid. It will attempt a symbolic proof if SymPy is installed; otherwise it falls back to numeric evaluation using high-precision arithmetic. When evaluating expressions containing lambdas, remember that free variables inside the lambda are not available in the restricted evaluation environment. Capture external variables and functions as default arguments, for example:

```python
# Check that e^x equals its 20-term Maclaurin series on [0, 1]
from gedc_rcp import solve

expr1 = "mp.e**x"
# capture x and factorial as default parameters so the lambda sees them
expr2 = "nsum(lambda k, y=x, fac=factorial: y**k / fac(k), [0, 20])"

res = solve({
    "type": "verify",
    "expr1": expr1,
    "expr2": expr2,
    "domain": {"x": [0, 1]},
    "eps": 1e-8,
})
assert res["status"] == "proved"
```

### Deterministic manifests
Every task returns a manifest dictionary recording the RCP version, current precision, UTC time and a list of public names exported by the core. A SHA-1 digest of the manifest is included to help detect accidental changes.

## Installation

This project requires Python 3.8 or higher. Install the runtime dependencies with:

```bash
pip install -r requirements.txt
```

The only mandatory dependency is `mpmath` for high-precision interval arithmetic. If SymPy is available, the `verify` task will use it for quick symbolic proofs.

The file `gedc_core.py` in this repository is a stub containing only a golden ratio function and an example function. Replace it with your own implementation when integrating with a full GEDC core.

## Usage

Import the `solve` function from `gedc_rcp` and construct a task dictionary. The `type` field selects which primitive to invoke and the remaining keys specify parameters.

### Command-line interface

For convenience the repository includes a small CLI wrapper. It accepts a JSON task either via a `--task` argument, from a file or from standard input and prints the result as JSON. Use the `--pretty` flag to pretty-print the response.

```bash
python gedc_rcp_cli.py --task '{"type": "bound", "which": "tr", "x": 1}' --pretty

python gedc_rcp_cli.py --file mytask.json

# Read from stdin
echo '{"type": "verify", "expr1": "mp.e**x", "expr2": "nsum(lambda k, y=x, fac=factorial: y**k/fac(k), [0, 20])", "domain": {"x": [0, 1]}}' | \
  python gedc_rcp_cli.py --pretty
```

### Bound a series

```python
from gedc_rcp import solve

# Bound the positive series tr(x) for x = 1 with 150 terms
result = solve({
    "type": "bound",
    "which": "tr",   # or "tr_alt" for the alternating version
    "x": 1.0,
    "N": 150,
    "eps": 1e-8,
})

if result["status"] == "proved":
    interval = result["certificate"]["value_interval"]
    print(f"tr(1) lies in {interval} with tail ≤ eps")
else:
    print("Tail bound too big – increase N or eps")
```

### Scan reciprocity

```python
from gedc_rcp import solve

task = {
    "type": "scan",
    "alpha": 0.5,
    "t_grid": [0.5, 1.0, 2.0, 4.0],
    "N": 120,
    "eps": 0.2,
}
res = solve(task)
print(res["status"])
print(res["certificate"]["bounds"])
```

The resulting certificate contains the maximal and minimal residual interval bounds, a list of per-grid traces (capped to 200 entries), and the worst offending grid point. If the scan is inconclusive, consider increasing N (more terms) or adjusting the grid.

### Verify an identity

As shown above, provide two expressions as strings together with a domain over which to test equality. To verify Euler's identity e^x = ∑_{k=0}^∞ x^k/k! on [0,1] using a 20-term truncation:

```python
from gedc_rcp import solve

expr1 = "mp.e**x"
expr2 = "nsum(lambda k, y=x, fac=factorial: y**k / fac(k), [0, 20])"

res = solve({
    "type": "verify",
    "expr1": expr1,
    "expr2": expr2,
    "domain": {"x": [0, 1]},
    "eps": 1e-8,
})
print(res["status"])
print(res.get("witness"))
```

If SymPy is installed and can symbolically simplify the difference to zero, the proof is immediate. Otherwise a 5-point grid is sampled; if the maximum absolute difference is ≤ eps the result is proved, else it is disproved and a witness grid point is returned.

## Certificates and statuses

Each task returns a dictionary with at least the fields:
- **status** – one of `proved`, `disproved` or `inconclusive`.
- **manifest** – a deterministic manifest summarising the run (version, precision, time, exported names and SHA-1).

Additional fields depend on the task:
- **certificate** – structured data summarising the proof, including tail bounds, residual intervals, worst points and traces.
- **witness** – for `verify` and `scan` tasks, the grid point where the maximal difference occurs (useful when the statement is false or cannot be proved at the requested precision).

## Contributing

Contributions are welcome! Feel free to file issues or pull requests to suggest improvements. The current implementation is deliberately minimal and experimental – documentation, testing and additional tasks are especially appreciated.

## License

This project is distributed under the terms of the MIT License. See LICENSE for details.
