# GEDC Reasoning Co-Processor (RCP)

A certified numerical reasoning toolkit with interval arithmetic, JSON certificates, and deterministic manifests for reproducibility. Originally built around theta-function verification, now extended to **quantum computing simulation with noise channels and error correction**.

**Author:** Lucas Postma ([@BeingAsSuch](https://x.com/BeingAsSuch))  
**Organization:** DataDyne Solutions LLC  
**License:** MIT

---

## Overview

This repository provides two main components:

1. **GEDC-RCP (Python)** — Certified theta-function verifier with interval arithmetic
2. **Quantum Certainty Engine (React)** — Browser-based quantum simulator with density matrices, noise channels, and QEC

Both share a core philosophy: **certified bounds, not point estimates**. Every computation returns interval bounds that rigorously contain the true value.

---

## Quantum Certainty Engine

The Quantum Certainty Engine (QCE) is a React artifact that simulates quantum circuits with real noise models and quantum error correction, using complex interval arithmetic to provide certified fidelity bounds.

### Features

**Complex Interval Arithmetic**
- Each amplitude represented as a box on the complex plane: `[re_lo, re_hi] + i[im_lo, im_hi]`
- Rigorous propagation of uncertainty through all operations

**Density Matrix Formalism**
- Mixed states via density matrices ρ
- Trace preservation verification (Tr(ρ) = 1)
- Fidelity computation with certified bounds

**Noise Channels**

| Channel | Model | Formula |
|---------|-------|---------|
| Depolarizing | Random Pauli errors | ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ) |
| Amplitude Damping | T1 decay | Kraus operators K₀, K₁ with γ parameter |
| Dephasing | T2 decay | ρ → (1-p)ρ + pZρZ |

**Quantum Error Correction**
- 3-qubit bit-flip code implementation
- Encoding: |0⟩ → |000⟩, |1⟩ → |111⟩
- Syndrome measurement via Z₀Z₁ and Z₁Z₂ parity checks
- Single bit-flip error correction

**Scenarios**
- No Protection — baseline fidelity loss
- 3-Qubit Bit-Flip Code — QEC demonstration
- Bell State with Noise — entanglement degradation
- T1 Amplitude Damping — energy relaxation
- T2 Dephasing — phase randomization
- QEC Comparison — side-by-side with/without error correction

### What It Proves

With the QEC Comparison scenario at 5% error rate:

```
WITHOUT QEC: ~85% fidelity [0.83, 0.87]
WITH QEC:    ~95% fidelity [0.93, 0.97]
IMPROVEMENT: +10% (certified interval bounds)
```

This demonstrates that the 3-qubit bit-flip code provides **certified improvement** under depolarizing noise, with rigorous bounds on the fidelity gain.

### Limitations

The Quantum Certainty Engine is a **certified classical simulator**. It does not:
- Run on actual quantum hardware
- Replace real QEC implementations (surface codes, etc.)
- Model all physical noise sources
- Scale beyond small qubit counts (browser-based)

It **does** provide:
- Certified bounds on simulated quantum states
- Educational demonstration of noise and QEC concepts
- Validation that simulation code is numerically correct
- Rigorous fidelity bounds under uncertain error rates

---

## GEDC-RCP (Python)

The original Python implementation for certified theta-function verification.

### Features

**Interval-bound series**  
Compute partial sums of tr(x) = Σ e^{-an²x} and its alternating version with rigorous tail bounds.

**Theta reciprocity scans**  
Certify |t^α θ(t) - θ(1/t)| ≤ ε across a grid of points.

**Identity verification**  
Compare expressions on a grid with optional SymPy symbolic proof.

**Deterministic manifests**  
SHA-1 digests for reproducibility.

### Installation

```bash
pip install -r requirements.txt
```

Required: `mpmath`. Optional: `sympy` for symbolic proofs.

### Usage

```python
from gedc_rcp import solve

# Bound the positive series tr(x) for x = 1
result = solve({
    "type": "bound",
    "which": "tr",
    "x": 1.0,
    "N": 150,
    "eps": 1e-8,
})

if result["status"] == "proved":
    interval = result["certificate"]["value_interval"]
    print(f"tr(1) lies in {interval}")
```

### Command-line Interface

```bash
python gedc_rcp_cli.py --task '{"type": "bound", "which": "tr", "x": 1}' --pretty
```

### Scan Reciprocity

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

### Verify an Identity

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
```

---

## Architecture

```
gedc-rcp/
├── gedc_rcp.py                      # Python RCP implementation
├── gedc_core.py                     # Core stub (replace with your implementation)
├── gedc_rcp_cli.py                  # CLI wrapper
├── quantum-certainty-engine.jsx     # React quantum simulator (v1 - ideal circuits)
├── quantum-certainty-engine-v2.jsx  # React quantum simulator (v2 - noise + QEC)
├── math-certainty-engine.jsx        # React classical math verifier
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Certificates and Statuses

All tasks return:
- **status** — `proved`, `disproved`, or `inconclusive`
- **manifest** — version, precision, time, SHA-1 digest
- **certificate** — structured proof data (bounds, traces, witnesses)

---

## Citation

```bibtex
@software{gedc_rcp2025,
  author = {Lucas Postma},
  title = {GEDC-RCP: Certified Numerical Reasoning \& Quantum Simulation},
  year = {2025},
  url = {https://github.com/DataDyneSolutions/gedc-rcp}
}
```

For derivative works, please include:

> Based on GEDC-RCP by Lucas Postma ([@BeingAsSuch](https://x.com/BeingAsSuch)), 2025

---

## Contributing

Contributions welcome! Areas of interest:
- Additional noise channels (crosstalk, leakage, measurement error)
- More QEC codes (Steane [[7,1,3]], surface code)
- Threshold analysis and fault tolerance
- Performance optimization
- Documentation and testing

---

## License

MIT License. See LICENSE for details.

---

**Note:** The Quantum Certainty Engine demonstrates certified classical simulation of quantum systems with noise. It is an educational and research tool, not a replacement for real quantum error correction on physical hardware. See the "Limitations" section for honest positioning of what this tool does and does not do.
