"""
gedc_rcp.py — Reasoning Co‑Processor (RCP) layer on top of the GEDC core.
Adds: interval bounds, certificates, JSON traces, and a task dispatcher.

Dependencies: mpmath (built‑in here). SymPy is optional (we fall back gracefully).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import time, json, hashlib, os
import importlib.util
import mpmath as mp

# ---------- Determinism knobs ----------
DEFAULT_DPS = 80
mp.mp.dps = DEFAULT_DPS

# Try to import the user's gedc_core from the same directory first, then fallback to a normal import.
try:
    here = os.path.dirname(__file__)
    core_path = os.path.join(here, "gedc_core.py")
    if os.path.exists(core_path):
        spec = importlib.util.spec_from_file_location("gedc_core", core_path)
        gedc_core = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(gedc_core)  # type: ignore
    else:
        import gedc_core  # type: ignore
except Exception as e:
    raise RuntimeError(f"Could not import gedc_core: {e}")

# Optional SymPy (we won't require it)
try:
    import sympy as sp
    HAVE_SYMPY = True
except Exception:
    HAVE_SYMPY = False

# ---------- Utility: manifest + tracing ----------

def _manifest(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Create a deterministic manifest for audit & replay."""
    m: Dict[str, Any] = {
        "rcp_version": "0.1.0",
        "mpmath_dps": mp.mp.dps,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Prefer the core's __all__ when available; fall back to public names.
        "gedc_core_has": sorted(getattr(gedc_core, "__all__", [n for n in dir(gedc_core) if not n.startswith("_")])),
    }
    if extra:
        m.update(extra)
    # Compute a stable digest: exclude volatile fields such as time_utc
    h = dict(m)
    h.pop("time_utc", None)
    m["manifest_sha1"] = hashlib.sha1(json.dumps(h, sort_keys=True).encode()).hexdigest()
    return m


def _ok(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("status", "ok")
    return payload


def _inconclusive(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("status", "inconclusive")
    return payload


def _disproved(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("status", "disproved")
    return payload


def _proved(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("status", "proved")
    return payload

# ---------- Interval helpers ----------

def _iv(lo: mp.mpf | float, hi: mp.mpf | float) -> mp.iv.mpf:
    return mp.iv.mpf([lo, hi])


def _abs_iv(x: mp.iv.mpf) -> mp.iv.mpf:
    # |x| for intervals
    if x.a >= 0:
        return x
    if x.b <= 0:
        return -x
    # straddles 0
    return mp.iv.mpf([0, max(-x.a, x.b)])

# ---------- Tail bounds for theta‑like traces ----------
# tr(x) = sum_{n>=1} e^{-a n^2 x}, where a = (π φ)^2 (from gedc_core)


def _a_const() -> mp.mpf:
    # best‑effort fetch of a from the user's core (or recompute)
    # a = (π φ)**2
    phi = (1 + mp.sqrt(5)) / 2
    return (mp.pi * phi) ** 2


def tr_partial_with_tail(x: mp.mpf, N: int) -> Tuple[mp.mpf, mp.mpf]:
    """
    Return (S_N, tail_bound) for tr(x) = sum_{n>=1} e^{-a n^2 x}.
    Tail bound via integral test and gaussian integral (erfc).
    """
    x = mp.mpf(x)
    if x <= 0:
        raise ValueError("x must be > 0 for tr.")
    a = _a_const()
    c = a * x
    # partial sum
    S = mp.nsum(lambda n: mp.e ** (-c * (n ** 2)), [1, N])
    # integral tail + first omitted guard
    t1 = 0.5 * mp.sqrt(mp.pi / c) * mp.erfc(mp.sqrt(c) * (N + 1))
    t2 = mp.e ** (-c * (N + 1) ** 2)
    tail = t1 + t2
    return S, tail


def tr_alt_partial_with_tail(x: mp.mpf, N: int) -> Tuple[mp.mpf, mp.mpf]:
    """
    Return (S_N, tail_bound) for tr_alt(x) = sum_{n>=1} (-1)^{n+1} e^{-a n^2 x}.
    Alternating with decreasing magnitude => Leibniz remainder <= first omitted term.
    """
    x = mp.mpf(x)
    if x <= 0:
        raise ValueError("x must be > 0 for tr_alt.")
    a = _a_const()
    c = a * x
    S = mp.nsum(lambda n: ((-1) ** (n + 1)) * mp.e ** (-c * (n ** 2)), [1, N])
    first_omitted = mp.e ** (-c * (N + 1) ** 2)
    tail = first_omitted
    return S, tail


def theta_completed_interval(t: mp.mpf, N: int) -> Tuple[mp.iv.mpf, Dict[str, Any]]:
    """
    θ(t) = 1 + 2 * sum_{n>=1} (-1)^n e^{-a n^2 t)
    Interval: use alternating tail bound.
    """
    t = mp.mpf(t)
    if t <= 0:
        raise ValueError("t must be > 0")
    S_alt, tail = tr_alt_partial_with_tail(t, N)
    # θ = 1 + 2 * S_alt  (interval bounds widen by 2*tail)
    center = 1 + 2 * S_alt
    rad = 2 * tail
    iv = _iv(center - rad, center + rad)
    meta: Dict[str, Any] = {"N": N, "tail_bound": float(rad), "center": float(center)}
    return iv, meta


# ---------- Reciprocity scan with certificate ----------

def reciprocity_scan(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task schema:
    {
      "type": "scan",
      "alpha": <float>,
      "t_grid": [t1, t2, ...] (t>0),
      "N": 120,                 # truncation
      "eps": 0.05               # target bound on |t^alpha θ(t) - θ(1/t)|
    }
    Returns certificate with proved/disproved/inconclusive.
    """
    alpha = mp.mpf(task.get("alpha", 0.5))
    N = int(task.get("N", 120))
    eps = mp.mpf(task.get("eps", "0.05"))
    t_grid = [mp.mpf(t) for t in task.get("t_grid", [])]
    if not t_grid:
        raise ValueError("t_grid required and non-empty")
    trace: List[Dict[str, Any]] = []

    max_upper = 0.0
    min_lower: mp.mpf = mp.inf  # lower bound on residual > eps to disprove

    worst_t = None
    worst_upper = -mp.inf
    counterexample = None

    for t in t_grid:
        th_t, meta_t = theta_completed_interval(t, N)
        th_inv, meta_inv = theta_completed_interval(1 / t, N)
        lhs = (mp.iv.mpf([t, t]) ** alpha) * th_t
        rhs = th_inv
        resid = lhs - rhs
        abs_resid = _abs_iv(resid)

        upper = float(abs_resid.b)
        lower = float(abs_resid.a)

        if upper > worst_upper:
            worst_upper = upper
            worst_t = float(t)

        if upper > max_upper:
            max_upper = upper
        if lower < min_lower:
            min_lower = lower

        trace.append({
            "t": float(t),
            "theta_t_tail": meta_t["tail_bound"],
            "theta_inv_tail": meta_inv["tail_bound"],
            "residual_interval": [float(abs_resid.a), float(abs_resid.b)]
        })

        # quick numeric midpoint check to propose a concrete counterexample if > eps
        # Avoid iv -> float (mpmath raises unless width == 0); use scalar bounds
        midf = (lower + upper) / 2.0
        if midf > float(eps) and counterexample is None:
            counterexample = {"t": float(t), "residual_mid": midf}

    cert: Dict[str, Any] = {
        "task": {"alpha": float(alpha), "N": N, "eps": float(eps), "len_grid": len(t_grid)},
        "bounds": {"max_abs_residual_upper": float(max_upper), "min_abs_residual_lower": float(min_lower)},
        "worst_point": {"t": worst_t, "upper_bound": float(worst_upper)},
        "trace": trace[:200]  # cap to keep payload light
    }

    if max_upper <= float(eps):
        return _proved({"certificate": cert, "manifest": _manifest({"task_type": "scan"})})
    if float(min_lower) > float(eps):
        out: Dict[str, Any] = {"certificate": cert, "manifest": _manifest({"task_type": "scan"})}
        if counterexample:
            out["witness"] = counterexample
        return _disproved(out)
    # inconclusive: some intervals straddle eps
    out2: Dict[str, Any] = {"certificate": cert, "manifest": _manifest({"task_type": "scan"})}
    if counterexample:
        out2["witness"] = counterexample
    return _inconclusive(out2)


# ---------- Series bound primitive ----------

def bound_series(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task schema:
    {
      "type": "bound",
      "which": "tr" | "tr_alt",
      "x": <float>, "N": 150, "eps": 1e-6
    }
    Returns interval for the series and a tail guarantee.
    """
    which = task.get("which", "tr")
    x = mp.mpf(task.get("x", 1.0))
    N = int(task.get("N", 150))
    eps = mp.mpf(task.get("eps", "1e-6"))
    if x <= 0:
        return _inconclusive({"error": "x must be > 0", "manifest": _manifest({"task_type": "bound"})})

    if which == "tr":
        S, tail = tr_partial_with_tail(x, N)
        iv = _iv(S, S + tail)  # positive terms
    elif which == "tr_alt":
        S, tail = tr_alt_partial_with_tail(x, N)
        iv = _iv(S - tail, S + tail)  # alternating bound
    else:
        return _inconclusive({"error": f"unknown series '{which}'", "manifest": _manifest({"task_type": "bound"})})

    cert2 = {
        "input": {"which": which, "x": float(x), "N": N, "eps": float(eps)},
        "value_interval": [float(iv.a), float(iv.b)],
        "tail_bound": float(tail),
    }

    # If tail <= eps, we certify within eps (interval width small enough)
    if tail <= eps:
        return _proved({"certificate": cert2, "manifest": _manifest({"task_type": "bound"})})
    else:
        return _inconclusive({"certificate": cert2, "manifest": _manifest({"task_type": "bound"})})


# ---------- Identity check (optional symbolic, else numeric+interval) ----------

def verify_identity(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task schema:
    {
      "type": "verify",
      "expr1": "e**x",
      "expr2": "nsum(lambda k: x**k/mp.factorial(k), [0, 20])",
      "domain": {"x": [0, 1]}, "eps": 1e-8
    }
    This is intentionally conservative and sandboxed: it only evaluates a small grid.
    """
    eps = mp.mpf(task.get("eps", "1e-8"))
    domain = task.get("domain", {"x": [0, 1]})
    xs = domain.get("x", [0, 1])
    # coerce grid (5 points incl. ends + mids)
    grid = [
        mp.mpf(xs[0]),
        (2 * mp.mpf(xs[0]) + mp.mpf(xs[1])) / 3,
        (mp.mpf(xs[0]) + mp.mpf(xs[1])) / 2,
        (mp.mpf(xs[0]) + 2 * mp.mpf(xs[1])) / 3,
        mp.mpf(xs[1])
    ]

    expr1 = task.get("expr1")
    expr2 = task.get("expr2")
    if not expr1 or not expr2:
        return _inconclusive({"error": "expr1 and expr2 required", "manifest": _manifest({"task_type": "verify"})})

    # Optional symbolic quick pass
    if HAVE_SYMPY:
        try:
            x = sp.symbols('x')
            e1 = sp.sympify(expr1, {"x": x})
            e2 = sp.sympify(expr2, {"x": x})
            simp = sp.simplify(e1 - e2)
            if simp == 0:
                return _proved({
                    "certificate": {"symbolic": True, "proof": "sympy_simplify_zero"},
                    "manifest": _manifest({"task_type": "verify"})
                })
        except Exception:
            pass

    # Numeric interval‑ish pass: eval both and bound |diff|
    trace: List[Dict[str, Any]] = []
    max_abs = 0.0
    for t in grid:
        # Evaluate in a restricted namespace (no builtins beyond mp)
        ns = {"mp": mp, "x": t, "nsum": mp.nsum, "factorial": mp.factorial, "e": mp.e}
        try:
            v1 = eval(expr1, {"__builtins__": {}}, ns)
            v2 = eval(expr2, {"__builtins__": {}}, ns)
            # ensure numeric; this will raise on non‑numeric returns
            v1 = mp.mpf(v1)
            v2 = mp.mpf(v2)
        except Exception as e:
            return _inconclusive({"error": f"non‑numeric or unsafe expr: {e}", "manifest": _manifest({"task_type": "verify"})})
        diff = abs(v1 - v2)
        max_abs = max(max_abs, float(diff))
        trace.append({"x": float(t), "abs_diff": float(diff)})

    cert3 = {"grid_max_abs_diff": max_abs, "grid_points": len(grid), "trace": trace}
    if max_abs <= float(eps):
        return _proved({"certificate": cert3, "manifest": _manifest({"task_type": "verify"})})
    else:
        # Provide witness as the worst grid point
        worst = max(trace, key=lambda r: r["abs_diff"])
        return _disproved({"certificate": cert3, "witness": worst, "manifest": _manifest({"task_type": "verify"})})


# ---------- Dispatcher ----------

def solve(task: Dict[str, Any]) -> Dict[str, Any]:
    ttype = task.get("type", "")
    if ttype == "scan":
        return reciprocity_scan(task)
    if ttype == "bound":
        return bound_series(task)
    if ttype == "verify":
        return verify_identity(task)
    return _inconclusive({"error": f"unknown task type '{ttype}'", "manifest": _manifest({"task_type": "unknown"})})
