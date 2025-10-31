import streamlit as st
import numpy as np
import pandas as pd
import numexpr as ne
import plotly.graph_objects as go
from scipy import stats
from typing import Dict, Any, Tuple, List
import io
import re
import unicodedata
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Optional SciPy imports
try:
    from scipy import stats as _scipy_stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
    stats = _scipy_stats
except Exception:
    SCIPY_AVAILABLE = False

# Seaborn palette for Plotly
PALETTE = sns.color_palette("deep", 8).as_hex()

# Fast approximate mode for continuous distributions
def approx_mode(x, bins=100):
    counts, edges = np.histogram(x, bins=bins)
    i = np.argmax(counts)
    return 0.5 * (edges[i] + edges[i + 1])

# Safe constants available in expressions
SAFE_CONST = {"pi": np.pi, "e": np.e, "inf": np.inf, "nan": np.nan}

def _slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    if not re.match(r"^[a-z_]", s):
        s = "r_" + s
    return s or "result"

# Check for openpyxl availability
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

st.set_page_config(page_title="ProbCalcMC", layout="wide")
st.title("ProbCalcMC – Custom Monte Carlo simulation")
st.markdown("<div style='margin-top:-0.5rem; font-size:0.9rem; color:#666'><em>by Lars Hjelm</em></div>", unsafe_allow_html=True)
# Enlarge primary buttons slightly
st.markdown("""
<style>
div.stButton > button[kind="primary"] {
  font-size: 1.05rem;
  padding: 0.6rem 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar: global controls ---
with st.sidebar:
    st.header("Settings")
    n_samples = st.number_input("Number of Monte Carlo samples", 1000, 2_000_000, 50_000, step=1000, help="How many draws to simulate for each variable.")
    seed = st.number_input("Random seed (optional)", value=0, min_value=0, step=1)
    if seed:
        np.random.seed(int(seed))

    st.markdown("---")
    st.subheader("Variables")

# Up to 256 symbols a..z, aa..az, ba..bz ... up to 256

def make_symbols(n: int) -> List[str]:
    base = [chr(ord('a') + i) for i in range(26)]
    if n <= 26:
        return base[:n]
    syms = base.copy()
    prefix_idx = 0
    while len(syms) < n and prefix_idx < 26:
        for b in base:
            syms.append(base[prefix_idx] + b)
            if len(syms) == n:
                break
        prefix_idx += 1
    return syms[:n]

# Distribution registry (21+ commonly used)
DISTROS = {
    "Constant": {"params": [("value", 1.0, float)]},
    "Normal": {"params": [("mean", 5.0, float), ("sd", 1.0, float)]},
    "Lognormal": {"params": [("mu (log-mean)", 1.0, float), ("sigma (log-sd)", 1.0, float)]},
    "Uniform": {"params": [("low", 0.0, float), ("high", 10.0, float)]},
    "Triangular": {"params": [("low", 0.0, float), ("mode", 5.0, float), ("high", 10.0, float)]},
    "PERT": {"params": [("min", 0.0, float), ("most_likely", 5.0, float), ("max", 10.0, float), ("lambda (shape)", 4.0, float)]},
    "Subjective Beta": {"params": [("min", 0.0, float), ("max", 10.0, float), ("p10 (10th percentile)", 2.0, float), ("p50 (median)", 5.0, float), ("p90 (90th percentile)", 8.0, float)]},
    "Bernoulli": {"params": [("p", 0.5, float)]},
    "Binomial": {"params": [("n", 10, int), ("p", 0.5, float)]},
    "Poisson": {"params": [("lam", 5.0, float)]},
    "Exponential": {"params": [("rate (1/scale)", 1.0, float)]},
    "Gamma": {"params": [("shape", 2.0, float), ("scale", 2.0, float)]},
    "Beta": {"params": [("alpha", 2.0, float), ("beta", 2.0, float)]},
    "Weibull": {"params": [("shape (k)", 1.5, float), ("scale (lambda)", 5.0, float)]},
    "Geometric": {"params": [("p", 0.5, float)]},
    "Pareto": {"params": [("shape (a)", 3.0, float), ("scale (xm)", 1.0, float)]},
    "StudentT": {"params": [("df", 5.0, float)]},
    "Cauchy": {"params": [("x0", 0.0, float), ("gamma", 1.0, float)]},
    "Laplace": {"params": [("loc", 5.0, float), ("scale", 1.0, float)]},
    "Erlang": {"params": [("k (integer)", 2, int), ("rate", 1.0, float)]},
    "Discrete": {"params": [("values (comma-separated)", "1,5,10", str), ("weights (comma-sep, optional)", "", str)]},
    "TruncNormal": {"params": [("mean", 5.0, float), ("sd", 1.0, float), ("low", -1e10, float), ("high", 1e10, float)]},
    "TruncLognormal": {"params": [("mu", 1.0, float), ("sigma", 1.0, float), ("low", 0.0, float), ("high", 1e10, float)]},
}

# Add StretchBeta (PERT-style stretch beta)
DISTROS["StretchBeta"] = {
    "params": [
        ("min", 0.0, float),
        ("mode", 0.5, float),
        ("max", 1.0, float),
        ("lambda (shape)", 4.0, float),
    ]
}

HELP_SAFE_FUNCTIONS = """
**Build custom formulas using your variables and mathematical functions:**

**Available Operators:** `+`, `-`, `*`, `/`, `**` (power), `()` (parentheses)

**Available Functions:** `abs`, `sqrt`, `exp`, `log`, `log10`, `min`, `max`, `where`, `clip`, `sin`, `cos`, `tan`, `floor`, `ceil`, `round`

**Examples:**
- **Sum**: `a + b + c` - add variables together
- **Product**: `a * b * c` - multiply variables
- **Ratio**: `a / b` - divide one by another
- **Power**: `a**2` - square a variable (use ** for exponentiation)
- **Complex**: `(a + b) / max(c, 1e-6)` - add a and b, then divide by the maximum of c or 0.000001
- **Conditional**: `where(a > 0, a * b, a / b)` - if a > 0, multiply by b, else divide by b
- **Safe division**: `a / max(b, 0.001)` - avoids division by zero
- **Absolute value**: `abs(a - b)` - absolute difference
- **Square root**: `sqrt(a * b)` - square root of product
- **Trigonometry**: `sin(a) + cos(b)` - use trigonometric functions
- **Minimum/Maximum**: `min(a, b, c)` or `max(a, b, c)` - find extreme values

**Tip**: Use variables `a`, `b`, `c`, etc. in your formula (as defined in the sidebar). The display below will show their full names.

**Referencing earlier results (derived variables):**
- Formulas are evaluated top-to-bottom. After **Formula 1** is computed, you can reference it as **`f1`** in later formulas; Formula 2 becomes **`f2`**, etc.
- Each result is also exposed by its name as **`res_<slug>`**, where `<slug>` is your result name lowercased, spaces → underscores, and non-alphanumerics removed.
  - Example: result name `Net Profit (EUR)` becomes `res_net_profit_eur`.
- Variable names from the sidebar are also available as aliases (slugified), e.g., Name `Bus` → `bus`.
- Example usage:
  - Formula 1: `profit = revenue - cost`
  - Formula 2: `margin = f1 / max(revenue, 1e-9)` (uses Formula 1)
  - Formula 3: `kpi = res_profit / res_margin`

Referencing earlier formulas:

Use f1, f2, ... to refer to Formula 1, Formula 2, etc.
Use res_<slug> to refer to a result by name (lowercase, spaces→underscore, symbols removed). Example: Result name "Net Profit (EUR)" → res_net_profit_eur
Variable names from the sidebar are also available as aliases (slugified), e.g. Name "Bus" → bus.
"""

SAFE_FUNCS = {
    'abs': np.abs,
    'sqrt': np.sqrt,
    'exp': np.exp,
    'log': np.log,
    'log10': np.log10,
    'min': np.minimum,
    'max': np.maximum,
    'where': np.where,
    'clip': np.clip,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'floor': np.floor,
    'ceil': np.ceil,
    'round': np.round,
}

# --- Helpers to sample distributions ---

def sample_distribution(kind: str, params: Dict[str, Any], n: int) -> np.ndarray:
    if kind == "Constant":
        return np.full(n, float(params["value"]))
    elif kind == "Normal":
        sd = float(params["sd"])
        if sd <= 0:
            raise ValueError("Normal: sd must be > 0.")
        return np.random.normal(float(params["mean"]), sd, size=n)
    elif kind == "Lognormal":
        sigma = float(params["sigma (log-sd)"])
        if sigma <= 0:
            raise ValueError("Lognormal: sigma (log-sd) must be > 0.")
        return np.random.lognormal(float(params["mu (log-mean)"]), sigma, size=n)
    elif kind == "Uniform":
        low, high = float(params["low"]), float(params["high"])
        if not (low < high):
            raise ValueError("Uniform: low must be < high.")
        return np.random.uniform(low, high, size=n)
    elif kind == "Triangular":
        low, mode, high = float(params["low"]), float(params["mode"]), float(params["high"]) 
        if not (low <= mode <= high):
            raise ValueError("Triangular: require low ≤ mode ≤ high.")
        if low == high:
            return np.full(n, low)
        return np.random.triangular(low, mode, high, size=n)
    elif kind == "PERT":
        a, m, b = float(params["min"]), float(params["most_likely"]), float(params["max"]) 
        lam = float(params["lambda (shape)"])
        if not (a <= m <= b):
            raise ValueError("PERT: require min ≤ most_likely ≤ max.")
        if a == b:
            return np.full(n, a)
        if lam <= 0:
            raise ValueError("PERT: lambda (shape) must be > 0.")
        # Beta-PERT: alpha = 1 + lambda*(m-a)/(b-a), beta = 1 + lambda*(b-m)/(b-a)
        alpha = 1 + lam*(m - a)/(b - a)
        beta = 1 + lam*(b - m)/(b - a)
        x = np.random.beta(alpha, beta, size=n)
        return a + x*(b - a)
    elif kind == "Subjective Beta":
        lo, hi = float(params["min"]), float(params["max"])
        p10 = float(params["p10 (10th percentile)"])
        p50 = float(params["p50 (median)"])
        p90 = float(params["p90 (90th percentile)"])
        if not (lo < hi):
            raise ValueError("Subjective Beta: min must be < max.")
        # normalize to [0,1]
        def nz(x): return (x - lo) / (hi - lo)
        q10, q50, q90 = map(nz, (p10, p50, p90))
        q10 = np.clip(q10, 1e-6, 1 - 1e-6)
        q50 = np.clip(q50, 1e-6, 1 - 1e-6)
        q90 = np.clip(q90, 1e-6, 1 - 1e-6)
        if SCIPY_AVAILABLE:
            def loss(ab):
                a, b = ab
                if a <= 0 or b <= 0:
                    return 1e9
                return (
                    (stats.beta.ppf(0.10, a, b) - q10) ** 2 +
                    (stats.beta.ppf(0.50, a, b) - q50) ** 2 +
                    (stats.beta.ppf(0.90, a, b) - q90) ** 2
                )
            res = minimize(loss, x0=[2.0, 2.0], bounds=[(1e-3, None), (1e-3, None)])
            a, b = (2.0, 2.0) if (not res.success) else res.x
        else:
            # Fallback: PERT-style using an estimated mode from quantiles (crude)
            mode_est = np.clip(0.25 * q10 + 0.5 * q50 + 0.25 * q90, 1e-6, 1 - 1e-6)
            lam = 4.0
            a = 1 + lam * mode_est
            b = 1 + lam * (1.0 - mode_est)
        x = np.random.beta(a, b, size=n)
        return lo + x * (hi - lo)
    elif kind == "Bernoulli":
        p = float(params["p"]) 
        if not (0.0 <= p <= 1.0):
            raise ValueError("Bernoulli: p must be between 0 and 1.")
        return np.random.binomial(1, p, size=n)
    elif kind == "Binomial":
        nn, p = int(params["n"]), float(params["p"]) 
        if nn < 0:
            raise ValueError("Binomial: n must be ≥ 0.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("Binomial: p must be between 0 and 1.")
        return np.random.binomial(nn, p, size=n)
    elif kind == "Poisson":
        lam = float(params["lam"]) 
        if lam < 0:
            raise ValueError("Poisson: lambda must be ≥ 0.")
        return np.random.poisson(lam, size=n)
    elif kind == "Exponential":
        rate = float(params["rate (1/scale)"])
        if rate <= 0:
            raise ValueError("Exponential: rate must be > 0.")
        return np.random.exponential(1.0/rate, size=n)
    elif kind == "Gamma":
        shape, scale = float(params["shape"]), float(params["scale"]) 
        if shape <= 0 or scale <= 0:
            raise ValueError("Gamma: shape and scale must be > 0.")
        return np.random.gamma(shape, scale, size=n)
    elif kind == "Beta":
        a, b = float(params["alpha"]), float(params["beta"]) 
        if a <= 0 or b <= 0:
            raise ValueError("Beta: alpha and beta must be > 0.")
        return np.random.beta(a, b, size=n)
    elif kind == "Weibull":
        k, lam = float(params["shape (k)"]), float(params["scale (lambda)"]) 
        if k <= 0 or lam <= 0:
            raise ValueError("Weibull: shape k and scale lambda must be > 0.")
        return lam * np.random.weibull(k, size=n)
    elif kind == "Geometric":
        p = float(params["p"]) 
        if not (0.0 < p <= 1.0):
            raise ValueError("Geometric: p must be in (0, 1].")
        return np.random.geometric(p, size=n)
    elif kind == "Pareto":
        a, xm = float(params["shape (a)"]), float(params["scale (xm)"]) 
        if a <= 0 or xm <= 0:
            raise ValueError("Pareto: shape a and scale xm must be > 0.")
        return xm * (1 + np.random.pareto(a, size=n))
    elif kind == "StudentT":
        df = float(params["df"]) 
        if df <= 0:
            raise ValueError("StudentT: df must be > 0.")
        return np.random.standard_t(df, size=n)
    elif kind == "Cauchy":
        x0, gamma = float(params["x0"]), float(params["gamma"]) 
        if gamma <= 0:
            raise ValueError("Cauchy: gamma must be > 0.")
        return x0 + gamma * np.random.standard_cauchy(size=n)
    elif kind == "Laplace":
        loc, scale = float(params["loc"]), float(params["scale"]) 
        if scale <= 0:
            raise ValueError("Laplace: scale must be > 0.")
        return np.random.laplace(loc, scale, size=n)
    elif kind == "Erlang":
        k, rate = int(params["k (integer)"]), float(params["rate"]) 
        if k <= 0 or rate <= 0:
            raise ValueError("Erlang: k must be ≥ 1 and rate must be > 0.")
        return np.random.gamma(k, 1.0/rate, size=n)
    elif kind == "Discrete":
        vals = [float(x.strip()) for x in str(params["values (comma-separated)"]).split(',') if x.strip() != ""]
        if not vals:
            raise ValueError("Discrete: provide at least one value.")
        wtxt = str(params["weights (comma-sep, optional)"]).strip()
        if wtxt:
            weights = [float(x.strip()) for x in wtxt.split(',') if x.strip() != ""]
            if len(weights) != len(vals):
                raise ValueError("Discrete: weights length must match values length.")
            if any(w < 0 for w in weights):
                raise ValueError("Discrete: weights must be non-negative.")
            weights = np.array(weights, dtype=float)
            total = weights.sum()
            if total <= 0:
                raise ValueError("Discrete: sum of weights must be > 0.")
            weights = weights / total
        else:
            weights = np.ones(len(vals)) / len(vals)
        return np.random.choice(vals, size=n, p=weights)
    elif kind == "TruncNormal":
        mean, sd, low, high = float(params["mean"]), float(params["sd"]), float(params["low"]), float(params["high"])
        if not (low < high):
            raise ValueError("TruncNormal: low must be < high.")
        if sd <= 0:
            raise ValueError("TruncNormal: sd must be > 0.")
        if SCIPY_AVAILABLE:
            a, b = (low - mean) / sd, (high - mean) / sd
            x = stats.truncnorm.rvs(a, b, loc=mean, scale=sd, size=n)
            return x
        x = np.random.normal(mean, sd, size=n*2)
        x = x[(x >= low) & (x <= high)]
        if x.size < n:
            # Top-up if too few
            extra = np.random.normal(mean, sd, size=n*4)
            extra = extra[(extra >= low) & (extra <= high)]
            x = np.concatenate([x, extra])
        return x[:n]
    elif kind == "TruncLognormal":
        mu, sigma, low, high = float(params["mu"]), float(params["sigma"]), float(params["low"]), float(params["high"])
        if sigma <= 0:
            raise ValueError("TruncLognormal: sigma must be > 0.")
        if not (low < high):
            raise ValueError("TruncLognormal: low must be < high.")
        x = np.random.lognormal(mu, sigma, size=n*2)
        x = x[(x >= low) & (x <= high)]
        if x.size < n:
            extra = np.random.lognormal(mu, sigma, size=n*4)
            extra = extra[(extra >= low) & (extra <= high)]
            x = np.concatenate([x, extra])
        return x[:n]
    elif kind == "StretchBeta":
        lo = float(params["min"]) 
        mode = float(params["mode"]) 
        hi = float(params["max"]) 
        lam = float(params["lambda (shape)"]) 
        if hi == lo:
            return np.full(n, lo)
        if not (lo <= mode <= hi):
            raise ValueError("StretchBeta: mode must lie between min and max.")
        if lam <= 0:
            raise ValueError("StretchBeta: lambda (shape) must be > 0.")
        alpha = 1 + lam * (mode - lo) / (hi - lo)
        beta = 1 + lam * (hi - mode) / (hi - lo)
        x = np.random.beta(alpha, beta, size=n)
        return lo + x * (hi - lo)
    else:
        raise ValueError(f"Unknown distribution: {kind}")

# --- UI to define variables ---
max_vars = 256
num_vars = st.sidebar.number_input("How many variables?", 1, max_vars, 3, step=1)
var_symbols = make_symbols(int(num_vars))

variables_config: Dict[str, Dict[str, Any]] = {}

for sym in var_symbols:
    with st.sidebar.expander(f"{sym}", expanded=False):
        name = st.text_input(f"Name for `{sym}` (optional)", key=f"name_{sym}", help="Enter a descriptive name. Tip: You can copy-paste special characters like Greek letters (α, β, γ, θ, φ, λ, σ, μ, π, ρ) or subscripts/superscripts from online character maps.")
        prob = st.slider(f"Occurrence probability for `{sym}`", 0.0, 1.0, 1.0, 0.01, key=f"prob_{sym}")
        dtype = st.selectbox(f"Distribution type for `{sym}`", list(DISTROS.keys()), index=1 if sym=='a' else 0, key=f"dtype_{sym}")
        params_spec = DISTROS[dtype]["params"]
        param_values = {}
        for label, default, ptype in params_spec:
            if ptype == float:
                val = st.number_input(label + f" ({sym})", value=float(default), key=f"{sym}_{label}")
            elif ptype == int:
                val = st.number_input(label + f" ({sym})", value=int(default), step=1, key=f"{sym}_{label}")
            elif ptype == str:
                val = st.text_input(label + f" ({sym})", value=str(default), key=f"{sym}_{label}")
            else:
                val = st.text_input(label + f" ({sym})", value=str(default), key=f"{sym}_{label}")
            param_values[label] = val
        variables_config[sym] = {"name": name.strip() or sym, "prob": float(prob), "type": dtype, "params": param_values}

# --- Build samples for each variable ---
@st.cache_data(show_spinner=False)
def simulate_variables(config: Dict[str, Dict[str, Any]], n: int, seed_local: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Returns (conditional_samples, unconditional_samples) tuples"""
    if seed_local:
        rng_state = np.random.get_state()
        np.random.seed(seed_local)
    conditional_samples = {}
    unconditional_samples = {}
    for sym, spec in config.items():
        try:
            x_unconditional = sample_distribution(spec["type"], spec["params"], n)
        except ValueError as e:
            raise ValueError(f"Variable '{sym}' ({spec.get('name', sym)}): {e}")
        unconditional_samples[sym] = x_unconditional
        
        # Apply occurrence probability for conditional
        p = float(spec["prob"]) if "prob" in spec else 1.0
        if p < 1.0:
            mask = np.random.binomial(1, p, size=n)
            x_conditional = x_unconditional * mask
        else:
            x_conditional = x_unconditional.copy()
        conditional_samples[sym] = x_conditional
    if seed_local:
        np.random.set_state(rng_state)
    return conditional_samples, unconditional_samples

samples, unconditional_samples = simulate_variables(variables_config, int(n_samples), int(seed))

# --- Show variable distributions with conditional plots ---
if samples:
    st.subheader("Variable Distributions")
    
    with st.expander("Statistical Terminology", expanded=False):
        st.markdown("""
        - **Mean (Arithmetic Mean)**: The average value of all samples. For revenue, it's the expected return.
        - **SD (Standard Deviation)**: A measure of variability. Higher SD means more spread around the mean.
        - **Mode**: Most frequent value.
        - **Skew**: Measures asymmetry. Positive = right tail (high outliers), Negative = left tail (low outliers), Zero = symmetric.
        - **P10 (High Value)**: 90% of outcomes are below this value - represents the high/optimistic scenario.
        - **P50 (Median)**: 50% of values are below/above this - most likely value.
        - **P90 (Low Value)**: Only 10% of outcomes are below this value - represents the low/conservative scenario.
        - **Conditional Distribution**: When occurrence probability < 1, this shows the distribution including zero values (when the event didn't occur).
        - **Unconditional Distribution**: The underlying distribution without application of occurrence probability (as if the event always occurred).
        """)
        
        # Short guide to available distributions
        distro_descriptions = {
            "Constant": "Fixed value.",
            "Normal": "Bell-shaped around mean with standard deviation.",
            "Lognormal": "Right-skewed; log of the variable is Normal.",
            "Uniform": "All values between low and high equally likely.",
            "Triangular": "Defined by low, mode (peak), and high.",
            "PERT": "Smoothed triangular using min, most likely, max, with shape λ.",
            "Subjective Beta": "Beta distribution fit using min/max and P10/P50/P90.",
            "Bernoulli": "0/1 outcome with probability p of 1.",
            "Binomial": "Number of successes in n Bernoulli trials.",
            "Poisson": "Counts of events with average rate λ.",
            "Exponential": "Time between events; mean = 1/rate.",
            "Gamma": "Positive, skewed; shape and scale parameters.",
            "Beta": "Bounded [0,1] shape controlled by α and β.",
            "Weibull": "Flexible positive distribution; reliability/size.",
            "Geometric": "Trials until first success with probability p.",
            "Pareto": "Heavy-tailed; scale xm and shape a.",
            "StudentT": "Heavy-tailed around 0 with degrees of freedom.",
            "Cauchy": "Very heavy-tailed centered at x0 with width γ.",
            "Laplace": "Double-exponential; sharp peak, heavy tails.",
            "Erlang": "Gamma with integer shape (k), rate.",
            "Discrete": "Pick from listed values with optional weights.",
            "TruncNormal": "Normal truncated to [low, high].",
            "TruncLognormal": "Lognormal truncated to [low, high].",
            "StretchBeta": "PERT-style beta scaled to [min, max] with mode and λ."
        }
        lines = ["**Distributions (short guide):**"]
        for name in DISTROS.keys():
            desc = distro_descriptions.get(name, "See parameters.")
            lines.append(f"- **{name}**: {desc}")
        st.markdown("\n".join(lines))
    
    for sym in samples.keys():
        values_cond = samples[sym]
        values_uncond = unconditional_samples[sym]
        
        if len(values_cond) == 0:
            continue
        
        var_name = variables_config[sym]["name"]
        has_occurrence = variables_config[sym].get("prob", 1.0) < 1.0
        
        with st.expander(f"Distribution of {var_name} ({sym})", expanded=False):
            # Calculate statistics for conditional (current)
            mean_cond = float(np.mean(values_cond))
            p10_cond = float(np.percentile(values_cond, 90))  # Inverted
            p50_cond = float(np.percentile(values_cond, 50))
            p90_cond = float(np.percentile(values_cond, 10))  # Inverted
            
            # Calculate statistics for unconditional (without occurrence probability)
            mean_uncond = float(np.mean(values_uncond))
            p10_uncond = float(np.percentile(values_uncond, 90))  # Inverted
            p50_uncond = float(np.percentile(values_uncond, 50))
            p90_uncond = float(np.percentile(values_uncond, 10))  # Inverted
            
            # Display statistics with conditional/unconditional
            if has_occurrence:
                st.markdown("**Conditional Distribution (with occurrence probability):**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{mean_cond:,.3g}")
                with col2:
                    st.metric("P90", f"{p90_cond:,.3g}")
                with col3:
                    st.metric("P50", f"{p50_cond:,.3g}")
                with col4:
                    st.metric("P10", f"{p10_cond:,.3g}")
                
                st.markdown("**Unconditional Distribution (without occurrence probability):**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{mean_uncond:,.3g}")
                with col2:
                    st.metric("P90", f"{p90_uncond:,.3g}")
                with col3:
                    st.metric("P50", f"{p50_uncond:,.3g}")
                with col4:
                    st.metric("P10", f"{p10_uncond:,.3g}")
            else:
                # No occurrence probability, only show one set
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{mean_cond:,.3g}")
                with col2:
                    st.metric("P90", f"{p90_cond:,.3g}")
                with col3:
                    st.metric("P50", f"{p50_cond:,.3g}")
                with col4:
                    st.metric("P10", f"{p10_cond:,.3g}")
            
            # Create plot with histogram and CDF overlaid
            fig = go.Figure()
            
            # Add histograms
            if has_occurrence:
                # Conditional histogram
                fig.add_histogram(
                    x=values_cond,
                    nbinsx=100,
                    name="Conditional Histogram",
                    marker_color=PALETTE[0],
                    opacity=0.6,
                    histnorm='probability density',
                    hovertemplate="%{x:.4g}"
                )
                # Unconditional histogram
                fig.add_histogram(
                    x=values_uncond,
                    nbinsx=100,
                    name="Unconditional Histogram",
                    marker_color=PALETTE[1],
                    opacity=0.4,
                    histnorm='probability density',
                    hovertemplate="%{x:.4g}"
                )
            else:
                # No occurrence probability, only show one histogram
                fig.add_histogram(
                    x=values_cond,
                    nbinsx=100,
                    name=f"{var_name} Histogram",
                    marker_color=PALETTE[0],
                    opacity=0.85,
                    histnorm='probability density',
                    hovertemplate="%{x:.4g}"
                )
            
            # Add vertical lines for P10, P50, Mean, P90 (conditional)
            for val, label, color in [(p10_cond, "P10", "red"), (p50_cond, "P50", "orange"), (mean_cond, "Mean", "green"), (p90_cond, "P90", "blue")]:
                fig.add_vline(
                    x=val,
                    line_width=2,
                    line_dash="dash",
                    line_color=color,
                    opacity=0.75
                )
                fig.add_annotation(
                    x=val,
                    y=1.02,
                    yref="paper",
                    xanchor="center",
                    showarrow=False,
                    text=f"{label}={val:,.3g}",
                    font=dict(color=color, size=10)
                )
            
            if has_occurrence:
                # Add unconditional vertical lines (lighter)
                for val, label, color in [(p10_uncond, "P10", "red"), (p50_uncond, "P50", "orange"), (mean_uncond, "Mean", "green"), (p90_uncond, "P90", "blue")]:
                    fig.add_vline(
                        x=val,
                        line_width=1.5,
                        line_dash="dot",
                        line_color=color,
                        opacity=0.4
                    )
            
            # Calculate CDFs (probability of exceedance: P(Value > threshold))
            sorted_cond = np.sort(values_cond)
            cdf_cond_at_or_below = np.arange(1, len(sorted_cond) + 1) / len(sorted_cond)
            cdf_cond = 1 - cdf_cond_at_or_below  # Invert to show probability of EXCEEDANCE
            
            if has_occurrence:
                sorted_uncond = np.sort(values_uncond)
                cdf_uncond_at_or_below = np.arange(1, len(sorted_uncond) + 1) / len(sorted_uncond)
                cdf_uncond = 1 - cdf_uncond_at_or_below  # Invert to show probability of EXCEEDANCE
                
                # Add unconditional CDF
                fig.add_trace(go.Scatter(
                    x=sorted_uncond,
                    y=cdf_uncond,
                    mode='lines',
                    name="Unconditional CDF",
                    line=dict(color=PALETTE[2], width=3, dash='solid'),
                    opacity=1.0,
                    yaxis="y2",
                    hovertemplate="Value: %{x:.4g}<br>Probability of Exceedance: %{y:.1%}<extra></extra>"
                ))
                
                # Add conditional CDF
                fig.add_trace(go.Scatter(
                    x=sorted_cond,
                    y=cdf_cond,
                    mode='lines',
                    name="Conditional CDF",
                    line=dict(color=PALETTE[1], width=3, dash='dash'),
                    opacity=1.0,
                    yaxis="y2",
                    hovertemplate="Value: %{x:.4g}<br>Probability of Exceedance: %{y:.1%}<extra></extra>"
                ))
                
                # Add markers for both
                for val_cond, val_uncond, label, color in [
                    (p10_cond, p10_uncond, "P10", "red"),
                    (p50_cond, p50_uncond, "P50", "orange"),
                    (mean_cond, mean_uncond, "Mean", "green"),
                    (p90_cond, p90_uncond, "P90", "blue")
                ]:
                    cdf_val_cond_at_or_below = np.searchsorted(sorted_cond, val_cond, side='left') / len(sorted_cond)
                    cdf_val_cond = 1 - cdf_val_cond_at_or_below  # Exceedance probability
                    cdf_val_uncond_at_or_below = np.searchsorted(sorted_uncond, val_uncond, side='left') / len(sorted_uncond)
                    cdf_val_uncond = 1 - cdf_val_uncond_at_or_below  # Exceedance probability
                    
                    # Conditional marker
                    fig.add_trace(go.Scatter(
                        x=[val_cond],
                        y=[cdf_val_cond],
                        mode='markers+text',
                        name=f"{label} (cond)",
                        marker=dict(size=12, color=color),
                        text=[label],
                        textposition="top center",
                        yaxis="y2",
                        hovertemplate=f'{label} (cond)<br>Value: {val_cond:.4g}<br>Probability of Exceedance: {cdf_val_cond:.1%}<extra></extra>',
                        showlegend=False
                    ))
                    # Unconditional marker
                    fig.add_trace(go.Scatter(
                        x=[val_uncond],
                        y=[cdf_val_uncond],
                        mode='markers',
                        name=f"{label} (uncond)",
                        marker=dict(size=8, color=color, opacity=0.5, symbol='circle-open'),
                        yaxis="y2",
                        hovertemplate=f'{label} (uncond)<br>Value: {val_uncond:.4g}<br>Probability of Exceedance: {cdf_val_uncond:.1%}<extra></extra>',
                        showlegend=False
                    ))
            else:
                # Add single CDF
                fig.add_trace(go.Scatter(
                    x=sorted_cond,
                    y=cdf_cond,
                    mode='lines',
                    name=f"CDF {var_name}",
                    line=dict(color=PALETTE[2], width=2),
                    opacity=0.95,
                    yaxis="y2",
                    hovertemplate="Value: %{x:.4g}<br>Probability of Exceedance: %{y:.1%}<extra></extra>"
                ))
                
                # Add markers
                for val, label, color in [(p10_cond, "P10", "red"), (p50_cond, "P50", "orange"), (mean_cond, "Mean", "green"), (p90_cond, "P90", "blue")]:
                    cdf_val_at_or_below = np.searchsorted(sorted_cond, val, side='left') / len(sorted_cond)
                    cdf_val = 1 - cdf_val_at_or_below  # Exceedance probability
                    fig.add_trace(go.Scatter(
                        x=[val],
                        y=[cdf_val],
                        mode='markers+text',
                        name=label,
                        marker=dict(size=12, color=color),
                        text=[label],
                        textposition="top center",
                        yaxis="y2",
                        hovertemplate=f'{label}<br>Value: {val:.4g}<br>Probability of Exceedance: {cdf_val:.1%}<extra></extra>',
                        showlegend=False
                    ))
            
            # Update layout
            fig.update_layout(
                xaxis_title=f"{var_name} ({sym})",
                yaxis_title="Probability Density",
                margin=dict(l=40, r=50, t=40, b=60),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                height=500,
                hovermode='x unified',
                barmode='overlay',
                bargap=0.02,
                yaxis2=dict(
                    overlaying="y",
                    side="right",
                    range=[0, 1],
                    showgrid=False,
                    title="Probability of Exceedance"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

# After plotting all variable distributions, show the input variables summary
show_exceedance = st.toggle("Use exceedance convention (P10=high)", value=True, help="If on: P10 is the 90th percentile.")

def p_lo_hi(x: np.ndarray):
    if show_exceedance:
        return np.percentile(x, 90), np.percentile(x, 10)
    else:
        return np.percentile(x, 10), np.percentile(x, 90)

has_occurrence = any(variables_config[sym].get("prob", 1.0) < 1.0 for sym in variables_config.keys())
order_cols = ["mean", "mode", "min", "p90", "p50", "p10", "max", "sd", "skew", "kurtosis"]
var_summary_rows = []
for sym in samples.keys():
    v_cond = samples[sym]
    v_uncond = unconditional_samples[sym]
    var_name = variables_config.get(sym, {}).get("name", sym)
    label_base = f"{var_name} ({sym})"
    # local summarize using same logic as results section
    def summarize_local(x: np.ndarray) -> Dict[str, float]:
        p10, p90 = p_lo_hi(x)
        return {
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)),
            "mode": float(approx_mode(x)),
            "skew": float(stats.skew(x, bias=False)) if SCIPY_AVAILABLE else float("nan"),
            "kurtosis": float(stats.kurtosis(x, fisher=False, bias=False)) if SCIPY_AVAILABLE else float("nan"),
            "p10": float(p10),
            "p50": float(np.percentile(x, 50)),
            "p90": float(p90),
            "p5": float(np.percentile(x, 95)),
            "p95": float(np.percentile(x, 5)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }
    if has_occurrence:
        var_summary_rows.append({"variable": f"{label_base} (conditional)", **summarize_local(v_cond)})
        var_summary_rows.append({"variable": f"{label_base} (unconditional)", **summarize_local(v_uncond)})
    else:
        var_summary_rows.append({"variable": label_base, **summarize_local(v_cond)})
if var_summary_rows:
    var_summary_df = pd.DataFrame(var_summary_rows)
    present = [c for c in order_cols if c in var_summary_df.columns]
    var_summary_df = var_summary_df[["variable"] + present]
    st.subheader("Input Variables — Distribution Summary")
    st.dataframe(var_summary_df, use_container_width=True)

st.markdown("---")

# --- Formula builder ---
st.subheader("Results – Custom Formulas")

# Build variable mapping for display
var_mapping = {}
for sym in var_symbols:
    if sym in variables_config:
        var_name = variables_config[sym]["name"]
        var_mapping[sym] = var_name

# Display variable mapping
with st.expander("Available Variables", expanded=False):
    if var_mapping:
        st.markdown("**Variable mapping for formulas:**")
        for sym, name in var_mapping.items():
            if name != sym:  # Only show if name is different from symbol
                st.code(f"{sym} = {name}", language="text")
    else:
        st.markdown("No variables defined yet.")

# Manage multiple result formulas
if "formulas" not in st.session_state:
    st.session_state.formulas = [
        {"name": "result", "expr": "a + b + c"}
    ]

cols = st.columns([2,1,1])
with cols[0]:
    # Show formula help in expandable box
    with st.expander("How to Build Formulas", expanded=False):
        st.markdown(HELP_SAFE_FUNCTIONS)
    
    # Preview derived aliases based on the current formulas order
    if st.session_state.formulas:
        with st.expander("Aliases available for later formulas (preview)", expanded=False):
            for i, f in enumerate(st.session_state.formulas, start=1):
                nm = (f["name"].strip() or "result")
                slug = _slugify(nm)
                st.code(f"Formula {i}: {nm}  →  f{i}  and  res_{slug}", language="text")
    
    # Build alias maps for display replacements (f# and res_<slug>)
    alias_index_to_name = {i: (f["name"].strip() or "result") for i, f in enumerate(st.session_state.formulas, start=1)}
    alias_slug_to_name = { _slugify(name): name for name in alias_index_to_name.values() }
    
    for i, f in enumerate(st.session_state.formulas):
        with st.expander(f"Formula {i+1}: {f['name']}", expanded=True):
            cols_f = st.columns([3,1])
            with cols_f[0]:
                f["name"] = st.text_input("Result name", value=f["name"], key=f"fname_{i}", help="Tip: Use subscripts with underscores or LaTeX. Examples: Oil_prod → Oil_{prod}, or write N_{gas}.")
                f["expr"] = st.text_input("Expression", value=f["expr"], key=f"fexpr_{i}", help="Use a,b,c or sidebar names (slugified), and earlier results: f1, f2, ... or res_<slug>.")
            with cols_f[1]:
                if st.button("Delete", key=f"del_{i}_{f['name']}"):
                    if "_to_delete" not in st.session_state:
                        st.session_state._to_delete = []
                    st.session_state._to_delete.append(i)
            
            # Replace symbols with names in the displayed formula
            display_expr = f["expr"]
            for sym, var_name in var_mapping.items():
                if sym in display_expr and var_name != sym:
                    display_expr = display_expr.replace(sym, var_name)
            
            # Replace references to earlier formulas (f# and res_<slug>) with result names for display
            def _repl_f(m):
                idx = int(m.group(1))
                name = alias_index_to_name.get(idx)
                if name:
                    return name.replace(' ', r'\ ')
                return m.group(0)
            display_expr = re.sub(r"\bf(\d+)\b", _repl_f, display_expr)
            
            def _repl_res(m):
                slug = m.group(1)
                name = alias_slug_to_name.get(slug)
                if name:
                    return name.replace(' ', r'\ ')
                return m.group(0)
            display_expr = re.sub(r"\bres_([a-z0-9_]+)\b", _repl_res, display_expr, flags=re.IGNORECASE)
            
            # Basic prettification for LaTeX rendering (sqrt, fractions, powers, multiplication)
            def to_latex(expr: str) -> str:
                s = expr
                # Powers: **n -> ^{n}
                s = re.sub(r"\*\*\s*([0-9]+)", r"^{\1}", s)
                # Multiplication: * -> \\cdot
                s = s.replace('*', '\\cdot ')
                
                # sqrt(... ) -> \\sqrt{...}
                def replace_sqrt(t: str) -> str:
                    out = []
                    i = 0
                    while i < len(t):
                        if t.startswith('sqrt(', i):
                            i0 = i + 5
                            depth = 1
                            j = i0
                            while j < len(t) and depth > 0:
                                if t[j] == '(': 
                                    depth += 1
                                elif t[j] == ')':
                                    depth -= 1
                                j += 1
                            inside = t[i0:j-1] if depth == 0 else t[i0:]
                            out.append('\\sqrt{' + inside + '}')
                            i = j
                        else:
                            out.append(t[i])
                            i += 1
                    return ''.join(out)
                s = replace_sqrt(s)
                
                # Simple top-level division: a/b -> \\frac{a}{b} (respect () and {} nesting)
                def top_level_frac(t: str) -> str:
                    depth_paren = 0
                    depth_brace = 0
                    for idx, ch in enumerate(t):
                        if ch == '(':
                            depth_paren += 1
                        elif ch == ')':
                            depth_paren = max(0, depth_paren - 1)
                        elif ch == '{':
                            depth_brace += 1
                        elif ch == '}':
                            depth_brace = max(0, depth_brace - 1)
                        elif ch == '/' and depth_paren == 0 and depth_brace == 0:
                            left = t[:idx].strip()
                            right = t[idx+1:].strip()
                            if left and right:
                                return f"\\frac{{{left}}}{{{right}}}"
                    return t
                s = top_level_frac(s)
                
                # Also convert a/b inside \\sqrt{ ... } to \\frac{a}{b}
                def convert_frac_inside_sqrt(t: str) -> str:
                    out = []
                    i = 0
                    while i < len(t):
                        if t.startswith('\\sqrt{', i):
                            # find matching closing '}' for this sqrt
                            j = i + 6  # position after '\\sqrt{'
                            depth = 1
                            start = j
                            while j < len(t) and depth > 0:
                                if t[j] == '{':
                                    depth += 1
                                elif t[j] == '}':
                                    depth -= 1
                                j += 1
                            inside = t[start:j-1] if depth == 0 else t[start:]
                            inside_conv = top_level_frac(inside)
                            out.append('\\sqrt{' + inside_conv + '}')
                            i = j
                        else:
                            out.append(t[i])
                            i += 1
                    return ''.join(out)
                s = convert_frac_inside_sqrt(s)
                return s
            
            latex_expr = to_latex(display_expr)
            
            # Left-hand side result name, support subscripts, no \\mathrm
            name_raw = f["name"]
            if any(ch in name_raw for ch in ['\\', '{', '}']):
                lhs_tex = name_raw  # assume user provided LaTeX
            elif '_' in name_raw:
                base, sub = name_raw.split('_', 1)
                lhs_tex = base.replace(' ', r'\ ') + '_{' + sub + '}'
            else:
                lhs_tex = name_raw.replace(' ', r'\ ')
            st.latex(f"{lhs_tex} = {latex_expr}")
            # Removed the raw expression caption below the equation
    
    # Apply deletions (if any)
    if getattr(st.session_state, "_to_delete", None):
        for idx in sorted(st.session_state._to_delete, reverse=True):
            if 0 <= idx < len(st.session_state.formulas):
                del st.session_state.formulas[idx]
        st.session_state._to_delete = []

if st.button("Add formula"):
    st.session_state.formulas.append({"name": f"result{len(st.session_state.formulas)+1}", "expr": "a + b"})

with cols[1]:
    if st.button("Clear formulas"):
        st.session_state.formulas = []
with cols[2]:
    compute_now = st.button("▶ Run simulation & evaluate", type="primary")

# --- Evaluate formulas ---
def evaluate_expression(expr: str, env: Dict[str, np.ndarray]) -> np.ndarray:
    # Build evaluation namespace: variables and allowed funcs
    local_dict = {**env, **SAFE_FUNCS, **SAFE_CONST}
    
    # Check for undefined variables
    # Extract potential variable names from expression
    # This regex finds words that are likely variables (not function names)
    potential_vars = set(re.findall(r'\b[a-z_][a-z0-9_]*\b', expr.lower()))
    allowed_names = set(list(local_dict.keys()) + list(SAFE_FUNCS.keys()))
    undefined_vars = potential_vars - set(local_dict.keys()) - {'and', 'or', 'not', 'true', 'false'}
    
    if undefined_vars:
        hint = "If you meant an earlier formula result, use f# (e.g., f1) or res_<slug> (e.g., res_net_profit)."
        raise ValueError(f"Undefined variables in expression '{expr}': {', '.join(undefined_vars)}. {hint}")
    
    # numexpr cannot see python functions; we'll use a fallback: vectorized eval via eval with restricted globals
    # First try numexpr for speed if no names conflict with funcs
    try:
        # Replace function names not supported by numexpr by numpy form using eval instead
        unsupported = any(fn in expr for fn in ["where", "clip", "round", "floor", "ceil", "sin", "cos", "tan", "log10"])
        if not unsupported:
            result = ne.evaluate(expr, local_dict=local_dict)
        else:
            # Fallback to eval
            result = eval(expr, {"__builtins__": {}}, local_dict)
        
        # Convert to numpy array and handle infinities/NaN
        result = np.asarray(result, dtype=float)
        
        # Replace infinities with NaN for consistency
        result = np.where(np.isinf(result), np.nan, result)
        
        return result
    except ZeroDivisionError:
        raise ValueError(f"Division by zero in expression '{expr}'. Check that variables don't contain zeros when used as divisors.")
    except NameError as e:
        raise ValueError(f"Variable not defined in expression '{expr}': {e}")
    except Exception as e:
        # Provide better error messages
        error_msg = str(e)
        if "division" in error_msg.lower() or "zero" in error_msg.lower():
            raise ValueError(f"Division by zero in expression '{expr}'. Check that variables don't contain zeros when used as divisors.")
        else:
            raise ValueError(f"Error evaluating expression '{expr}': {e}")

results: Dict[str, np.ndarray] = {}
results_unconditional: Dict[str, np.ndarray] = {}

if compute_now and st.session_state.formulas:
    with st.spinner("Computing results..."):
        errors = []

        # Work on mutable environments that we enrich as we go
        env_cond = {**samples}
        env_uncond = {**unconditional_samples}

        # Add aliases for variables by their given Name (slugified)
        for sym, spec in variables_config.items():
            var_name = (spec.get("name", "") or "").strip()
            if var_name:
                alias = _slugify(var_name)
                if alias and alias not in env_cond:
                    env_cond[alias] = samples.get(sym)
                if alias and alias not in env_uncond:
                    env_uncond[alias] = unconditional_samples.get(sym)

        for idx, f in enumerate(st.session_state.formulas, start=1):
            nm = (f["name"].strip() or "result")
            ex = f["expr"].strip()
            try:
                # Evaluate using the current enriched environments
                y_cond = evaluate_expression(ex, env_cond)
                y_uncond = evaluate_expression(ex, env_uncond)

                # Store results
                y_cond = np.asarray(y_cond, dtype=float)
                y_uncond = np.asarray(y_uncond, dtype=float)
                results[nm] = y_cond
                results_unconditional[nm] = y_uncond

                # Build aliases for subsequent formulas
                alias_index = f"f{idx}"
                alias_slug = f"res_{_slugify(nm)}"

                # Push into environments so later formulas can use them
                env_cond[alias_index] = y_cond
                env_cond[alias_slug] = y_cond
                env_uncond[alias_index] = y_uncond
                env_uncond[alias_slug] = y_uncond

            except ValueError as e:
                errors.append(f"Formula '{nm}': {e}")

        # Display errors if any
        if errors:
            st.error("**Errors occurred during computation:**")
            for err in errors:
                st.error(f"- {err}")

        # Only proceed if we have results
        if errors and not results:
            st.warning("No results to display. Please fix the errors above.")

    if results:
        # Show derived-variable mapping for clarity
        with st.expander("Derived variables from formulas", expanded=False):
            st.markdown("You can reference earlier results in later formulas:")
            for i, f in enumerate(st.session_state.formulas, start=1):
                nm = (f["name"].strip() or "result")
                slug = _slugify(nm)
                st.code(f"Formula {i}:  {nm}  →  f{i}  and  res_{slug}", language="text")

        # Check if any variable has occurrence probability
        has_occurrence = any(variables_config[sym].get("prob", 1.0) < 1.0 for sym in variables_config.keys())
        
        # Summary table - show conditional and unconditional
        def summarize(x: np.ndarray, prefix: str = "") -> Dict[str, float]:
            mode_val = float(approx_mode(x))
            # Calculate skewness and kurtosis if SciPy available
            skew_val = float(stats.skew(x, bias=False)) if SCIPY_AVAILABLE else float("nan")
            kurtosis_val = float(stats.kurtosis(x, fisher=False, bias=False)) if SCIPY_AVAILABLE else float("nan")
            p10, p90 = p_lo_hi(x)
            
            stats_dict = {
                "mean": float(np.mean(x)),
                "sd": float(np.std(x, ddof=1)),
                "mode": mode_val,
                "skew": skew_val,
                "kurtosis": kurtosis_val,
                "p10": float(p10),
                "p50": float(np.percentile(x, 50)),
                "p90": float(p90),
                "p5": float(np.percentile(x, 95)),
                "p95": float(np.percentile(x, 5)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            }
            
            if prefix:
                return {f"{k}_{prefix}": v for k, v in stats_dict.items()}
            return stats_dict
        
        # --- Show input variables summary table (before results table) ---
        order_cols = ["mean", "mode", "min", "p90", "p50", "p10", "max", "sd", "skew", "kurtosis"]
        var_summary_rows = []
        for sym in samples.keys():
            v_cond = samples[sym]
            v_uncond = unconditional_samples[sym]
            var_name = variables_config.get(sym, {}).get("name", sym)
            label_base = f"{var_name} ({sym})"
            if has_occurrence:
                var_summary_rows.append({"variable": f"{label_base} (conditional)", **summarize(v_cond)})
                var_summary_rows.append({"variable": f"{label_base} (unconditional)", **summarize(v_uncond)})
            else:
                var_summary_rows.append({"variable": label_base, **summarize(v_cond)})
        if var_summary_rows:
            var_summary_df = pd.DataFrame(var_summary_rows)
            # Reorder and drop p5/p95
            present = [c for c in order_cols if c in var_summary_df.columns]
            var_summary_df = var_summary_df[["variable"] + present]
            st.subheader("Input Variables — Distribution Summary")
            st.dataframe(var_summary_df, use_container_width=True)
        

        summary_rows = []
        for k, v_cond in results.items():
            v_uncond = results_unconditional[k]
            
            if has_occurrence:
                # Show two rows: one for conditional, one for unconditional
                row_cond = {"result": f"{k} (conditional)", **summarize(v_cond)}
                row_uncond = {"result": f"{k} (unconditional)", **summarize(v_uncond)}
                summary_rows.append(row_cond)
                summary_rows.append(row_uncond)
            else:
                # Only show one row
                row = {"result": k, **summarize(v_cond)}
                summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        # Reorder and drop p5/p95
        present_res = [c for c in order_cols if c in summary_df.columns]
        summary_df = summary_df[["result"] + present_res]
        st.dataframe(summary_df, use_container_width=True)

        # Helper function to create a single plot
        def create_plot(values, title, color=PALETTE[0], show_legend=True):
            """Create histogram + CDF plot for a single distribution"""
            fig = go.Figure()
            
            # Calculate statistics with inverted percentiles
            mean_val = float(np.mean(values))
            p10 = float(np.percentile(values, 90))  # High value
            p50 = float(np.percentile(values, 50))
            p90 = float(np.percentile(values, 10))  # Low value
            
            # Add histogram
            fig.add_histogram(
                x=values,
                nbinsx=100,
                name=title,
                marker_color=color,
                opacity=0.85,
                histnorm='probability density',
                hovertemplate="%{x:.4g}",
                showlegend=show_legend
            )
            
            # Update layout with secondary y-axis for CDF
            fig.update_layout(
                xaxis_title=f"{title}",
                yaxis_title="Probability Density",
                margin=dict(l=40, r=50, t=40, b=60),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                height=450,
                hovermode='x unified',
                barmode='overlay',
                bargap=0.02,
                yaxis2=dict(
                    overlaying="y",
                    side="right",
                    range=[0, 1],
                    showgrid=False,
                    title="Probability of Exceedance"
                )
            )
            
            # Calculate CDF
            sorted_v = np.sort(values)
            cdf_vals = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
            cdf_vals = 1 - cdf_vals  # Exceedance
            
            # Add CDF
            fig.add_trace(go.Scatter(
                x=sorted_v,
                y=cdf_vals,
                mode='lines',
                name=f"CDF {title}",
                line=dict(color=PALETTE[2], width=3),
                opacity=1.0,
                yaxis="y2",
                hovertemplate="Value: %{x:.4g}<br>Probability of Exceedance: %{y:.1%}<extra></extra>",
                showlegend=show_legend
            ))
            
            # Add markers at P10, P50, Mean, P90 on CDF
            def cdf_at(val):
                # Exceedance probability at value
                idx = np.searchsorted(sorted_v, val, side='left')
                return 1 - (idx / len(sorted_v))
            for val, label, mcolor in [(p10, "P10", "red"), (p50, "P50", "orange"), (mean_val, "Mean", "green"), (p90, "P90", "blue")]:
                fig.add_trace(go.Scatter(
                    x=[val],
                    y=[cdf_at(val)],
                    mode='markers+text',
                    name=label,
                    text=[label],
                    textposition="top center",
                    marker=dict(size=10, color=mcolor),
                    yaxis="y2",
                    hovertemplate="Value: %{x:.4g}<br>Exceedance: %{y:.1%}<extra></extra>",
                    showlegend=False
                ))
            return fig
        
        # Plots - Interactive Plotly with histogram and CDF overlaid
        for k in results.keys():
            v_cond = results[k]
            v_uncond = results_unconditional[k]
            
            # Check if any variable has occurrence probability
            has_occurrence = any(variables_config[sym].get("prob", 1.0) < 1.0 for sym in variables_config.keys())
            
            if has_occurrence:
                # Show three plots: conditional, unconditional, combined
                st.markdown(f"#### {k} - Distribution Plots")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Conditional plot
                    fig_cond = create_plot(v_cond, f"{k} (Conditional)", color=PALETTE[0])
                    st.plotly_chart(fig_cond, use_container_width=True)
                
                with col2:
                    # Unconditional plot
                    fig_uncond = create_plot(v_uncond, f"{k} (Unconditional)", color=PALETTE[1])
                    st.plotly_chart(fig_uncond, use_container_width=True)
                
                # Combined plot
                fig_combined = go.Figure()
                
                # Add both histograms
                fig_combined.add_histogram(
                    x=v_cond,
                    nbinsx=100,
                    name="Conditional Histogram",
                    marker_color=PALETTE[0],
                    opacity=0.6,
                    histnorm='probability density',
                    hovertemplate="%{x:.4g}"
                )
                fig_combined.add_histogram(
                    x=v_uncond,
                    nbinsx=100,
                    name="Unconditional Histogram",
                    marker_color=PALETTE[1],
                    opacity=0.4,
                    histnorm='probability density',
                    hovertemplate="%{x:.4g}"
                )
                
                # Calculate stats for both
                mean_cond = float(np.mean(v_cond))
                p10_cond = float(np.percentile(v_cond, 90))
                p50_cond = float(np.percentile(v_cond, 50))
                p90_cond = float(np.percentile(v_cond, 10))
                
                mean_uncond = float(np.mean(v_uncond))
                p10_uncond = float(np.percentile(v_uncond, 90))
                p50_uncond = float(np.percentile(v_uncond, 50))
                p90_uncond = float(np.percentile(v_uncond, 10))
                
                # Add conditional vertical lines
                for val, label, line_color in [(p10_cond, "P10", "red"), (p50_cond, "P50", "orange"), (mean_cond, "Mean", "green"), (p90_cond, "P90", "blue")]:
                    fig_combined.add_vline(
                        x=val,
                        line_width=2,
                        line_dash="dash",
                        line_color=line_color,
                        opacity=0.75
                    )
                
                # Add unconditional vertical lines (lighter)
                for val, label, line_color in [(p10_uncond, "P10", "red"), (p50_uncond, "P50", "orange"), (mean_uncond, "Mean", "green"), (p90_uncond, "P90", "blue")]:
                    fig_combined.add_vline(
                        x=val,
                        line_width=1.5,
                        line_dash="dot",
                        line_color=line_color,
                        opacity=0.4
                    )
                
                # Add both CDFs (probability of exceedance: P(Value > threshold))
                sorted_cond = np.sort(v_cond)
                sorted_uncond = np.sort(v_uncond)
                cdf_cond_at_or_below = np.arange(1, len(sorted_cond) + 1) / len(sorted_cond)
                cdf_cond = 1 - cdf_cond_at_or_below  # Invert to show probability of EXCEEDANCE
                cdf_uncond_at_or_below = np.arange(1, len(sorted_uncond) + 1) / len(sorted_uncond)
                cdf_uncond = 1 - cdf_uncond_at_or_below  # Invert to show probability of EXCEEDANCE
                
                fig_combined.add_trace(go.Scatter(
                    x=sorted_cond,
                    y=cdf_cond,
                    mode='lines',
                    name="Conditional CDF",
                    line=dict(color=PALETTE[1], width=3, dash='dash'),
                    opacity=1.0,
                    yaxis="y2",
                    hovertemplate="Value: %{x:.4g}<br>Probability of Exceedance: %{y:.1%}<extra></extra>"
                ))
                
                fig_combined.add_trace(go.Scatter(
                    x=sorted_uncond,
                    y=cdf_uncond,
                    mode='lines',
                    name="Unconditional CDF",
                    line=dict(color=PALETTE[2], width=3, dash='solid'),
                    opacity=1.0,
                    yaxis="y2",
                    hovertemplate="Value: %{x:.4g}<br>Probability of Exceedance: %{y:.1%}<extra></extra>"
                ))
                
                # Add markers for both
                for val_cond, val_uncond, label, line_color in [
                    (p10_cond, p10_uncond, "P10", "red"),
                    (p50_cond, p50_uncond, "P50", "orange"),
                    (mean_cond, mean_uncond, "Mean", "green"),
                    (p90_cond, p90_uncond, "P90", "blue")
                ]:
                    cdf_val_cond_at_or_below = np.searchsorted(sorted_cond, val_cond, side='left') / len(sorted_cond)
                    cdf_val_cond = 1 - cdf_val_cond_at_or_below  # Exceedance probability
                    cdf_val_uncond_at_or_below = np.searchsorted(sorted_uncond, val_uncond, side='left') / len(sorted_uncond)
                    cdf_val_uncond = 1 - cdf_val_uncond_at_or_below  # Exceedance probability
                    
                    # Conditional marker
                    fig_combined.add_trace(go.Scatter(
                        x=[val_cond],
                        y=[cdf_val_cond],
                        mode='markers+text',
                        name=f"{label} (cond)",
                        marker=dict(size=12, color=line_color),
                        text=[label],
                        textposition="top center",
                        yaxis="y2",
                        hovertemplate=f'{label} (cond)<br>Value: {val_cond:.4g}<br>Probability of Exceedance: {cdf_val_cond:.1%}<extra></extra>',
                        showlegend=False
                    ))
                    # Unconditional marker
                    fig_combined.add_trace(go.Scatter(
                        x=[val_uncond],
                        y=[cdf_val_uncond],
                        mode='markers',
                        name=f"{label} (uncond)",
                        marker=dict(size=8, color=line_color, opacity=0.5, symbol='circle-open'),
                        yaxis="y2",
                        hovertemplate=f'{label} (uncond)<br>Value: {val_uncond:.4g}<br>Probability of Exceedance: {cdf_val_uncond:.1%}<extra></extra>',
                        showlegend=False
                    ))
                
                fig_combined.update_layout(
                    xaxis_title=f"{k}",
                    yaxis_title="Probability Density",
                    margin=dict(l=40, r=50, t=40, b=60),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                    height=450,
                    hovermode='x unified',
                    barmode='overlay',
                    bargap=0.02,
                    yaxis2=dict(
                        overlaying="y",
                        side="right",
                        range=[0, 1],
                        showgrid=False,
                        title="Probability of Exceedance"
                    )
                )
                
                st.markdown("**Combined View:**")
                st.plotly_chart(fig_combined, use_container_width=True)
                
            else:
                # Only show one plot (no occurrence probability)
                fig = create_plot(v_cond, k, color=PALETTE[0])
                st.plotly_chart(fig, use_container_width=True)

        # Prepare data for download with trial numbers
        trial_df = pd.DataFrame({**samples, **results})
        trial_df.insert(0, 'Trial', range(1, len(trial_df) + 1))
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download samples as CSV",
                data=trial_df.to_csv(index=False).encode("utf-8"),
                file_name="probcalcmc_samples.csv",
                mime="text/csv"
            )
        with col2:
            if OPENPYXL_AVAILABLE:
                # Create Excel file with multiple sheets
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Sheet 1: Trial samples and results
                    trial_df.to_excel(writer, sheet_name='Trial Samples', index=False)

                    # Sheet 1b: Trial samples and results (Unconditional)
                    trial_df_uncond = pd.DataFrame({**unconditional_samples, **results_unconditional})
                    trial_df_uncond.insert(0, 'Trial', range(1, len(trial_df_uncond) + 1))
                    trial_df_uncond.to_excel(writer, sheet_name='Trial Samples (Uncond)', index=False)
                    
                    # Sheet 2: Summary statistics (with inverted percentiles)
                    stats_rows = []
                    for result_name in results.keys():
                        arr = results[result_name]
                        mode_res = approx_mode(arr)
                        stats_rows.append({
                            'Result': result_name,
                            'Mean': float(np.mean(arr)),
                            'SD': float(np.std(arr, ddof=1)),
                            'Skew': float(stats.skew(arr, bias=False)) if SCIPY_AVAILABLE else float('nan'),
                            'Kurtosis': float(stats.kurtosis(arr, fisher=False, bias=False)) if SCIPY_AVAILABLE else float('nan'),
                            'Mode': float(mode_res),
                            'P10': float(np.percentile(arr, 90)),
                            'P50': float(np.percentile(arr, 50)),
                            'P90': float(np.percentile(arr, 10)),
                            'P5': float(np.percentile(arr, 95)),
                            'P95': float(np.percentile(arr, 5)),
                            'Min': float(np.min(arr)),
                            'Max': float(np.max(arr))
                        })
                    stats_df = pd.DataFrame(stats_rows)
                    stats_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
                    
                    # Sheet 3: Input Statistics (variables)
                    var_export_rows = []
                    for sym in samples.keys():
                        v = samples[sym]
                        var_export_rows.append({
                            'Variable': sym,
                            'Name': variables_config.get(sym, {}).get('name', sym),
                            'Mean': float(np.mean(v)),
                            'SD': float(np.std(v, ddof=1)),
                            'Skew': float(stats.skew(v, bias=False)) if SCIPY_AVAILABLE else float('nan'),
                            'Kurtosis': float(stats.kurtosis(v, fisher=False, bias=False)) if SCIPY_AVAILABLE else float('nan'),
                            'Mode': float(approx_mode(v)),
                            'P10': float(np.percentile(v, 90)),
                            'P50': float(np.percentile(v, 50)),
                            'P90': float(np.percentile(v, 10)),
                            'P5': float(np.percentile(v, 95)),
                            'P95': float(np.percentile(v, 5)),
                            'Min': float(np.min(v)),
                            'Max': float(np.max(v))
                        })
                    var_stats_df = pd.DataFrame(var_export_rows)
                    var_stats_df.to_excel(writer, sheet_name='Input Statistics', index=False)
                    
                    # Sheet 4: Variable Map (symbol → name)
                    var_map_df = pd.DataFrame([
                        {'Symbol': sym, 'Name': variables_config.get(sym, {}).get('name', sym)}
                        for sym in variables_config.keys()
                    ])
                    var_map_df.to_excel(writer, sheet_name='Variable Map', index=False)

                    # Sheet 5: Derived Map (formulas → alias names)
                    derived_map_df = pd.DataFrame([
                        {
                            'Formula #': i,
                            'Result Name': (f["name"].strip() or "result"),
                            'Alias f#': f"f{i}",
                            'Alias res_<slug>': f"res_{_slugify(f['name'].strip() or 'result')}"
                        }
                        for i, f in enumerate(st.session_state.formulas, start=1)
                    ])
                    derived_map_df.to_excel(writer, sheet_name='Derived Map', index=False)

                    # Sheet 6: Formulae (traceability)
                    formula_rows = [
                        {"#": i, "Result Name": (f["name"].strip() or "result"), "Expression": f["expr"].strip()}
                        for i, f in enumerate(st.session_state.formulas, start=1)
                    ]
                    pd.DataFrame(formula_rows).to_excel(writer, sheet_name='Formulae', index=False)
                
                st.download_button(
                    "Download as Excel",
                    data=output.getvalue(),
                    file_name="probcalcmc_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.download_button(
                    "Download as Excel",
                    disabled=True,
                    help="Install openpyxl for Excel export: pip install openpyxl"
                )

# --- Show underlying variable samples (optional) ---
st.markdown("---")
with st.expander("Show variable samples (first 10 rows)"):
    display_df = pd.DataFrame(samples)
    display_df.insert(0, 'Trial', range(1, len(display_df) + 1))
    st.dataframe(display_df.head(10), use_container_width=True)

st.caption("Tip: Use the sidebar to add up to 256 variables. Set occurrence probabilities to model events that only sometimes happen. Build any formula you like with the allowed functions. Interactive plots show histogram and cumulative probability distribution.")
