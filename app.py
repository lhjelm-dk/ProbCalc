# Standard library imports
import io
import re
import unicodedata
import itertools
from typing import Dict, Any, Tuple, List, Optional

# Third-party imports
import streamlit as st
import numpy as np
import pandas as pd
import numexpr as ne
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib import cm
import plotly.graph_objects as go
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

# SimDec color palettes
SIMDEC_QUALITATIVE_CMAPS = [
    "SimDec native (default)",
    "SimDec sunset",
    "SimDec ocean",
    "SimDec pastel",
    "Seaborn deep",
    "tab20c",
    "tab20",
    "tab10",
    "Accent",
    "Dark2",
    "Paired",
    "Pastel1",
    "Pastel2",
    "Set1",
    "Set2",
    "Set3",
    "Qualitative10",
    "Spectral",
    "viridis",
    "plasma",
    "magma",
    "cividis",
]

CUSTOM_SIMDEC_PALETTES = {
    "SimDec sunset": [
        "#2C3E50",
        "#E74C3C",
        "#F39C12",
        "#27AE60",
        "#9B59B6",
        "#16A085",
    ],
    "SimDec ocean": [
        "#003f5c",
        "#2f4b7c",
        "#665191",
        "#a05195",
        "#d45087",
        "#f95d6a",
        "#ff7c43",
        "#ffa600",
    ],
    "SimDec pastel": [
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#d9d9d9",
        "#bc80bd",
    ],
}

# Optional SimDec imports
SIMDEC_AVAILABLE = False
try:
    from simdec import decompose, plot_bins, plot_box
    SIMDEC_AVAILABLE = True
except Exception:
    try:
        import simdec as _simdec

        def decompose(df: pd.DataFrame, *, inputs, output, bins=10, states=None):
            inputs_df = df[inputs]
            output_series = df[output]
            si_result = _simdec.sensitivity_indices(inputs_df, output_series)
            return _simdec.decomposition(
                inputs=inputs_df,
                output=output_series,
                sensitivity_indices=si_result.si,
                dec_limit=1,
                auto_ordering=True,
                states=states,
                statistic="mean",
            )

        def plot_bins(result):
            fig, ax = plt.subplots(figsize=(10, 5))
            palette_name = getattr(result, "palette_name", "SimDec native (default)")
            palette_colors = _get_simdec_palette(result.states, palette_name)
            if palette_colors is None:
                palette_colors = _simdec.palette(result.states)
            _simdec.visualization(
                bins=result.bins,
                palette=palette_colors,
                ax=ax,
                kind="histogram",
                n_bins="auto",
            )
            ax.set_xlabel("Result")
            ax.set_ylabel("Probability")
            ax.set_title("Scenario Histogram")
            if palette_name != "SimDec native (default)":
                _apply_palette_to_figure(fig, palette_colors)
            _style_simdec_plot(fig)
            return fig

        def plot_box(result):
            fig, ax = plt.subplots(figsize=(10, 5))
            palette_name = getattr(result, "palette_name", "SimDec native (default)")
            palette_colors = _get_simdec_palette(result.states, palette_name)
            if palette_colors is None:
                palette_colors = _simdec.palette(result.states)
            _simdec.visualization(
                bins=result.bins,
                palette=palette_colors,
                ax=ax,
                kind="boxplot",
            )
            ax.set_xlabel("Result")
            ax.set_title("Scenario Box Plot")
            if palette_name != "SimDec native (default)":
                _apply_palette_to_figure(fig, palette_colors)
            _style_simdec_plot(fig)
            return fig

        SIMDEC_AVAILABLE = True
    except Exception:
        SIMDEC_AVAILABLE = False

# SimDec helper functions
def _extract_matplotlib_figure(plot_obj) -> Optional[Figure]:
    """Best-effort extraction of a Matplotlib Figure from common plot objects."""
    if isinstance(plot_obj, Figure):
        return plot_obj
    if isinstance(plot_obj, Axes):
        return plot_obj.figure
    if isinstance(plot_obj, tuple):
        for item in plot_obj:
            fig = _extract_matplotlib_figure(item)
            if fig is not None:
                return fig
        return None
    figure_attr = getattr(plot_obj, "figure", None)
    if isinstance(figure_attr, Figure):
        return figure_attr
    return None

def _display_simdec_plot(plot_obj):
    """Render SimDec plot output (Matplotlib or Plotly) inside Streamlit."""
    if plot_obj is None:
        return
    if isinstance(plot_obj, go.Figure):
        st.plotly_chart(plot_obj, use_container_width=True)
        return

    fig = _extract_matplotlib_figure(plot_obj)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    # Fallback: display object as-is
    st.write(plot_obj)

def _get_simdec_palette(states, palette_name: str = "SimDec native (default)") -> Optional[List[List[float]]]:
    """Return RGBA colours for the requested SimDec states using matplotlib palettes."""
    if palette_name in (None, "SimDec native (default)"):
        return None

    if palette_name in CUSTOM_SIMDEC_PALETTES:
        colors = CUSTOM_SIMDEC_PALETTES[palette_name]
        colors_rgba = [mcolors.to_rgba(color) for color in colors]
        total_states = 1
        for entry in states:
            if isinstance(entry, list):
                total_states *= max(len(entry), 1)
            elif isinstance(entry, int):
                total_states *= max(entry, 1)
        total_states = max(total_states, 1)
        cycled = [colors_rgba[i % len(colors_rgba)] for i in range(total_states)]
        return [(*color[:3], float(color[3]) if len(color) == 4 else 1.0) for color in cycled]

    def _total_states() -> int:
        total = 1
        for entry in states:
            if isinstance(entry, list):
                total *= max(len(entry), 1)
            elif isinstance(entry, int):
                total *= max(entry, 1)
            else:
                total *= 1
        return max(total, 1)

    try:
        cmap = cm.get_cmap(palette_name)
        n_colors = _total_states()
        colors = [cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]
        return [(*color[:3], float(color[3]) if len(color) == 4 else 1.0) for color in colors]
    except Exception:
        return None

def _format_state_labels(result) -> List[List[str]]:
    """Format state labels based on label_strategy."""
    label_strategy = getattr(result, "label_strategy", "Low / Medium / High")
    var_names = getattr(result, "var_names", None)
    states = getattr(result, "states", None)
    bins = getattr(result, "bins", None)

    if not var_names or not states or bins is None:
        return []

    state_labels = []
    for var_idx, (var_name, state) in enumerate(zip(var_names, states)):
        labels = []
        n_states = state if isinstance(state, int) else len(state) if isinstance(state, list) else 1

        if label_strategy == "Low / Medium / High":
            if n_states == 2:
                labels = ["Low", "High"]
            elif n_states == 3:
                labels = ["Low", "Medium", "High"]
            else:
                labels = [f"State {i+1}" for i in range(n_states)]
        elif label_strategy == "Percentile ranges":
            for i in range(n_states):
                lower_pct = (i / n_states) * 100
                upper_pct = ((i + 1) / n_states) * 100
                labels.append(f"P{lower_pct:.0f}–P{upper_pct:.0f}")
        else:  # Numeric value ranges
            var_data = bins.get(var_name, [])
            if var_data and len(var_data) >= n_states:
                for i in range(n_states):
                    if i < len(var_data):
                        lower = float(np.min(var_data[i])) if hasattr(var_data[i], '__iter__') else float(var_data[i])
                        upper = float(np.max(var_data[i])) if hasattr(var_data[i], '__iter__') else float(var_data[i])
                        lower_pct = (i / n_states) * 100
                        upper_pct = ((i + 1) / n_states) * 100
                labels.append(f"P{lower_pct:.0f}–P{upper_pct:.0f} ({lower:.3g}–{upper:.3g})")

        state_labels.append(labels)

    return state_labels

def _get_simdec_scenario_labels(result) -> List[str]:
    """Compose scenario descriptions for legend entries."""
    labels: List[str] = []

    var_names = getattr(result, "var_names", None)
    states = getattr(result, "states", None)
    if not var_names or not states:
        return labels

    state_values = _format_state_labels(result)
    if not state_values:
        state_values = [
            ["low", "high"] if (isinstance(state, int) and state == 2) else
            ["low", "medium", "high"] if (isinstance(state, int) and state == 3) else
            [f"state {i+1}" for i in range(state if isinstance(state, int) else 1)]
            for state in states
        ]

    for combination in itertools.product(*state_values):
        parts = [f"{var}: {state}" for var, state in zip(var_names, combination)]
        labels.append(" | ".join(parts))

    return labels

def _add_simdec_legend(result, fig) -> None:
    """Append a descriptive legend beneath the supplied Matplotlib figure."""
    matplotlib_fig = _extract_matplotlib_figure(fig)
    if matplotlib_fig is None or not matplotlib_fig.axes:
        return

    palette = _get_simdec_palette(result.states, getattr(result, "palette_name", "SimDec native (default)"))
    if palette is None:
        try:
            from simdec import palette as _simdec_palette
            palette = np.asarray(_simdec_palette(result.states)).tolist()
        except Exception:
            palette = [(*color, 1.0) for color in sns.color_palette("deep", len(result.states))]
    if not palette:
        return

    scenario_labels = _get_simdec_scenario_labels(result)
    if not scenario_labels:
        scenario_labels = [f"Scenario {idx + 1}" for idx in range(len(palette))]

    ax = matplotlib_fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    existing_labels = set(labels)

    legend_handles = []
    for idx, color in enumerate(palette):
        label = scenario_labels[idx] if idx < len(scenario_labels) else f"Scenario {idx + 1}"
        if label in existing_labels:
            continue
        try:
            patch = Patch(facecolor=color, edgecolor="black", label=label)
        except Exception:
            continue
        legend_handles.append(patch)

    if not legend_handles:
        return

    handles.extend(legend_handles)
    labels.extend([h.get_label() for h in legend_handles])
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fontsize="small",
        frameon=False,
    )
    matplotlib_fig.subplots_adjust(bottom=0.3)

def _style_simdec_plot(fig) -> None:
    """Apply consistent typography and line widths to SimDec figures."""
    matplotlib_fig = _extract_matplotlib_figure(fig)
    if matplotlib_fig is None:
        return

    for ax in matplotlib_fig.axes:
        ax.tick_params(labelsize=9)
        if ax.title:
            ax.title.set_fontsize(11)
        ax.set_xlabel(ax.get_xlabel(), fontsize=10)
        ax.set_ylabel(ax.get_ylabel(), fontsize=10)

        for patch in ax.patches:
            try:
                patch.set_linewidth(0.6)
            except Exception:
                continue
        for line in ax.lines:
            try:
                current = line.get_linewidth() or 1.0
                line.set_linewidth(max(current * 0.8, 0.6))
            except Exception:
                continue
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

def _build_simdec_table(result, palette) -> Optional[pd.DataFrame]:
    try:
        from simdec import tableau as _tableau
        legend_df, styler = _tableau(
            var_names=result.var_names,
            statistic=result.statistic,
            states=result.states,
            bins=result.bins,
            palette=palette,
        )
    except Exception:
        return None
    return legend_df

def _apply_palette_to_figure(fig, palette: List[List[float]]) -> None:
    matplotlib_fig = _extract_matplotlib_figure(fig)
    if matplotlib_fig is None:
        return

    for ax in matplotlib_fig.axes:
        palette_cycle = itertools.cycle(palette)
        for patch in ax.patches:
            color = next(palette_cycle)
            try:
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
                patch.set_alpha(color[3] if len(color) == 4 else 1.0)
            except Exception:
                continue
        if hasattr(ax, "artists"):
            palette_cycle = itertools.cycle(palette)
            for artist in getattr(ax, "artists", []):
                color = next(palette_cycle)
                try:
                    artist.set_facecolor(color)
                    artist.set_edgecolor("black")
                    artist.set_alpha(color[3] if len(color) == 4 else 1.0)
                    artist.set_sizes([36])
                except Exception:
                    continue

st.set_page_config(page_title="ProbCalcMC", layout="wide")
st.markdown("# ProbCalcMC – Custom Monte Carlo simulation <span style='font-size:0.7em; color:#666; font-weight:400'>(v0.85)</span>", unsafe_allow_html=True)
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

# ============================================================================
# GLOBAL SESSION STATE INITIALIZATION - RUNS ONCE AT APP STARTUP
# ============================================================================
def initialize_session_state_globally():
    """
    ROCK-SOLID SESSION STATE MANAGEMENT:
    
    This function initializes ALL session_state keys ONCE at app startup.
    It NEVER overwrites existing values, ensuring user changes are preserved.
    
    Principles:
    1. Single Source of Truth: st.session_state is the ONLY source of truth
    2. Initialize Once: Use a flag to ensure defaults are set only once
    3. Never Overwrite: Only set keys that don't exist
    4. Safe Access: Always use .get() with defaults, never direct access
    5. Widget Pattern: All widgets use key=, which auto-updates session_state
    """
    # Flag to ensure initialization happens only once
    if "_session_state_initialized" not in st.session_state:
        # Default values from app.py (v0.8)
        # CRITICAL: Do NOT initialize parameter values here - let widgets handle them
        # This prevents overwriting user-entered values when page reloads
        DEFAULT_VALUES_V08 = {
            "name_a": "K",
            "name_b": "μ",
            "name_c": "ρ",
            "dtype_a": "Triangular",
            "dtype_b": "Normal",
            "dtype_c": "PERT",
            # REMOVED: Parameter defaults (a_low, a_mode, etc.) - let widgets initialize these
            # This prevents overwriting user-entered values
            "prob_a": 1.0,
            "prob_b": 1.0,
            "prob_c": 1.0,
        }
        
        # Initialize default values ONLY if they don't exist
        # CRITICAL: Only initialize name, dtype, and prob - NOT parameters
        for key, default_value in DEFAULT_VALUES_V08.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # Initialize other global settings
        if "num_vars" not in st.session_state:
            st.session_state.num_vars = 3
        if "n_samples" not in st.session_state:
            st.session_state.n_samples = 50_000
        if "seed" not in st.session_state:
            st.session_state.seed = 0
        if "variables_config" not in st.session_state:
            st.session_state.variables_config = {}
        if "var_symbols" not in st.session_state:
            st.session_state.var_symbols = []
        if "formulas" not in st.session_state:
            st.session_state.formulas = [
                {"name": "V_p", "expr": "sqrt((a + (4/3)*b) / c)"}
            ]
        if "use_correlation" not in st.session_state:
            st.session_state.use_correlation = False
        if "correlation_matrix" not in st.session_state:
            st.session_state.correlation_matrix = None
        if "correlation_values" not in st.session_state:
            st.session_state.correlation_values = {}
        if "correlation_var_symbols" not in st.session_state:
            st.session_state.correlation_var_symbols = None
        
        # Mark as initialized
        st.session_state._session_state_initialized = True

def ensure_variable_keys_initialized(var_symbols: List[str]):
    """
    Ensure all session_state keys exist for given variable symbols.
    This is called when the number of variables changes or when needed.
    NEVER overwrites existing values - only initializes missing keys.
    """
    DEFAULT_VALUES_V08 = {
        "name_a": "K", "name_b": "μ", "name_c": "ρ",
        "dtype_a": "Triangular", "dtype_b": "Normal", "dtype_c": "PERT",
        "a_low": 30e9, "a_mode": 35e9, "a_high": 40e9,
        "b_mean": 30e9, "b_sd": 3e9,
        "c_min": 2550.0, "c_most_likely": 2650.0, "c_max": 2800.0,
        "prob_a": 1.0, "prob_b": 1.0, "prob_c": 1.0,
    }
    
    for sym in var_symbols:
        name_key = f"name_{sym}"
        prob_key = f"prob_{sym}"
        dtype_key = f"dtype_{sym}"
        
        # Initialize name (only if missing)
        if name_key not in st.session_state:
            if sym in ['a', 'b', 'c']:
                st.session_state[name_key] = DEFAULT_VALUES_V08.get(f"name_{sym}", sym)
            else:
                st.session_state[name_key] = sym
        
        # Initialize probability (only if missing)
        # CRITICAL: Check backup key first, then variables_config, then default
        prob_backup_key = f"{prob_key}_backup"
        if prob_key not in st.session_state:
            # Priority 1: Check backup key (most reliable - saved directly)
            if prob_backup_key in st.session_state:
                st.session_state[prob_key] = float(st.session_state[prob_backup_key])
            else:
                # Priority 2: Check existing variables_config to preserve user's setting
                existing_config = st.session_state.get("variables_config", {})
                if sym in existing_config:
                    existing_prob = existing_config[sym].get("prob")
                    if existing_prob is not None:
                        st.session_state[prob_key] = float(existing_prob)
                        # Also save to backup for future
                        st.session_state[prob_backup_key] = float(existing_prob)
                    else:
                        st.session_state[prob_key] = DEFAULT_VALUES_V08.get(f"prob_{sym}", 1.0)
                else:
                    st.session_state[prob_key] = DEFAULT_VALUES_V08.get(f"prob_{sym}", 1.0)
        
        # Initialize distribution type (only if missing)
        if dtype_key not in st.session_state:
            if sym == 'a':
                st.session_state[dtype_key] = "Triangular"
            elif sym == 'b':
                st.session_state[dtype_key] = "Normal"
            elif sym == 'c':
                st.session_state[dtype_key] = "PERT"
            else:
                st.session_state[dtype_key] = list(DISTROS.keys())[0] if DISTROS else "Triangular"
        
        # Initialize previous dtype tracker (only if missing) - prevents false change detection
        prev_dtype_key = f"{dtype_key}_prev"
        if prev_dtype_key not in st.session_state:
            st.session_state[prev_dtype_key] = st.session_state.get(dtype_key, "Triangular")
        
        # Initialize distribution parameters (only if missing)
        # Get current dtype (may have been set above or by user)
        dtype = st.session_state.get(dtype_key)
        if dtype and dtype in DISTROS:
            params_spec = DISTROS[dtype]["params"]
            for label, default, ptype in params_spec:
                param_key = f"{sym}_{label}"
                if param_key not in st.session_state:
                    # Get default from app.py defaults if available
                    param_default_map = {
                        ("a", "low"): "a_low", ("a", "mode"): "a_mode", ("a", "high"): "a_high",
                        ("b", "mean"): "b_mean", ("b", "sd"): "b_sd",
                        ("c", "min"): "c_min", ("c", "most_likely"): "c_most_likely", ("c", "max"): "c_max",
                    }
                    app_default_key = param_default_map.get((sym, label))
                    if app_default_key and app_default_key in DEFAULT_VALUES_V08:
                        param_default = DEFAULT_VALUES_V08[app_default_key]
                    else:
                        param_default = default
                    
                    if ptype == float:
                        st.session_state[param_key] = float(param_default)
                    elif ptype == int:
                        st.session_state[param_key] = int(param_default)
                    else:
                        st.session_state[param_key] = str(param_default)

# Initialize session state globally (runs once)
initialize_session_state_globally()

# --- Sidebar: Navigation and Settings ---
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page",
        ["Start", "Input Variables", "Dependency", "Formula Definition", "Results"],
        key="nav_page"
    )
    
    st.markdown("---")
    st.header("Settings")
    n_samples = st.number_input("Number of Monte Carlo samples", 1000, 2_000_000, 
                                value=st.session_state.get("n_samples", 50_000), 
                                step=1000, help="How many draws to simulate for each variable.",
                                key="n_samples_input")
    st.session_state.n_samples = int(n_samples)
    seed = st.number_input("Random seed (optional)", value=st.session_state.get("seed", 0), 
                          min_value=0, step=1, key="seed_input")
    st.session_state.seed = int(seed)
    if seed:
        np.random.seed(int(seed))

    st.markdown("---")
    st.header("Percentile Convention")
    show_exceedance = st.toggle(
        "Use exceedance convention (P10=high)", 
        value=st.session_state.get("show_exceedance", True), 
        help="If ON: P10 = 90th percentile (high value), P90 = 10th percentile (low value). If OFF: P10 = 10th percentile (low value), P90 = 90th percentile (high value).",
        key="show_exceedance_toggle"
    )
    st.session_state.show_exceedance = show_exceedance

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

**Available Functions:** `abs`, `sqrt`, `exp`, `log`, `log10`, `log2`, `min`, `max`, `where`, `clip`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`, `round`, `sign`, `pow`

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
- **Inverse Trigonometry**: `arcsin(x)`, `arccos(x)`, `arctan(x)` - inverse trigonometric functions
- **Hyperbolic Functions**: `sinh(x)`, `cosh(x)`, `tanh(x)` - hyperbolic trigonometric functions
- **Power Function**: `pow(a, b)` - raise a to the power of b (alternative to `**`)
- **Sign Function**: `sign(x)` - returns -1, 0, or 1 based on sign of x
- **Logarithm Base 2**: `log2(x)` - logarithm base 2
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
Variable names from the sidebar are also available as aliases (slugified), e.g. Name "Velocity" → velocity.
"""

SAFE_FUNCS = {
    'abs': np.abs,
    'sqrt': np.sqrt,
    'exp': np.exp,
    'log': np.log,
    'log10': np.log10,
    'log2': np.log2,
    'min': np.minimum,
    'max': np.maximum,
    'where': np.where,
    'clip': np.clip,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'arcsin': np.arcsin,
    'arccos': np.arccos,
    'arctan': np.arctan,
    'sinh': np.sinh,
    'cosh': np.cosh,
    'tanh': np.tanh,
    'floor': np.floor,
    'ceil': np.ceil,
    'round': np.round,
    'sign': np.sign,
    'pow': np.power,
}

# --- Helpers to sample distributions ---

def sample_distribution(kind: str, params: Dict[str, Any], n: int, uniform: Optional[np.ndarray] = None) -> np.ndarray:
    if kind == "Constant":
        return np.full(n, float(params["value"]))
    elif kind == "Normal":
        sd = float(params["sd"])
        if sd <= 0:
            raise ValueError("Normal: sd must be > 0.")
        mean = float(params["mean"])
        if uniform is not None:
            # Use inverse CDF (PPF) to preserve correlation
            return stats.norm.ppf(uniform, loc=mean, scale=sd)
        return np.random.normal(mean, sd, size=n)
    elif kind == "Lognormal":
        sigma = float(params["sigma (log-sd)"])
        if sigma <= 0:
            raise ValueError("Lognormal: sigma (log-sd) must be > 0.")
        mu = float(params["mu (log-mean)"])
        if uniform is not None:
            # Use inverse CDF (PPF) to preserve correlation
            return np.exp(stats.norm.ppf(uniform, loc=mu, scale=sigma))
        return np.random.lognormal(mu, sigma, size=n)
    elif kind == "Uniform":
        low, high = float(params["low"]), float(params["high"])
        if not (low < high):
            raise ValueError("Uniform: low must be < high.")
        if uniform is not None:
            # Use inverse CDF (PPF) to preserve correlation
            return low + uniform * (high - low)
        return np.random.uniform(low, high, size=n)
    elif kind == "Triangular":
        low, mode, high = float(params["low"]), float(params["mode"]), float(params["high"]) 
        if not (low <= mode <= high):
            raise ValueError("Triangular: require low ≤ mode ≤ high.")
        if low == high:
            return np.full(n, low)
        if uniform is not None:
            # Use inverse CDF (PPF) to preserve correlation
            if SCIPY_AVAILABLE:
                return stats.triang.ppf(uniform, c=(mode-low)/(high-low), loc=low, scale=high-low)
            else:
                # Manual inverse CDF for triangular
                c = (mode - low) / (high - low)
                result = np.zeros_like(uniform)
                mask1 = uniform < c
                mask2 = uniform >= c
                result[mask1] = low + np.sqrt(uniform[mask1] * (high - low) * (mode - low))
                result[mask2] = high - np.sqrt((1 - uniform[mask2]) * (high - low) * (high - mode))
                return result
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
        if uniform is not None:
            # Use inverse CDF (PPF) to preserve correlation
            if SCIPY_AVAILABLE:
                x = stats.beta.ppf(uniform, alpha, beta)
            else:
                # Fallback: approximate using normal approximation (not ideal but works)
                mean_beta = alpha / (alpha + beta)
                var_beta = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
                x = np.clip(stats.norm.ppf(uniform, loc=mean_beta, scale=np.sqrt(var_beta)), 0, 1)
            return a + x*(b - a)
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
        if uniform is not None:
            # Use inverse CDF (PPF) to preserve correlation
            if SCIPY_AVAILABLE:
                return stats.expon.ppf(uniform, scale=1.0/rate)
            else:
                # Manual inverse CDF: -ln(1-u)/rate
                return -np.log(1 - uniform) / rate
        return np.random.exponential(1.0/rate, size=n)
    elif kind == "Gamma":
        shape, scale = float(params["shape"]), float(params["scale"]) 
        if shape <= 0 or scale <= 0:
            raise ValueError("Gamma: shape and scale must be > 0.")
        if uniform is not None:
            # Use inverse CDF (PPF) to preserve correlation
            if SCIPY_AVAILABLE:
                return stats.gamma.ppf(uniform, a=shape, scale=scale)
            else:
                # Fallback: approximate (not ideal but works)
                mean_gamma = shape * scale
                var_gamma = shape * scale**2
                return np.clip(stats.norm.ppf(uniform, loc=mean_gamma, scale=np.sqrt(var_gamma)), 0, None)
        return np.random.gamma(shape, scale, size=n)
    elif kind == "Beta":
        a, b = float(params["alpha"]), float(params["beta"]) 
        if a <= 0 or b <= 0:
            raise ValueError("Beta: alpha and beta must be > 0.")
        if uniform is not None:
            # Use inverse CDF (PPF) to preserve correlation
            if SCIPY_AVAILABLE:
                return stats.beta.ppf(uniform, a, b)
            else:
                # Fallback: approximate using normal approximation
                mean_beta = a / (a + b)
                var_beta = (a * b) / ((a + b)**2 * (a + b + 1))
                return np.clip(stats.norm.ppf(uniform, loc=mean_beta, scale=np.sqrt(var_beta)), 0, 1)
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

# --- Page Rendering Functions ---

def render_start_page():
    """Page 1: Start - Documentation and Help"""
    st.markdown("## Welcome to ProbCalcMC v0.85")
    st.markdown("**ProbCalcMC** is a Streamlit app for probabilistic modeling and Monte Carlo simulations.")
    
    with st.expander("How to Use ProbCalcMC", expanded=True):
        st.markdown("""
        **Workflow:**
        1. **Input Variables** - Define your input variables and their distributions
        2. **Dependency** - (Optional) Configure correlations between variables
        3. **Formula Definition** - Build formulas using your variables and run simulation
        4. **Results** - View distributions, sensitivity analysis, and export data
        
        **Key Features:**
        - Up to 256 variables with 20+ distribution types
        - Formula engine with chaining (reference earlier results)
        - Optional correlation/dependency modeling
        - Interactive plots and sensitivity analysis
        - CSV and Excel export
        """)
    
    with st.expander("How to Build Formulas", expanded=False):
        st.markdown(HELP_SAFE_FUNCTIONS)
    
    with st.expander("Statistical Terminology", expanded=False):
        st.markdown("""
        - **Mean (Arithmetic Mean)**: The average value of all samples
        - **SD (Standard Deviation)**: A measure of variability. Higher SD means more spread around the mean
        - **Mode**: Most frequent value
        - **Skew**: Measures asymmetry. Positive = right tail (high outliers), Negative = left tail (low outliers), Zero = symmetric
        - **P10 (High Value)**: 90% of outcomes are below this value - represents the high/optimistic scenario
        - **P50 (Median)**: 50% of values are below/above this - most likely value
        - **P90 (Low Value)**: Only 10% of outcomes are below this value - represents the low/conservative scenario
        - **Conditional Distribution**: When occurrence probability < 1, this shows the distribution including zero values (when the event didn't occur)
        - **Unconditional Distribution**: The underlying distribution without application of occurrence probability (as if the event always occurred)
        """)
        
        # Distribution descriptions
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
    
def render_input_variables_page():
    """Page 2: Input Variables - Variable Definition"""
    st.markdown("## Input Variables")
    st.markdown("Define your input variables and their probability distributions.")
    
    # DEBUG: Show all session_state keys at page start
    debug_mode = st.checkbox("🔍 Debug Mode (show session_state info)", value=False, key="debug_mode")
    if debug_mode:
        st.write("**DEBUG: All session_state keys at page start:**")
        all_keys = [k for k in st.session_state.keys() if not k.startswith("_")]
        param_keys = [k for k in all_keys if "_" in k and any(k.startswith(f"{sym}_") for sym in ['a', 'b', 'c'])]
        st.write(f"Total keys: {len(all_keys)}")
        st.write(f"Parameter keys: {param_keys}")
        for key in sorted(param_keys):
            st.write(f"- {key} = {st.session_state[key]} (type: {type(st.session_state[key]).__name__})")
        st.markdown("---")
    
    # Variable count selector
    max_vars = 256
    num_vars = st.number_input("How many variables?", 1, max_vars, 
                               value=st.session_state.get("num_vars", 3), step=1, key="num_vars_input")
    st.session_state.num_vars = int(num_vars)
    var_symbols = make_symbols(int(num_vars))
    
    # CRITICAL: Do NOT call ensure_variable_keys_initialized here
    # It was overwriting user-entered parameter values
    # Instead, let widgets handle parameter initialization directly
    # Only initialize basic keys (name, prob, dtype) in the loop below
    if debug_mode:
        st.write("**DEBUG: Parameter keys before widget creation:**")
        for sym in var_symbols[:3]:  # Show first 3 variables
            for label in ["low", "mode", "high", "mean", "sd", "min", "most_likely", "max"]:
                key = f"{sym}_{label}"
                if key in st.session_state:
                    st.write(f"- {key} = {st.session_state[key]}")
        st.markdown("---")
    
    # Update var_symbols in session_state
    st.session_state.var_symbols = var_symbols
    
    # Tip for users
    st.markdown("**💡 Tip:** Enter parameter values and click 'Update Distribution' for each variable to see the distribution plot and statistics.")
    
    # ========================================================================
    # VARIABLE DEFINITION: Build from session_state (always latest values)
    # ========================================================================
    # CRITICAL: We build variables_config from session_state values
    # session_state is updated automatically by widgets with key=
    # This ensures we always use the latest user-entered values
    variables_config: Dict[str, Dict[str, Any]] = {}
    
    # Variable definition UI - widgets update session_state automatically
    for sym in var_symbols:
        with st.expander(f"Variable {sym}", expanded=False):
            name_key = f"name_{sym}"
            prob_key = f"prob_{sym}"
            dtype_key = f"dtype_{sym}"
            
            # DEBUG: Show session_state status before initialization
            if debug_mode:
                st.write(f"**DEBUG: Before initialization for {sym}**")
                st.write(f"- {name_key} in session_state: {name_key in st.session_state}")
                st.write(f"- {prob_key} in session_state: {prob_key in st.session_state}")
                st.write(f"- {dtype_key} in session_state: {dtype_key in st.session_state}")
                if name_key in st.session_state:
                    st.write(f"- {name_key} value: {st.session_state[name_key]}")
                if prob_key in st.session_state:
                    st.write(f"- {prob_key} value: {st.session_state[prob_key]}")
                if dtype_key in st.session_state:
                    st.write(f"- {dtype_key} value: {st.session_state[dtype_key]}")
            
            # CRITICAL: Only initialize name, prob, and dtype if missing
            # Do NOT call ensure_variable_keys_initialized here - it might interfere
            # We'll handle parameter initialization separately below
            if name_key not in st.session_state:
                DEFAULT_NAMES = {"a": "K", "b": "μ", "c": "ρ"}
                st.session_state[name_key] = DEFAULT_NAMES.get(sym, sym)
            prob_backup_key = f"{prob_key}_backup"
            if prob_key not in st.session_state:
                # Priority 1: Check backup key (most reliable - saved directly)
                if prob_backup_key in st.session_state:
                    st.session_state[prob_key] = float(st.session_state[prob_backup_key])
                else:
                    # Priority 2: Try to get from existing variables_config first to preserve user's setting
                    existing_config = st.session_state.get("variables_config", {})
                    if sym in existing_config:
                        existing_prob = existing_config[sym].get("prob")
                        if existing_prob is not None:
                            st.session_state[prob_key] = float(existing_prob)
                            # Also save to backup for future
                            st.session_state[prob_backup_key] = float(existing_prob)
                        else:
                            st.session_state[prob_key] = 1.0
                    else:
                        st.session_state[prob_key] = 1.0
            if dtype_key not in st.session_state:
                if sym == 'a':
                    st.session_state[dtype_key] = "Triangular"
                elif sym == 'b':
                    st.session_state[dtype_key] = "Normal"
                elif sym == 'c':
                    st.session_state[dtype_key] = "PERT"
                else:
                    st.session_state[dtype_key] = list(DISTROS.keys())[0] if DISTROS else "Triangular"
            
            # Get current values from session_state
            # These values persist across page navigation because widgets with key= update session_state automatically
            name_default = st.session_state.get(name_key, sym)
            
            # Create widgets - they update session_state automatically
            name_widget_value = st.text_input(f"Name for `{sym}` (optional)", 
                                value=name_default,
                                key=name_key, 
                                help="Enter a descriptive name. Tip: You can copy-paste special characters like Greek letters (α, β, γ, θ, φ, λ, σ, μ, π, ρ) or subscripts/superscripts from online character maps.")
            
            # Get prob value - widget with key= automatically manages session_state[prob_key]
            # CRITICAL: Only initialize if it doesn't exist, and check backup key, variables_config, then default
            prob_backup_key = f"{prob_key}_backup"
            if prob_key not in st.session_state:
                # Priority 1: Check backup key (most reliable - saved directly)
                if prob_backup_key in st.session_state:
                    st.session_state[prob_key] = float(st.session_state[prob_backup_key])
                else:
                    # Priority 2: Check existing variables_config to preserve user's setting
                    existing_config = st.session_state.get("variables_config", {})
                    if sym in existing_config:
                        existing_prob = existing_config[sym].get("prob")
                        if existing_prob is not None:
                            st.session_state[prob_key] = float(existing_prob)
                            # Also save to backup for future
                            st.session_state[prob_backup_key] = float(existing_prob)
                        else:
                            st.session_state[prob_key] = 1.0
                    else:
                        st.session_state[prob_key] = 1.0
            else:
                # If prob_key exists, check if it was reset to default (1.0) but backup has a different value
                # This handles cases where something reset prob_key after the widget saved it
                if (st.session_state[prob_key] == 1.0 and 
                    prob_backup_key in st.session_state and 
                    abs(float(st.session_state[prob_backup_key]) - 1.0) > 0.001):
                    # prob_key was reset to default, but backup has the real value - restore it
                    # We can set this BEFORE the widget is created, so the widget will use the restored value
                    st.session_state[prob_key] = float(st.session_state[prob_backup_key])
                elif prob_backup_key not in st.session_state:
                    # If backup doesn't exist, create it from current value
                    st.session_state[prob_backup_key] = float(st.session_state[prob_key])
            
            # Probability slider - widget with key= automatically reads from and writes to st.session_state[prob_key]
            # The widget persists the value automatically, so it will be preserved across navigation
            prob_widget_value = st.slider(f"Occurrence probability for `{sym}`", 0.0, 1.0, 
                           value=float(st.session_state[prob_key]), 
                           step=0.01, key=prob_key,
                           help="Adjust probability (click 'Update Distribution' to see changes)")
            
            # Save to backup key to ensure value persists across navigation
            prob_backup_key = f"{prob_key}_backup"
            st.session_state[prob_backup_key] = float(prob_widget_value)
            
            # If prob_key was reset to default but backup has a different value, restore it
            if (st.session_state[prob_key] == 1.0 and 
                prob_backup_key in st.session_state and 
                abs(float(st.session_state[prob_backup_key]) - 1.0) > 0.001):
                # Backup is already saved above, will be checked on next render
                pass
            
            distro_list = list(DISTROS.keys())
            current_dtype = st.session_state.get(dtype_key, "Triangular")
            dtype_index = distro_list.index(current_dtype) if current_dtype in distro_list else 0
            
            # Track previous distribution type to detect changes
            prev_dtype_key = f"{dtype_key}_prev"
            prev_dtype = st.session_state.get(prev_dtype_key, current_dtype)
            
            # Distribution type selectbox - widget updates session_state automatically
            dtype_widget_value = st.selectbox(f"Distribution type for `{sym}`", distro_list, 
                                index=dtype_index,
                                key=dtype_key,
                                help="Select distribution type (click 'Update Distribution' to see changes)")
            
            # Get dtype from widget return value
            dtype = dtype_widget_value if dtype_widget_value in DISTROS else "Triangular"
            
            # CRITICAL: If distribution type changed, we need to handle parameter keys
            # BUT we should NOT delete old keys - they might be needed if user switches back
            # Instead, we'll just ensure new keys exist for the current distribution type
            if dtype != prev_dtype:
                # Update previous dtype tracker
                st.session_state[prev_dtype_key] = dtype
                # Note: We do NOT delete old parameter keys - they're preserved in case user switches back
            
            # CRITICAL: Do NOT initialize parameters here - let widgets handle it
            # This ensures that existing values in session_state are NEVER overwritten
            # Widgets with key= will read from session_state if the key exists
            # If the key doesn't exist, the widget will create it with the value= parameter
            params_spec = DISTROS[dtype]["params"]
            
            # ========================================================================
            # DISTRIBUTION PARAMETERS: Widgets update session_state automatically
            # ========================================================================
            # Widgets with key= automatically update session_state when user presses Enter
            # We read from session_state to get the latest values
            
            st.markdown("**Distribution Parameters:**")
            
            # Create parameter input widgets - they update session_state automatically
            # CRITICAL APPROACH: Let widgets handle initialization - they preserve existing values
            # Streamlit widgets with key= work as follows:
            # 1. If key exists in session_state → widget reads from session_state (value= is ignored)
            # 2. If key doesn't exist → widget uses value= and creates the key in session_state
            # This means existing user values are ALWAYS preserved
            param_values = {}
            for label, default, ptype in params_spec:
                param_key = f"{sym}_{label}"
                
                # Determine default value for this parameter (only used if key doesn't exist)
                DEFAULT_VALUES_V08 = {
                    "a_low": 30e9, "a_mode": 35e9, "a_high": 40e9,
                    "b_mean": 30e9, "b_sd": 3e9,
                    "c_min": 2550.0, "c_most_likely": 2650.0, "c_max": 2800.0,
                }
                param_default_map = {
                    ("a", "low"): "a_low", ("a", "mode"): "a_mode", ("a", "high"): "a_high",
                    ("b", "mean"): "b_mean", ("b", "sd"): "b_sd",
                    ("c", "min"): "c_min", ("c", "most_likely"): "c_most_likely", ("c", "max"): "c_max",
                }
                app_default_key = param_default_map.get((sym, label))
                if app_default_key and app_default_key in DEFAULT_VALUES_V08:
                    param_default = DEFAULT_VALUES_V08[app_default_key]
                else:
                    param_default = default
                
                # Convert default to correct type
                if ptype == float:
                    default_val = float(param_default)
                elif ptype == int:
                    default_val = int(param_default)
                else:
                    default_val = str(param_default)
                
                # CRITICAL: Only restore from backup if main key is MISSING
                # If main key exists, trust it - it's the widget's current value (most recent)
                # The backup is just a safety net for when the main key gets deleted/reset
                backup_key = f"{param_key}_backup"
                if backup_key in st.session_state and param_key not in st.session_state:
                    # Main key doesn't exist but backup does - restore from backup
                    backup_val = st.session_state[backup_key]
                    default_val = backup_val if ptype == float else (int(backup_val) if ptype == int else str(backup_val))
                    if debug_mode:
                        st.write(f"**RESTORING from backup: {param_key} = {backup_val} (main key was missing)**")
                    # CRITICAL: Restore the main key BEFORE widget creation so widget reads correct value
                    if ptype == float:
                        st.session_state[param_key] = float(backup_val)
                    elif ptype == int:
                        st.session_state[param_key] = int(backup_val)
                    else:
                        st.session_state[param_key] = str(backup_val)
                elif backup_key in st.session_state and param_key in st.session_state:
                    # Both exist - check if they match (for debug info only)
                    backup_val = st.session_state[backup_key]
                    main_val = st.session_state[param_key]
                    if debug_mode:
                        if ptype == float:
                            matches = abs(float(main_val) - float(backup_val)) < 1e-10
                        elif ptype == int:
                            matches = int(main_val) == int(backup_val)
                        else:
                            matches = str(main_val) == str(backup_val)
                        if not matches:
                            st.write(f"**NOTE: main={main_val}, backup={backup_val} - trusting main (widget value)**")
                
                # DEBUG: Show session_state status before widget creation
                if debug_mode:
                    st.write(f"**DEBUG: Parameter {label} ({sym})**")
                    st.write(f"- Key: {param_key}")
                    st.write(f"- Key exists in session_state: {param_key in st.session_state}")
                    if param_key in st.session_state:
                        st.write(f"- session_state[{param_key}] = {st.session_state[param_key]}")
                        st.write(f"- Type: {type(st.session_state[param_key])}")
                    st.write(f"- Backup key exists: {backup_key in st.session_state}")
                    if backup_key in st.session_state:
                        st.write(f"- Backup value: {st.session_state[backup_key]}")
                    st.write(f"- Calculated default value: {default_val}")
                    st.write(f"- Calculated default type: {type(default_val)}")
                
                # CRITICAL: Use session_state value if it exists, otherwise use default
                # This ensures widgets read the correct value and don't overwrite with defaults
                # Streamlit widgets with key= read from session_state, but if value= doesn't match,
                # the widget might use value= and overwrite session_state - so we make them match!
                if param_key in st.session_state:
                    # Key exists - use its value as widget_default (this matches what widget will read)
                    widget_default = st.session_state[param_key]
                    if ptype == float:
                        widget_default = float(widget_default)
                    elif ptype == int:
                        widget_default = int(widget_default)
                    else:
                        widget_default = str(widget_default)
                    if debug_mode:
                        st.write(f"- Using session_state value as widget_default: {widget_default}")
                else:
                    # Key doesn't exist - use calculated default
                    widget_default = default_val
                    if debug_mode:
                        st.write(f"- Using calculated default as widget_default: {widget_default}")
                
                # Create widget - it will:
                # - Read from session_state[param_key] if it exists (preserving user values)
                # - Use widget_default if key doesn't exist (first time only)
                # - Automatically update session_state[param_key] when user changes value
                if ptype == float:
                    widget_value = st.number_input(
                        label + f" ({sym})", 
                        value=widget_default,  # Use session_state value if it exists, otherwise default
                        key=param_key,
                        help="Enter value and press Enter, then click 'Update Distribution' to see changes"
                    )
                    
                    # Widget automatically updates session_state, save to backup for persistence
                    if debug_mode:
                        st.write(f"- Widget returned value: {widget_value}")
                        st.write(f"- session_state[{param_key}] after widget: {st.session_state.get(param_key, 'NOT SET')}")
                        if param_key in st.session_state:
                            st.write(f"- Values match: {abs(float(widget_value) - float(st.session_state[param_key])) < 1e-10}")
                        else:
                            st.write(f"⚠️ WARNING: Key {param_key} NOT in session_state after widget creation!")
                    
                    param_values[label] = float(widget_value)
                    # Update backup key to ensure value persists across navigation
                    backup_key = f"{param_key}_backup"
                    st.session_state[backup_key] = float(widget_value)
                    if debug_mode:
                        st.write(f"- Backup updated: {backup_key} = {float(widget_value)}")
                elif ptype == int:
                    # Use session_state value if it exists, otherwise use default
                    if param_key in st.session_state:
                        widget_default = int(st.session_state[param_key])
                    else:
                        widget_default = default_val
                    widget_value = st.number_input(
                        label + f" ({sym})", 
                        value=widget_default,  # Use session_state value if it exists, otherwise default
                        step=1, 
                        key=param_key,
                        help="Enter value and press Enter, then click 'Update Distribution' to see changes"
                    )
                    param_values[label] = int(widget_value)
                    # Update backup key to ensure value persists across navigation
                    backup_key = f"{param_key}_backup"
                    st.session_state[backup_key] = int(widget_value)
                else:
                    # Use session_state value if it exists, otherwise use default
                    if param_key in st.session_state:
                        widget_default = str(st.session_state[param_key])
                    else:
                        widget_default = default_val
                    widget_value = st.text_input(
                        label + f" ({sym})", 
                        value=widget_default,  # Use session_state value if it exists, otherwise default
                        key=param_key,
                        help="Enter value and press Enter, then click 'Update Distribution' to see changes"
                    )
                    param_values[label] = str(widget_value)
                    # Update backup key to ensure value persists across navigation
                    backup_key = f"{param_key}_backup"
                    st.session_state[backup_key] = str(widget_value)
            
            # Add Update Distribution button for this variable
            update_dist_key = f"update_dist_{sym}"
            update_this_dist = st.button(
                "🔄 Update Distribution",
                key=update_dist_key,
                help="Click to update the distribution plot and statistics with current parameter values",
                use_container_width=True
            )
            
            # Track if this variable should show updated plot
            should_update_plot = update_this_dist or st.session_state.get(f"force_update_{sym}", False)
            if update_this_dist:
                st.session_state[f"force_update_{sym}"] = True
            if should_update_plot:
                st.session_state[f"force_update_{sym}"] = False
            
            # Use widget return values
            name = name_widget_value.strip() if name_widget_value else ""
            if not name:
                DEFAULT_NAMES = {"a": "K", "b": "μ", "c": "ρ"}
                name = DEFAULT_NAMES.get(sym, sym)
            
            prob = float(prob_widget_value)
            # Note: prob_widget_value is already saved to st.session_state[prob_key] automatically by the widget
            
            # Save prob to backup key to ensure it persists across navigation
            prob_backup_key = f"{prob_key}_backup"
            st.session_state[prob_backup_key] = prob
            
            # Build variables_config
            variables_config[sym] = {"name": name, "prob": prob, "type": dtype, "params": param_values}
            
            # Save parameter values to session_state as backup for persistence
            for label, value in param_values.items():
                param_key = f"{sym}_{label}"
                backup_key = f"{param_key}_backup"
                st.session_state[backup_key] = value
            
            # ========================================================================
            # DISTRIBUTION PLOT - Only updates when button is clicked or on first load
            # ========================================================================
            # Show/update the plot when user clicks "Update Distribution" or on first load
            
            # Check if we should show the plot (always show on first load, or when button clicked)
            show_plot_key = f"show_plot_{sym}"
            if show_plot_key not in st.session_state:
                st.session_state[show_plot_key] = True  # Show on first load
            
            # Update plot if button was clicked or if it's the first time
            if should_update_plot:
                st.session_state[show_plot_key] = True
            
            # Only show plot if flag is set
            if st.session_state[show_plot_key]:
                n_samples = st.session_state.get("n_samples", 50_000)
                seed = st.session_state.get("seed", 0)
                
                # Build plot config from CURRENT widget return values
                # CRITICAL: param_values, name, prob, dtype are all from widget return values above
                # These reflect the CURRENT values in the widgets (which update session_state automatically)
                try:
                    plot_config = {
                        sym: {
                            "name": name,  # From widget return value
                            "prob": prob,  # From widget return value
                            "type": dtype,  # From widget return value
                            "params": param_values.copy()  # From widget return values - CURRENT values
                        }
                    }
                    
                    # CRITICAL: Recalculate plot with CURRENT widget return values
                    # This uses the latest values the user entered (widgets update session_state automatically)
                    temp_samples, temp_unconditional_samples = _simulate_variables_uncached(plot_config, int(n_samples), int(seed))
                    
                    if sym in temp_samples and len(temp_samples[sym]) > 0:
                        values_cond = temp_samples[sym]
                        values_uncond = temp_unconditional_samples[sym]
                        has_occurrence = variables_config[sym].get("prob", 1.0) < 1.0
                        
                        # Get exceedance convention from sidebar toggle
                        show_exceedance = st.session_state.get("show_exceedance", True)
                        
                        # Calculate statistics - respect exceedance convention
                        # If exceedance convention: P10 = 90th percentile (high), P90 = 10th percentile (low)
                        # If standard convention: P10 = 10th percentile (low), P90 = 90th percentile (high)
                        mean_cond = float(np.mean(values_cond))
                        if show_exceedance:
                            p10_cond = float(np.percentile(values_cond, 90))  # High value (exceedance convention)
                            p90_cond = float(np.percentile(values_cond, 10))  # Low value (exceedance convention)
                        else:
                            p10_cond = float(np.percentile(values_cond, 10))  # Low value (standard convention)
                            p90_cond = float(np.percentile(values_cond, 90))  # High value (standard convention)
                        p50_cond = float(np.percentile(values_cond, 50))  # Median (always 50th percentile)
                        mode_cond = float(approx_mode(values_cond))
                        min_cond = float(np.min(values_cond))
                        max_cond = float(np.max(values_cond))
                        sd_cond = float(np.std(values_cond, ddof=1))
                        skew_cond = float(stats.skew(values_cond, bias=False)) if SCIPY_AVAILABLE else float("nan")
                        kurtosis_cond = float(stats.kurtosis(values_cond, fisher=False, bias=False)) if SCIPY_AVAILABLE else float("nan")
                        
                        mean_uncond = float(np.mean(values_uncond))
                        if show_exceedance:
                            p10_uncond = float(np.percentile(values_uncond, 90))  # High value (exceedance convention)
                            p90_uncond = float(np.percentile(values_uncond, 10))  # Low value (exceedance convention)
                        else:
                            p10_uncond = float(np.percentile(values_uncond, 10))  # Low value (standard convention)
                            p90_uncond = float(np.percentile(values_uncond, 90))  # High value (standard convention)
                        p50_uncond = float(np.percentile(values_uncond, 50))  # Median (always 50th percentile)
                        mode_uncond = float(approx_mode(values_uncond))
                        min_uncond = float(np.min(values_uncond))
                        max_uncond = float(np.max(values_uncond))
                        sd_uncond = float(np.std(values_uncond, ddof=1))
                        skew_uncond = float(stats.skew(values_uncond, bias=False)) if SCIPY_AVAILABLE else float("nan")
                        kurtosis_uncond = float(stats.kurtosis(values_uncond, fisher=False, bias=False)) if SCIPY_AVAILABLE else float("nan")
                        
                        # Display statistics as a table (similar to summary table)
                        st.markdown("---")
                        st.markdown("**Distribution Statistics:**")
                        
                        if has_occurrence:
                            # Show both conditional and unconditional in table format
                            stats_rows = [
                                {
                                    "variable": f"{name} ({sym}) - Conditional",
                                    "mean": mean_cond,
                                    "mode": mode_cond,
                                    "min": min_cond,
                                    "p90": p90_cond,
                                    "p50": p50_cond,
                                    "p10": p10_cond,
                                    "max": max_cond,
                                    "sd": sd_cond,
                                    "skew": skew_cond,
                                    "kurtosis": kurtosis_cond,
                                },
                                {
                                    "variable": f"{name} ({sym}) - Unconditional",
                                    "mean": mean_uncond,
                                    "mode": mode_uncond,
                                    "min": min_uncond,
                                    "p90": p90_uncond,
                                    "p50": p50_uncond,
                                    "p10": p10_uncond,
                                    "max": max_uncond,
                                    "sd": sd_uncond,
                                    "skew": skew_uncond,
                                    "kurtosis": kurtosis_uncond,
                                }
                            ]
                        else:
                            # Show only one row
                            stats_rows = [
                                {
                                    "variable": f"{name} ({sym})",
                                    "mean": mean_cond,
                                    "mode": mode_cond,
                                    "min": min_cond,
                                    "p90": p90_cond,
                                    "p50": p50_cond,
                                    "p10": p10_cond,
                                    "max": max_cond,
                                    "sd": sd_cond,
                                    "skew": skew_cond,
                                    "kurtosis": kurtosis_cond,
                                }
                            ]
                        
                        # Create DataFrame and display as table
                        stats_df = pd.DataFrame(stats_rows)
                        order_cols = ["mean", "mode", "min", "p90", "p50", "p10", "max", "sd", "skew", "kurtosis"]
                        present_cols = [c for c in order_cols if c in stats_df.columns]
                        stats_df = stats_df[["variable"] + present_cols]
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
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
                                name=f"{name} Histogram",
                                marker_color=PALETTE[0],
                                opacity=0.85,
                                histnorm='probability density',
                                hovertemplate="%{x:.4g}"
                            )
                        
                        # Add vertical lines for P10, P50, Mean, P90 (conditional)
                        # Color based on exceedance convention: if ON, P10=high (red), P90=low (blue); if OFF, P10=low (blue), P90=high (red)
                        p10_color = "red" if show_exceedance else "blue"  # High value if exceedance, low value if standard
                        p90_color = "blue" if show_exceedance else "red"  # Low value if exceedance, high value if standard
                        for val, label, color in [(p10_cond, "P10", p10_color), (p50_cond, "P50", "orange"), (mean_cond, "Mean", "green"), (p90_cond, "P90", p90_color)]:
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
                            # Color based on exceedance convention: if ON, P10=high (red), P90=low (blue); if OFF, P10=low (blue), P90=high (red)
                            p10_color = "red" if show_exceedance else "blue"  # High value if exceedance, low value if standard
                            p90_color = "blue" if show_exceedance else "red"  # Low value if exceedance, high value if standard
                            for val, label, color in [(p10_uncond, "P10", p10_color), (p50_uncond, "P50", "orange"), (mean_uncond, "Mean", "green"), (p90_uncond, "P90", p90_color)]:
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
                            # Color based on exceedance convention: if ON, P10=high (red), P90=low (blue); if OFF, P10=low (blue), P90=high (red)
                            p10_color = "red" if show_exceedance else "blue"  # High value if exceedance, low value if standard
                            p90_color = "blue" if show_exceedance else "red"  # Low value if exceedance, high value if standard
                            for val_cond, val_uncond, label, color in [
                                (p10_cond, p10_uncond, "P10", p10_color),
                                (p50_cond, p50_uncond, "P50", "orange"),
                                (mean_cond, mean_uncond, "Mean", "green"),
                                (p90_cond, p90_uncond, "P90", p90_color)
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
                                name=f"CDF {name}",
                                line=dict(color=PALETTE[2], width=2),
                                opacity=0.95,
                                yaxis="y2",
                                hovertemplate="Value: %{x:.4g}<br>Probability of Exceedance: %{y:.1%}<extra></extra>"
                            ))
                            
                            # Add markers
                            # Color based on exceedance convention: if ON, P10=high (red), P90=low (blue); if OFF, P10=low (blue), P90=high (red)
                            p10_color = "red" if show_exceedance else "blue"  # High value if exceedance, low value if standard
                            p90_color = "blue" if show_exceedance else "red"  # Low value if exceedance, high value if standard
                            for val, label, color in [(p10_cond, "P10", p10_color), (p50_cond, "P50", "orange"), (mean_cond, "Mean", "green"), (p90_cond, "P90", p90_color)]:
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
                            xaxis_title=f"{name} ({sym})",
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
                            ),
                            title=dict(
                                text=f"Distribution of {name} ({sym})",
                                font=dict(size=14),
                                x=0.5,
                                xanchor='center'
                            )
                        )
                        
                        # Use a stable key that only changes when we explicitly update
                        plot_key = f"plot_{sym}"
                        st.plotly_chart(fig, use_container_width=True, key=plot_key)
                    else:
                        st.warning(f"No samples generated for {sym}. Check parameter values.")
                except Exception as e:
                    # Show error if variable can't be simulated
                    st.error(f"Error calculating distribution for {sym}: {e}")
            else:
                # Show message if plot is not being displayed
                st.info("👆 Enter parameter values and click 'Update Distribution' to see the plot and statistics.")
    
    # CRITICAL: variables_config is built from session_state values (CURRENT values after widgets updated them)
    # Store it in session state IMMEDIATELY - this is the source of truth for simulation
    # This happens on EVERY page render, so it always has the latest values from session_state
    # session_state values are the source of truth - they reflect what widgets updated when user pressed Enter
    st.session_state.variables_config = variables_config.copy()  # Store a copy to prevent mutation issues
    st.session_state.var_symbols = var_symbols
    # Add timestamp to track when config was last updated
    import time
    st.session_state.variables_config_last_updated = time.time()
    
    # Show confirmation that values are stored
    if variables_config:
        st.success(f"✅ {len(variables_config)} variable(s) configured. Values are saved and will be used in simulation.")
    
    # Run full simulation for all variables (for summary table and use in other pages)
    if variables_config:
        st.markdown("---")
        st.subheader("Input Variables — Distribution Summary")
        
        # Show input parameters table
        st.markdown("**Input Parameters Used in Calculation:**")
        input_params_rows = []
        for sym in var_symbols:
            if sym in variables_config:
                var_spec = variables_config[sym]
                var_name = var_spec.get("name", sym)
                dtype = var_spec.get("type", "Unknown")
                params = var_spec.get("params", {})
                prob = var_spec.get("prob", 1.0)
                
                # Build parameter string
                param_strs = []
                for label, value in params.items():
                    if isinstance(value, float):
                        param_strs.append(f"{label}={value:,.3g}")
                    elif isinstance(value, int):
                        param_strs.append(f"{label}={value}")
                    else:
                        param_strs.append(f"{label}={value}")
                
                param_display = ", ".join(param_strs)
                if prob < 1.0:
                    param_display += f", prob={prob:.3f}"
                
                input_params_rows.append({
                    "Variable": f"{var_name} ({sym})",
                    "Distribution": dtype,
                    "Parameters": param_display
                })
        
        if input_params_rows:
            input_params_df = pd.DataFrame(input_params_rows)
            st.dataframe(input_params_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Get n_samples and seed from sidebar (stored in session state)
        n_samples = st.session_state.get("n_samples", 50_000)
        seed = st.session_state.get("seed", 0)
        
        # Run simulation for all variables
        try:
            samples, unconditional_samples = simulate_variables(variables_config, int(n_samples), int(seed))
            
            # Store samples in session state for use in other pages
            st.session_state.samples = samples
            st.session_state.unconditional_samples = unconditional_samples
            
            # Show summary table
            # Get exceedance convention from sidebar toggle (already set in sidebar)
            show_exceedance = st.session_state.get("show_exceedance", True)
            
            def p_lo_hi(x: np.ndarray):
                if show_exceedance:
                    return np.percentile(x, 90), np.percentile(x, 10)
                else:
                    return np.percentile(x, 10), np.percentile(x, 90)
            
            # Summary table of all input variables
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
                st.dataframe(var_summary_df, use_container_width=True)
        except ValueError as e:
            st.error(f"Error in variable definition: {e}")
            st.session_state.samples = {}
            st.session_state.unconditional_samples = {}

def _nearest_correlation_matrix(A, tol=1e-12, max_iter=100):
    """
    Higham (2002) nearest correlation matrix projection.
    Projects a matrix to the nearest positive semi-definite correlation matrix.
    """
    X = np.array(A, dtype=float, copy=True)
    X = 0.5 * (X + X.T)  # Make symmetric
    np.fill_diagonal(X, 1.0)  # Ensure diagonal is 1
    Y = X.copy()
    Delta_S = np.zeros_like(X)
    
    for _ in range(max_iter):
        R = Y - Delta_S
        eigval, eigvec = np.linalg.eigh(R)
        eigval = np.clip(eigval, 0.0, None)  # Remove negative eigenvalues
        X = (eigvec * eigval) @ eigvec.T
        X = 0.5 * (X + X.T)  # Ensure symmetry
        np.fill_diagonal(X, 1.0)
        Delta_S = X - R
        Y = X.copy()
        if np.linalg.norm(X - R, ord='fro') <= tol:
            break
    
    X = np.clip(X, -0.999, 0.999)
    np.fill_diagonal(X, 1.0)
    return X

def fix_correlation_matrix(corr_matrix: np.ndarray) -> np.ndarray:
    """Project a matrix to the nearest valid correlation matrix."""
    fixed = _nearest_correlation_matrix(corr_matrix)
    fixed = np.clip(fixed, -0.999, 0.999)
    np.fill_diagonal(fixed, 1.0)
    return fixed

def validate_dependency_matrix(dep_matrix: np.ndarray, var_names: List[str]) -> Tuple[bool, str]:
    """Validate dependency matrix for reasonable values."""
    dep_matrix = np.asarray(dep_matrix, dtype=float)
    n = len(var_names)
    
    if dep_matrix.shape != (n, n):
        return False, "Dependency matrix shape does not match variable count"
    
    if not np.allclose(dep_matrix, dep_matrix.T, atol=1e-8):
        return False, "Dependency matrix must be symmetric"
    
    if not np.allclose(np.diag(dep_matrix), 1.0, atol=1e-8):
        return False, "Diagonal entries must equal 1.0"
    
    off_diag = dep_matrix[np.triu_indices_from(dep_matrix, k=1)]
    if np.any(off_diag < -0.999) or np.any(off_diag > 0.999):
        return False, "Off-diagonal elements must be within [-0.999, 0.999]"
    
    try:
        _nearest_correlation_matrix(dep_matrix)
    except np.linalg.LinAlgError as exc:
        return False, f"Dependency matrix could not be stabilised: {exc}"
    
    return True, ""

def sample_correlated_variables(variables_config: Dict[str, Dict[str, Any]], 
                                corr_matrix: np.ndarray, 
                                var_symbols: List[str],
                                n: int, 
                                seed: int) -> Dict[str, np.ndarray]:
    """
    Generate correlated samples using Cholesky decomposition.
    Adapted from SCOPE-HC for ProbCalc's distribution types.
    """
    if not var_symbols or n <= 0:
        return {}
    
    # Fix correlation matrix to ensure it's positive semi-definite
    corr = fix_correlation_matrix(corr_matrix)
    
    # CRITICAL: Verify matrix is positive definite before Cholesky
    # Check eigenvalues - all must be > 0 for positive definite
    eigvals = np.linalg.eigvals(corr)
    min_eigval = np.min(eigvals)
    
    # If matrix is not positive definite, apply additional fixing
    if min_eigval <= 1e-10:
        # Add small value to diagonal to ensure positive definiteness
        # This is a common technique when Higham projection isn't enough
        corr = corr + np.eye(len(corr)) * max(1e-10, abs(min_eigval) + 1e-8)
        # Renormalize to ensure diagonal is 1
        diag_sqrt = np.sqrt(np.diag(corr))
        corr = corr / (diag_sqrt[:, None] * diag_sqrt[None, :])
        # Ensure diagonal is exactly 1
        np.fill_diagonal(corr, 1.0)
        # Clip to valid range
        corr = np.clip(corr, -0.999, 0.999)
        np.fill_diagonal(corr, 1.0)
        # Apply Higham projection again to ensure it's valid
        corr = fix_correlation_matrix(corr)
    
    # Handle extreme correlations (near ±1)
    extreme_pairs: List[Tuple[int, int, float]] = []
    for i in range(len(var_symbols)):
        for j in range(i + 1, len(var_symbols)):
            rho = corr[i, j]
            if abs(rho) >= 0.999:
                sign = float(np.sign(rho) or 1.0)
                extreme_pairs.append((i, j, sign))
                corr[i, j] = corr[j, i] = sign * 0.98
    
    # Final check: ensure matrix is still positive definite after adjustments
    eigvals_final = np.linalg.eigvals(corr)
    if np.min(eigvals_final) <= 1e-10:
        # Last resort: use eigendecomposition to create a valid correlation matrix
        eigvals_fixed, eigvecs = np.linalg.eigh(corr)
        eigvals_fixed = np.maximum(eigvals_fixed, 1e-8)  # Ensure all eigenvalues > 0
        corr = (eigvecs * eigvals_fixed) @ eigvecs.T
        # Renormalize
        diag_sqrt = np.sqrt(np.diag(corr))
        corr = corr / (diag_sqrt[:, None] * diag_sqrt[None, :])
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -0.999, 0.999)
        np.fill_diagonal(corr, 1.0)
    
    # Set random seed
    if seed:
        rng_state = np.random.get_state()
        np.random.seed(seed)
    
    # Generate independent standard normal random variables
    z = np.random.standard_normal((n, len(var_symbols)))
    
    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        if seed:
            np.random.set_state(rng_state)
        # Provide more helpful error message
        eigvals_check = np.linalg.eigvals(corr)
        min_eig = np.min(eigvals_check)
        raise np.linalg.LinAlgError(
            f"Unable to obtain Cholesky factor for correlation matrix. "
            f"Matrix is not positive definite (minimum eigenvalue: {min_eig:.2e}). "
            f"Please check your correlation values - they may be too extreme or inconsistent. "
            f"Try adjusting correlations to be less extreme (e.g., avoid values very close to ±1.0)."
        ) from exc
    
    # Transform to correlated normals
    correlated_z = z @ L.T
    
    # Convert to uniform [0,1] using normal CDF
    uniform_samples = stats.norm.cdf(correlated_z)
    uniform_samples = np.clip(uniform_samples, 1e-12, 1.0 - 1e-12)
    
    # Apply inverse CDF (PPF) for each variable's distribution
    samples: Dict[str, np.ndarray] = {}
    for i, sym in enumerate(var_symbols):
        if sym not in variables_config:
            continue
        
        cfg = variables_config[sym]
        dist_type = cfg.get("type", "Normal")
        params = cfg.get("params", {})
        u = uniform_samples[:, i]
        
        # Sample using inverse CDF based on distribution type
        try:
            samples[sym] = sample_distribution(dist_type, params, n, uniform=u)
        except Exception as e:
            # Fallback to independent sampling
            samples[sym] = sample_distribution(dist_type, params, n)
    
    # Handle extreme correlations by perfect dependence
    for i, j, sign in extreme_pairs:
        sym_i, sym_j = var_symbols[i], var_symbols[j]
        if sym_i in samples and sym_j in samples:
            x_orig = samples[sym_i].copy()
            y_orig = samples[sym_j].copy()
            order_x = np.argsort(x_orig)
            order_y = np.argsort(y_orig)
            if sign >= 0:
                y_rearranged = y_orig[order_y]
            else:
                y_rearranged = y_orig[order_y[::-1]]
            inv_order = np.empty_like(order_x)
            inv_order[order_x] = np.arange(len(order_x))
            samples[sym_j] = y_rearranged[inv_order]
    
    if seed:
        np.random.set_state(rng_state)
    
    return samples

def render_dependency_page():
    """Page 3: Dependency - Correlation Matrix with Cross Plots"""
    st.markdown("## Dependency Matrix")
    st.markdown("Define dependencies between parameters. Values range from -0.99 (strong negative) to +0.99 (strong positive).")
    
    # Check if variables are defined
    if "variables_config" not in st.session_state or not st.session_state.variables_config:
        st.warning("⚠️ Please define variables on the 'Input Variables' page first.")
        return
    
    variables_config = st.session_state.variables_config
    var_symbols = st.session_state.get("var_symbols", list(variables_config.keys()))
    
    if len(var_symbols) < 2:
        st.warning("⚠️ At least 2 variables must be defined to use dependency matrix")
        # CRITICAL: Don't reset correlation_values or correlation_matrix here
        # Just disable correlation, but keep the values for when user adds more variables
        st.session_state.use_correlation = False
        # Don't reset correlation_matrix - keep it for when variables are added back
        return
    
    # Get variable names for display
    var_names = [variables_config.get(sym, {}).get("name", sym) for sym in var_symbols]
    n_vars = len(var_symbols)
    
    # Initialize correlation matrix storage
    if "correlation_values" not in st.session_state:
        st.session_state.correlation_values = {}
    
    # Build correlation matrix from sliders
    corr_matrix = np.eye(n_vars)  # Start with identity matrix (no correlation)
    
    # Create sliders for each variable pair (matching the image style)
    st.markdown("### Define Correlations")
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            pair_key = f"{var_symbols[i]}_{var_symbols[j]}"
            widget_key = f"corr_{pair_key}"
            
            # CRITICAL: Read from widget's session_state key first (this is the source of truth)
            # Streamlit widgets with key= automatically store values in st.session_state[key]
            # This ensures values persist across navigation, value changes, and simulation runs
            if widget_key in st.session_state:
                # Widget key exists - use it (this is the current value from the widget)
                default_value = st.session_state[widget_key]
            elif pair_key in st.session_state.correlation_values:
                # Fallback to correlation_values dict (for backward compatibility)
                default_value = st.session_state.correlation_values[pair_key]
            else:
                # First time - use 0.0
                default_value = 0.0
            
            # Create slider with improved formatting
            # Widget with key= automatically manages session_state[widget_key]
            # The value= parameter should match what's in session_state to prevent resets
            value = st.slider(
                f"**{var_names[i]} ↔ {var_names[j]}**",
                min_value=-0.99,
                max_value=0.99,
                value=float(default_value),
                step=0.01,
                format="%.2f",
                key=widget_key,
                help=f"Correlation between {var_names[i]} and {var_names[j]}. Positive values indicate variables increase together, negative values indicate inverse relationship."
            )
            
            # Update correlation matrix with widget value (this is the source of truth)
            corr_matrix[i, j] = value
            corr_matrix[j, i] = value
            
            # Also store in correlation_values dict for backward compatibility
            # But the widget's session_state key (widget_key) is the primary source of truth
            st.session_state.correlation_values[pair_key] = value
    
    # Validate and fix correlation matrix
    is_valid, error_msg = validate_dependency_matrix(corr_matrix, var_symbols)
    
    if not is_valid:
        st.warning(f"⚠️ Correlation matrix needs adjustment: {error_msg}")
        st.info("Attempting to fix correlation matrix using Higham projection...")
        try:
            corr_matrix = fix_correlation_matrix(corr_matrix)
            is_valid, error_msg = validate_dependency_matrix(corr_matrix, var_symbols)
            if is_valid:
                st.success("✅ Correlation matrix fixed using Higham projection.")
            else:
                st.error(f"❌ Could not fix correlation matrix: {error_msg}")
                st.session_state.correlation_matrix = None
                st.session_state.use_correlation = False
                return
        except Exception as e:
            st.error(f"❌ Error fixing correlation matrix: {e}")
            st.session_state.correlation_matrix = None
            st.session_state.use_correlation = False
            return
    
    # Store validated correlation matrix AND the variable order used to build it
    # CRITICAL: Store the var_symbols order so we can verify it matches when applying correlation
    st.session_state.correlation_matrix = corr_matrix
    st.session_state.correlation_var_symbols = var_symbols.copy()  # Store the order used
    st.session_state.use_correlation = True
    
    # Generate samples for cross plots (before and after correlation)
    st.markdown("### Correlation Cross Plots")
    st.markdown("Scatter plots showing parameter relationships before (gray) and after (red) applying correlations. Every 100th trial is shown. Linear regression lines, formulas, and R² values are displayed.")
    st.markdown(
        '<div style="background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; border-radius: 4px;">'
        '<strong>Note:</strong> You need to run simulation on the \'Formula Definition\' page for cross plots to update with the latest correlation settings.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Check if simulation has been run - use actual simulation samples if available
    simulation_samples = st.session_state.get("samples", None)
    simulation_unconditional = st.session_state.get("unconditional_samples", None)
    
    # Determine which samples to use
    if simulation_samples is not None and all(sym in simulation_samples for sym in var_symbols):
        # Use actual simulation samples (these are the ones used in calculations)
        n_samples_actual = len(next(iter(simulation_samples.values())))
        
        # Check if correlation was actually applied in the simulation
        use_correlation_was_applied = st.session_state.get("use_correlation", False)
        
        if use_correlation_was_applied:
            # Correlation was applied - use simulation samples as "after"
            samples_after = simulation_samples.copy()
            
            # For "before", generate truly independent samples with same seed and n
            # This ensures fair comparison - same seed, same number of samples, but no correlation
            seed = st.session_state.get("seed", 0)
            if seed:
                np.random.seed(seed)
            samples_before = {}
            for sym in var_symbols:
                if sym in variables_config:
                    cfg = variables_config[sym]
                    dist_type = cfg.get("type", "Normal")
                    params = cfg.get("params", {})
                    # Generate independent samples (no correlation)
                    samples_before[sym] = sample_distribution(dist_type, params, n_samples_actual)
            
            st.info(f"Using actual simulation samples (n={n_samples_actual:,}) from the last simulation run. Correlation was applied. These are the samples used in your calculations.")
        else:
            # Correlation was NOT applied - both before and after are independent
            samples_after = simulation_samples.copy()
            samples_before = simulation_samples.copy()
            st.info(f"Using actual simulation samples (n={n_samples_actual:,}) from the last simulation run. Note: Correlation was not enabled during simulation.")
    else:
        # Simulation hasn't been run yet - generate preview samples
        preview_n = 10000
        seed = st.session_state.get("seed", 0)
        
        # Generate independent samples (before correlation)
        samples_before = {}
        if seed:
            np.random.seed(seed)
        for sym in var_symbols:
            if sym in variables_config:
                cfg = variables_config[sym]
                dist_type = cfg.get("type", "Normal")
                params = cfg.get("params", {})
                samples_before[sym] = sample_distribution(dist_type, params, preview_n)
        
        # Generate correlated samples (after correlation)
        try:
            samples_after = sample_correlated_variables(
                variables_config, 
                corr_matrix, 
                var_symbols, 
                preview_n, 
                seed
            )
        except Exception as e:
            st.error(f"❌ Error generating correlated samples: {e}")
            samples_after = samples_before.copy()  # Fallback to independent
        
        st.info(f"💡 Preview mode: Showing preview samples (n={preview_n:,}). Run simulation on 'Formula Definition' page to see cross plots using actual calculation data.")
    
    # Create cross plots for each variable pair (only show if correlation != 0)
    plot_every_nth = 100  # Show every 100th point for clarity
    
    # Track if we have any non-zero correlations to show
    has_nonzero_correlations = False
    
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            sym_i, sym_j = var_symbols[i], var_symbols[j]
            name_i, name_j = var_names[i], var_names[j]
            
            if sym_i not in samples_before or sym_j not in samples_before:
                continue
            
            # Get correlation value
            corr_value = corr_matrix[i, j]
            
            # Only show plot if correlation is significantly different from zero
            if abs(corr_value) < 0.001:
                continue
            
            has_nonzero_correlations = True
            
            # Sample data (every nth point)
            x_before = samples_before[sym_i][::plot_every_nth]
            y_before = samples_before[sym_j][::plot_every_nth]
            x_after = samples_after.get(sym_i, samples_before[sym_i])[::plot_every_nth]
            y_after = samples_after.get(sym_j, samples_before[sym_j])[::plot_every_nth]
            
            # Calculate linear regression for before
            if len(x_before) > 1:
                coeffs_before = np.polyfit(x_before, y_before, 1)
                poly_before = np.poly1d(coeffs_before)
                x_line_before = np.linspace(x_before.min(), x_before.max(), 100)
                y_line_before = poly_before(x_line_before)
                r2_before = np.corrcoef(x_before, y_before)[0, 1]**2
                formula_before = f"y = {coeffs_before[0]:.4f}x + {coeffs_before[1]:.4f}"
            else:
                coeffs_before = [0, 0]
                x_line_before = y_line_before = np.array([])
                r2_before = 0
                formula_before = "y = 0x + 0"
            
            # Calculate linear regression for after
            if len(x_after) > 1:
                coeffs_after = np.polyfit(x_after, y_after, 1)
                poly_after = np.poly1d(coeffs_after)
                x_line_after = np.linspace(x_after.min(), x_after.max(), 100)
                y_line_after = poly_after(x_line_after)
                r2_after = np.corrcoef(x_after, y_after)[0, 1]**2
                formula_after = f"y = {coeffs_after[0]:.4f}x + {coeffs_after[1]:.4f}"
            else:
                coeffs_after = [0, 0]
                x_line_after = y_line_after = np.array([])
                r2_after = 0
                formula_after = "y = 0x + 0"
            
            # Create plot
            fig = go.Figure()
            
            # Before correlation (darker gray)
            fig.add_trace(go.Scatter(
                x=x_before,
                y=y_before,
                mode='markers',
                name='Before correlation',
                marker=dict(color='gray', size=4, opacity=0.7),
                hovertemplate=f'{name_i}: %{{x:.4g}}<br>{name_j}: %{{y:.4g}}<extra></extra>'
            ))
            
            if len(x_line_before) > 0:
                fig.add_trace(go.Scatter(
                    x=x_line_before,
                    y=y_line_before,
                    mode='lines',
                    name=f'Before: {formula_before} (R² = {r2_before:.4f})',
                    line=dict(color='gray', width=2, dash='dash'),
                    hovertemplate='Regression line (before)<extra></extra>'
                ))
            
            # After correlation (red)
            fig.add_trace(go.Scatter(
                x=x_after,
                y=y_after,
                mode='markers',
                name='After correlation',
                marker=dict(color='red', size=4, opacity=0.7),
                hovertemplate=f'{name_i}: %{{x:.4g}}<br>{name_j}: %{{y:.4g}}<extra></extra>'
            ))
            
            if len(x_line_after) > 0:
                fig.add_trace(go.Scatter(
                    x=x_line_after,
                    y=y_line_after,
                    mode='lines',
                    name=f'After: {formula_after} (R² = {r2_after:.4f})',
                    line=dict(color='red', width=2),
                    hovertemplate='Regression line (after)<extra></extra>'
                ))
            
            fig.update_layout(
                title=f"{name_i} vs {name_j} (Correlation: {corr_value:.3f})",
                xaxis_title=name_i,
                yaxis_title=name_j,
                hovermode='closest',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show formulas and R² values
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Before Correlation:**")
                st.markdown(f"- Formula: `{formula_before}`")
                st.markdown(f"- R²: `{r2_before:.4f}`")
            with col2:
                st.markdown(f"**After Correlation:**")
                st.markdown(f"- Formula: `{formula_after}`")
                st.markdown(f"- R²: `{r2_after:.4f}`")
    
    st.markdown("---")
    
    # Show message if no correlations are defined
    if not has_nonzero_correlations:
        st.info("ℹ️ No correlations defined (all values are 0.000). Adjust the sliders above to see cross plots.")
    
    # Documentation section
    with st.expander("ℹ️ About Correlation Methods", expanded=True):
        st.markdown("""
        **Correlation Implementation:**
        
        **Higham Projection:**
        - Ensures the correlation matrix is positive semi-definite (mathematically valid)
        - Projects user-defined correlations to the nearest valid correlation matrix
        - Uses iterative algorithm (Higham, 2002) to find the closest valid matrix
        - Automatically applied when correlation matrix is invalid
        
        **Cholesky Decomposition:**
        - Generates correlated samples using Cholesky factorization
        - Process: L * L^T = correlation_matrix, then X = L * Z
        - Preserves individual variable distributions while introducing correlations
        - Mathematically rigorous and computationally efficient
        
        **How It Works:**
        1. User defines correlations via sliders (may not be perfectly valid)
        2. Higham projection ensures matrix is positive semi-definite
        3. Generate independent standard normal random variables Z
        4. Apply Cholesky decomposition: L = Cholesky(correlation_matrix)
        5. Transform to correlated normals: X = L * Z
        6. Convert to uniform [0,1] using normal CDF: U = Φ(X)
        7. Apply inverse CDF of each variable's distribution: Y = F⁻¹(U)
        
        **Key Benefits:**
        - **Distribution Preservation:** Each variable maintains its original distribution shape
        - **Mathematical Rigor:** Uses proper multivariate normal theory
        - **Stability:** Automatically handles invalid correlation matrices
        - **Visualization:** Cross plots show before/after correlation effects
        """)

def render_formula_definition_page():
    """Page 4: Formula Definition - Build Formulas and Run Simulation"""
    st.markdown("## Formula Definition")
    st.markdown("Define formulas to compute results from your input variables. Formulas can reference variables and earlier results.")
    
    # Define defaults dictionary for reference (used for fallback values)
    DEFAULT_VALUES_V08 = {
        "name_a": "K",
        "name_b": "μ",
        "name_c": "ρ",
        "dtype_a": "Triangular",
        "dtype_b": "Normal",
        "dtype_c": "PERT",
        "a_low": 30e9,
        "a_mode": 35e9,
        "a_high": 40e9,
        "b_mean": 30e9,
        "b_sd": 3e9,
        "c_min": 2550.0,
        "c_most_likely": 2650.0,
        "c_max": 2800.0,
        "prob_a": 1.0,
        "prob_b": 1.0,
        "prob_c": 1.0,
    }
    
    # Check if variables are defined
    if "variables_config" not in st.session_state or not st.session_state.variables_config:
        st.warning("⚠️ Please define variables on the 'Input Variables' page first.")
        return
    
    variables_config = st.session_state.variables_config
    # Ensure var_symbols is always defined, even if not in session_state
    var_symbols = st.session_state.get("var_symbols", [])
    if not var_symbols:
        # Fallback: get symbols from variables_config keys
        var_symbols = list(variables_config.keys())
        st.session_state.var_symbols = var_symbols
    
    # Build variable mapping for display
    var_mapping = {}
    for sym in var_symbols:
        if sym in variables_config:
            var_name = variables_config[sym].get("name", sym)
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
            {"name": "V_p", "expr": "sqrt((a + (4/3)*b) / c)"}
        ]
    
    cols = st.columns([2, 1, 1])
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
    alias_slug_to_name = {_slugify(name): name for name in alias_index_to_name.values()}
    
    for i, f in enumerate(st.session_state.formulas):
        with st.expander(f"Formula {i+1}: {f['name']}", expanded=True):
            cols_f = st.columns([3, 1])
            with cols_f[0]:
                f["name"] = st.text_input("Result name", value=f["name"], key=f"fname_{i}", 
                                        help="Tip: Use subscripts with underscores or LaTeX. Examples: Oil_prod → Oil_{prod}, or write N_{gas}.")
                f["expr"] = st.text_input("Expression", value=f["expr"], key=f"fexpr_{i}", 
                                        help="Use a,b,c or sidebar names (slugified), and earlier results: f1, f2, ... or res_<slug>.")
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
            
            # Basic prettification for LaTeX rendering
            def to_latex(expr: str) -> str:
                """
                Convert mathematical expression to LaTeX.
                Based on the working GitHub version.
                Processing order:
                1. Powers (**n -> ^{n})
                2. Multiplication (* -> \cdot)
                3. sqrt(...) -> \sqrt{...}
                4. Simple top-level division: a/b -> \frac{a}{b}
                5. Also convert a/b inside \sqrt{...} to \frac{a}{b}
                """
                s = expr
                
                # Step 1: Powers: **n -> ^{n}
                s = re.sub(r"\*\*\s*([0-9]+)", r"^{\1}", s)
                
                # Step 2: Multiplication: * -> \cdot
                s = s.replace('*', '\\cdot ')
                
                # Step 3: sqrt(...) -> \sqrt{...}
                def replace_sqrt(t: str) -> str:
                    out = []
                    i = 0
                    while i < len(t):
                        if t.startswith('sqrt(', i):
                            i0 = i + 5  # After 'sqrt('
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
                
                # Step 4: Simple top-level division: a/b -> \frac{a}{b} (respect () and {} nesting)
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
                
                # Step 5: Also convert a/b inside \sqrt{...} to \frac{a}{b}
                def convert_frac_inside_sqrt(t: str) -> str:
                    out = []
                    i = 0
                    while i < len(t):
                        if t.startswith('\\sqrt{', i):
                            # Find matching closing '}' for this sqrt
                            j = i + 6  # Position after '\\sqrt{'
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
            
            # Left-hand side result name, support subscripts
            name_raw = f["name"]
            if any(ch in name_raw for ch in ['\\', '{', '}']):
                lhs_tex = name_raw  # assume user provided LaTeX
            elif '_' in name_raw:
                base, sub = name_raw.split('_', 1)
                lhs_tex = base.replace(' ', r'\ ') + '_{' + sub + '}'
            else:
                lhs_tex = name_raw.replace(' ', r'\ ')
            # Display LaTeX using st.latex() for proper rendering
            st.latex(f"{lhs_tex} = {latex_expr}")
    
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
        pass  # Button moved below info message
    
    # ========================================================================
    # SIMULATION: Run when button is pressed
    # ========================================================================
    # Check if simulation should run
    run_simulation = st.session_state.get("run_simulation", False)
    
    # Reset the run_simulation flag after checking
    if run_simulation:
        st.session_state.run_simulation = False
    
    if st.session_state.formulas and run_simulation:
        # CRITICAL: Always rebuild variables_config from session_state widget values before simulation
        # Widgets with key= automatically update session_state, so we read from there
        # This ensures we use the absolute latest parameter values, even if user just changed them
        # Get var_symbols first
        var_symbols = st.session_state.get("var_symbols", [])
        if not var_symbols:
            # Fallback: try to get from variables_config keys
            existing_config = st.session_state.get("variables_config", {})
            if existing_config:
                var_symbols = list(existing_config.keys())
            else:
                st.warning("⚠️ No variables defined. Please go to 'Input Variables' page to define variables first.")
                var_symbols = []
        
        # FOOLPROOF: Rebuild variables_config from session_state widget values
        # session_state contains the latest widget values (updated automatically by widgets with key=)
        variables_config_rebuilt: Dict[str, Dict[str, Any]] = {}
        for sym in var_symbols:
            name_key = f"name_{sym}"
            prob_key = f"prob_{sym}"
            dtype_key = f"dtype_{sym}"
            
            # Read all values from session_state (updated by widgets with key=)
            # These are the CURRENT values shown in the widgets
            DEFAULT_NAMES = {"a": "K", "b": "μ", "c": "ρ"}
            name = st.session_state.get(name_key, "").strip()
            if not name:
                name = DEFAULT_NAMES.get(sym, sym)
            
            # Get prob from session_state, with fallback to existing config to preserve user's setting
            prob = st.session_state.get(prob_key)
            if prob is None:
                # Fallback: try to get from existing config to preserve user's setting
                existing_config = st.session_state.get("variables_config", {})
                if sym in existing_config:
                    prob = existing_config[sym].get("prob", 1.0)
                else:
                    prob = 1.0
            prob = float(prob)
            dtype = st.session_state.get(dtype_key)
            if not dtype or dtype not in DISTROS:
                # Fallback to default dtype based on symbol
                if sym == 'a':
                    dtype = "Triangular"
                elif sym == 'b':
                    dtype = "Normal"
                elif sym == 'c':
                    dtype = "PERT"
                else:
                    dtype = list(DISTROS.keys())[0] if DISTROS else "Triangular"
            
            # Rebuild params from session_state (these are the CURRENT widget values)
            params_spec = DISTROS[dtype]["params"]
            param_values_rebuilt = {}
            for label, default, ptype in params_spec:
                key = f"{sym}_{label}"
                # Read from session_state - this is the CURRENT value from the widget
                if key in st.session_state:
                    if ptype == float:
                        param_values_rebuilt[label] = float(st.session_state[key])
                    elif ptype == int:
                        param_values_rebuilt[label] = int(st.session_state[key])
                    else:
                        param_values_rebuilt[label] = str(st.session_state[key])
                else:
                    # Fallback: try to get from existing config
                    existing_config = st.session_state.get("variables_config", {})
                    if sym in existing_config and label in existing_config[sym].get("params", {}):
                        param_values_rebuilt[label] = existing_config[sym]["params"][label]
                    else:
                        # Last resort: use distribution default
                        param_values_rebuilt[label] = default
            
            variables_config_rebuilt[sym] = {"name": name, "prob": prob, "type": dtype, "params": param_values_rebuilt}
            
        # CRITICAL: Update session_state with rebuilt config BEFORE simulation
        # This ensures simulation uses the latest values
        st.session_state.variables_config = variables_config_rebuilt.copy()
        import time
        st.session_state.variables_config_last_updated = time.time()
        variables_config = variables_config_rebuilt
        
        if not variables_config:
            st.warning("⚠️ No variables defined. Please go to 'Input Variables' page to define variables first.")
        else:
            n_samples = st.session_state.get("n_samples", 50_000)
            seed = st.session_state.get("seed", 0)
            
            # AUTOMATIC: Regenerate samples from current variables_config
            # This runs automatically whenever the page loads with formulas defined
            try:
                samples, unconditional_samples = simulate_variables(variables_config, int(n_samples), int(seed))
                
                # Check if correlation should be applied
                use_correlation = st.session_state.get("use_correlation", False)
                correlation_matrix = st.session_state.get("correlation_matrix", None)
                correlation_var_symbols = st.session_state.get("correlation_var_symbols", None)
                
                # If correlation is enabled, apply correlation to the newly generated samples
                if use_correlation and correlation_matrix is not None:
                    # CRITICAL: Verify variable order matches the order used to build correlation matrix
                    if correlation_var_symbols is not None:
                        # Check if current var_symbols matches the order used for correlation matrix
                        if list(correlation_var_symbols) != list(var_symbols):
                            st.warning(f"⚠️ Variable order mismatch! Correlation matrix was built with order {correlation_var_symbols}, but current order is {var_symbols}. Correlation may not be applied correctly. Please reconfigure correlations on the Dependency page.")
                            # Try to reorder correlation matrix to match current var_symbols order
                            try:
                                # Create mapping from old order to new order
                                old_to_new = {sym: i for i, sym in enumerate(correlation_var_symbols)}
                                new_order = [old_to_new.get(sym, -1) for sym in var_symbols]
                                
                                # Check if all variables are present
                                if any(idx == -1 for idx in new_order) or len(new_order) != len(correlation_var_symbols):
                                    st.error("❌ Cannot reorder correlation matrix: variable set has changed. Please reconfigure correlations.")
                                    correlation_matrix = None
                                else:
                                    # Reorder correlation matrix
                                    correlation_matrix = correlation_matrix[np.ix_(new_order, new_order)]
                                    st.info("ℹ️ Correlation matrix reordered to match current variable order.")
                            except Exception as e:
                                st.error(f"❌ Error reordering correlation matrix: {e}. Please reconfigure correlations.")
                                correlation_matrix = None
                    
                    # Verify correlation matrix size matches number of variables
                    if correlation_matrix is not None and correlation_matrix.shape[0] != len(var_symbols):
                        st.error(f"❌ Correlation matrix size ({correlation_matrix.shape[0]}) does not match number of variables ({len(var_symbols)}). Please reconfigure correlations.")
                        correlation_matrix = None
                    
                    if correlation_matrix is not None:
                        try:
                            correlated_samples_dict = sample_correlated_variables(
                                variables_config, correlation_matrix, var_symbols, n_samples, seed
                            )
                            
                            # CRITICAL: Replace ALL samples with correlated samples (not merge)
                            # This ensures correlation is actually applied
                            for sym in var_symbols:
                                if sym in correlated_samples_dict:
                                    samples[sym] = correlated_samples_dict[sym].copy()
                                    # Apply occurrence probability if needed
                                    if sym in variables_config:
                                        p = float(variables_config[sym].get("prob", 1.0))
                                        if p < 1.0:
                                            # Need to regenerate mask with same seed for consistency
                                            if seed:
                                                np.random.seed(seed)
                                            mask = np.random.binomial(1, p, size=n_samples)
                                            samples[sym] = samples[sym] * mask
                            
                            # Verify correlation was applied by checking a sample correlation
                            if len(var_symbols) >= 2:
                                # Check correlation between first two variables
                                sym1, sym2 = var_symbols[0], var_symbols[1]
                                if sym1 in samples and sym2 in samples:
                                    actual_corr = np.corrcoef(samples[sym1], samples[sym2])[0, 1]
                                    expected_corr = correlation_matrix[0, 1]
                                    if abs(actual_corr - expected_corr) > 0.1:
                                        st.warning(f"⚠️ Correlation verification: Expected {expected_corr:.3f} between {sym1} and {sym2}, but got {actual_corr:.3f}. This may indicate an issue with correlation application.")
                        except Exception as e:
                            st.error(f"❌ Error applying correlation: {e}")
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
                            st.info("Continuing with uncorrelated samples...")
                
                # Store updated samples in session state
                st.session_state.samples = samples
                st.session_state.unconditional_samples = unconditional_samples
                
                # AUTOMATIC: Evaluate formulas immediately
                errors = []
                results: Dict[str, np.ndarray] = {}
                results_unconditional: Dict[str, np.ndarray] = {}

                # Work on mutable environments that we enrich as we go
                env_cond = {**samples}
                env_uncond = {**unconditional_samples} if unconditional_samples else {**samples}

                # Add aliases for variables by their given Name (slugified)
                for sym, spec in variables_config.items():
                    var_name = (spec.get("name", "") or "").strip()
                    if var_name:
                        alias = _slugify(var_name)
                        if alias and alias not in env_cond and sym in samples:
                            env_cond[alias] = samples.get(sym)
                        if alias and alias not in env_uncond and sym in unconditional_samples:
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
                    for err in errors:
                        st.error(err)
                
                # Store results in session state for Results page
                st.session_state.results = results
                st.session_state.results_unconditional = results_unconditional
                st.session_state.simulation_completed = True
                
                # Show status message
                if not errors:
                    st.success(f"✅ {len(results)} formula(s) evaluated successfully.")
                else:
                    st.warning("⚠️ Some formulas could not be evaluated. Check errors above.")
            except Exception as e:
                st.error(f"❌ Error during simulation: {type(e).__name__}: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        if st.session_state.formulas:
            st.info("Click 'RUN Simulation' button below to compute results.")
            # Run Simulation button
            if st.button("▶ RUN Simulation", type="primary", use_container_width=True):
                st.session_state.run_simulation = True
                st.rerun()

def render_results_page():
    """Page 5: Results - Analysis and Export"""
    st.markdown("## Results")
    
    # Check if simulation has been run (now runs automatically on formula page)
    if "simulation_completed" not in st.session_state or not st.session_state.get("simulation_completed", False):
        st.warning("⚠️ No simulation results found. Please go to 'Formula Definition' page and define formulas. Results update automatically.")
        return

    if "results" not in st.session_state or not st.session_state.results:
        st.warning("⚠️ No results to display. Please define formulas on the 'Formula Definition' page. Results update automatically.")
        return
    
    results = st.session_state.results
    results_unconditional = st.session_state.get("results_unconditional", {})
    # CRITICAL: Always use current variables_config from session_state
    # This ensures SimDec and other analyses use the latest variable definitions
    variables_config = st.session_state.get("variables_config", {})
    samples = st.session_state.get("samples", {})
    unconditional_samples = st.session_state.get("unconditional_samples", {})
    
    # Ensure variables_config is current by rebuilding from session_state if needed
    # This is a safety check to ensure consistency
    if variables_config:
        var_symbols_results = st.session_state.get("var_symbols", list(variables_config.keys()))
        # Verify all symbols in variables_config have corresponding session_state entries
        for sym in var_symbols_results:
            if sym in variables_config:
                # Check if name is in session_state and matches
                name_key = f"name_{sym}"
                if name_key in st.session_state:
                    stored_name = st.session_state[name_key].strip()
                    if stored_name and stored_name != variables_config[sym].get("name", ""):
                        # Update name from session_state
                        variables_config[sym]["name"] = stored_name
    
    # Show formulas that were evaluated
    st.markdown("### Formulas Evaluated")
    if "formulas" in st.session_state:
        for i, f in enumerate(st.session_state.formulas, start=1):
            nm = f.get("name", f"result{i}")
            expr = f.get("expr", "")
            
            # Display as code
            st.code(f"Formula {i}: {nm} = {expr}", language="text")
            
            # Also display as LaTeX
            # Get variable mapping for display
            var_mapping = {}
            var_symbols = st.session_state.get("var_symbols", [])
            variables_config = st.session_state.get("variables_config", {})
            for sym in var_symbols:
                if sym in variables_config:
                    var_name = variables_config[sym].get("name", sym)
                    var_mapping[sym] = var_name
            
            # Replace symbols with names in the displayed formula
            display_expr = expr
            for sym, var_name in var_mapping.items():
                if sym in display_expr and var_name != sym:
                    display_expr = display_expr.replace(sym, var_name)
            
            # Convert to LaTeX using the same function from formula definition page
            def to_latex(expr: str) -> str:
                """
                Convert mathematical expression to LaTeX.
                Based on the working GitHub version.
                Processing order:
                1. Powers (**n -> ^{n})
                2. Multiplication (* -> \cdot)
                3. sqrt(...) -> \sqrt{...}
                4. Simple top-level division: a/b -> \frac{a}{b}
                5. Also convert a/b inside \sqrt{...} to \frac{a}{b}
                """
                s = expr
                
                # Step 1: Powers: **n -> ^{n}
                s = re.sub(r"\*\*\s*([0-9]+)", r"^{\1}", s)
                
                # Step 2: Multiplication: * -> \cdot
                s = s.replace('*', '\\cdot ')
                
                # Step 3: sqrt(...) -> \sqrt{...}
                def replace_sqrt(t: str) -> str:
                    out = []
                    i = 0
                    while i < len(t):
                        if t.startswith('sqrt(', i):
                            i0 = i + 5  # After 'sqrt('
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
                
                # Step 4: Simple top-level division: a/b -> \frac{a}{b} (respect () and {} nesting)
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
                
                # Step 5: Also convert a/b inside \sqrt{...} to \frac{a}{b}
                def convert_frac_inside_sqrt(t: str) -> str:
                    out = []
                    i = 0
                    while i < len(t):
                        if t.startswith('\\sqrt{', i):
                            # Find matching closing '}' for this sqrt
                            j = i + 6  # Position after '\\sqrt{'
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
            
            # Format result name for LaTeX
            name_raw = nm
            if any(ch in name_raw for ch in ['\\', '{', '}']):
                lhs_tex = name_raw  # assume user provided LaTeX
            elif '_' in name_raw:
                base, sub = name_raw.split('_', 1)
                lhs_tex = base.replace(' ', r'\ ') + '_{' + sub + '}'
            else:
                lhs_tex = name_raw.replace(' ', r'\ ')
            
            try:
                latex_expr = to_latex(display_expr)
                # Display LaTeX using st.latex() for proper rendering
                st.latex(f"{lhs_tex} = {latex_expr}")
            except Exception:
                # If LaTeX conversion fails, just skip it
                pass
    
    st.markdown("---")

    # Check if any variable has occurrence probability
    has_occurrence = any(variables_config.get(sym, {}).get("prob", 1.0) < 1.0 for sym in variables_config.keys())
    
    # Summary statistics function
    def summarize(x: np.ndarray) -> Dict[str, float]:
        show_exceedance = True  # Use exceedance convention by default
        def p_lo_hi(x_arr: np.ndarray):
            if show_exceedance:
                return np.percentile(x_arr, 90), np.percentile(x_arr, 10)
            else:
                return np.percentile(x_arr, 10), np.percentile(x_arr, 90)
        
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
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    # Check if correlation was used and get correlation info
    use_correlation = st.session_state.get("use_correlation", False)
    correlation_matrix = st.session_state.get("correlation_matrix", None)
    correlation_var_symbols = st.session_state.get("correlation_var_symbols", None)
    var_symbols = st.session_state.get("var_symbols", list(variables_config.keys()))
    
    # Calculate actual correlations from samples and build correlation info text
    correlation_info = []
    if use_correlation and correlation_matrix is not None and samples:
        # Get variable names for display
        var_names_dict = {sym: variables_config.get(sym, {}).get("name", sym) for sym in var_symbols}
        
        # Calculate actual correlations from samples
        for i in range(len(var_symbols)):
            for j in range(i + 1, len(var_symbols)):
                sym_i, sym_j = var_symbols[i], var_symbols[j]
                if sym_i in samples and sym_j in samples:
                    actual_corr = np.corrcoef(samples[sym_i], samples[sym_j])[0, 1]
                    if abs(actual_corr) > 0.01:  # Only show significant correlations
                        name_i = var_names_dict.get(sym_i, sym_i)
                        name_j = var_names_dict.get(sym_j, sym_j)
                        # Calculate R²
                        r_squared = actual_corr ** 2
                        correlation_info.append(f"{name_i} ↔ {name_j} (R²: {r_squared:.4f})")
    
    correlation_text = ""
    if correlation_info:
        correlation_text = " **Note:** " + ", ".join(correlation_info) + " are correlated."
    
    # Results summary table
    st.markdown("### Results Summary Statistics")
    if correlation_text:
        st.info(f"Results shown WITH correlation applied.{correlation_text}")
    order_cols = ["mean", "mode", "min", "p90", "p50", "p10", "max", "sd", "skew", "kurtosis"]
    summary_rows = []
    for k, v_cond in results.items():
        v_uncond = results_unconditional.get(k, v_cond)

        if has_occurrence:
            row_cond = {"result": f"{k} (conditional)", **summarize(v_cond)}
            row_uncond = {"result": f"{k} (unconditional)", **summarize(v_uncond)}
            summary_rows.append(row_cond)
            summary_rows.append(row_uncond)
        else:
            row = {"result": k, **summarize(v_cond)}
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    present_res = [c for c in order_cols if c in summary_df.columns]
    summary_df = summary_df[["result"] + present_res]
    st.dataframe(summary_df, use_container_width=True)

    # Convergence Analysis
    with st.expander("📊 Sample Size Convergence Analysis", expanded=False):
        st.markdown("""
        **Determine the minimum number of trials needed for stable results.**
        
        This analysis runs simulations with increasing sample sizes and tracks when key statistics (mean, std, percentiles) stabilize.
        """)
        
        if st.button("Run Convergence Analysis", type="primary"):
            with st.spinner("Running convergence analysis... This may take a minute."):
                # Sample sizes to test (logarithmic progression)
                sample_sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
                max_samples = min(max(sample_sizes), 200000)  # Cap at 200K for performance
                sample_sizes = [n for n in sample_sizes if n <= max_samples]
                
                # Convergence threshold (relative change)
                threshold = st.slider("Convergence threshold (%)", 0.1, 5.0, 1.0, 0.1, 
                                     help="Statistics are considered converged when relative change between consecutive sample sizes is below this threshold")
                
                seed = st.session_state.get("seed", 0)
                convergence_data = {}
                
                # Track statistics for each result across sample sizes
                for result_name in results.keys():
                    stats_history = {
                        "n": [],
                        "mean": [],
                        "sd": [],
                        "p10": [],
                        "p50": [],
                        "p90": []
                    }
                    
                    prev_stats = None
                    converged_at = None
                    
                    for n in sample_sizes:
                        try:
                            # Run simulation with n samples
                            temp_samples, temp_unconditional = simulate_variables(
                                variables_config, n, seed
                            )
                            
                            # Apply correlation if enabled
                            if use_correlation and correlation_matrix is not None:
                                temp_samples = sample_correlated_variables(
                                    variables_config, correlation_matrix, var_symbols, n, seed
                                )
                            
                            # Build evaluation environment
                            env = {**temp_samples}
                            for sym, spec in variables_config.items():
                                var_name = (spec.get("name", "") or "").strip()
                                if var_name:
                                    alias = _slugify(var_name)
                                    if alias and alias not in env and sym in temp_samples:
                                        env[alias] = temp_samples[sym]
                            
                            # Add formula aliases as we evaluate
                            for idx, f in enumerate(st.session_state.formulas, start=1):
                                nm = (f["name"].strip() or "result")
                                if nm == result_name:
                                    ex = f["expr"].strip()
                                    try:
                                        y = evaluate_expression(ex, env)
                                        y = np.asarray(y, dtype=float)
                                        
                                        # Calculate statistics
                                        show_exceedance = True
                                        def p_lo_hi(x_arr: np.ndarray):
                                            if show_exceedance:
                                                return np.percentile(x_arr, 90), np.percentile(x_arr, 10)
                                            else:
                                                return np.percentile(x_arr, 10), np.percentile(x_arr, 90)
                                        
                                        p10, p90 = p_lo_hi(y)
                                        current_stats = {
                                            "mean": float(np.mean(y)),
                                            "sd": float(np.std(y, ddof=1)),
                                            "p10": float(p10),
                                            "p50": float(np.percentile(y, 50)),
                                            "p90": float(p90),
                                        }
                                        
                                        stats_history["n"].append(n)
                                        stats_history["mean"].append(current_stats["mean"])
                                        stats_history["sd"].append(current_stats["sd"])
                                        stats_history["p10"].append(current_stats["p10"])
                                        stats_history["p50"].append(current_stats["p50"])
                                        stats_history["p90"].append(current_stats["p90"])
                                        
                                        # Check convergence
                                        if prev_stats is not None:
                                            # Calculate relative changes
                                            rel_changes = {}
                                            for key in ["mean", "sd", "p10", "p50", "p90"]:
                                                if abs(prev_stats[key]) > 1e-10:
                                                    rel_changes[key] = abs((current_stats[key] - prev_stats[key]) / prev_stats[key]) * 100
                                                else:
                                                    rel_changes[key] = abs(current_stats[key] - prev_stats[key])
                                            
                                            max_change = max(rel_changes.values())
                                            
                                            if max_change < threshold and converged_at is None:
                                                converged_at = n
                                        
                                        prev_stats = current_stats
                                        
                                        # Add to environment for next formula
                                        env[f"f{idx}"] = y
                                        slug = _slugify(nm)
                                        if slug:
                                            env[f"res_{slug}"] = y
                                        
                                        break
                                    except Exception:
                                        continue
                        except Exception as e:
                            st.warning(f"Error at n={n}: {e}")
                            continue
                    
                    convergence_data[result_name] = {
                        "stats": stats_history,
                        "converged_at": converged_at
                    }
                
                # Display results
                for result_name, data in convergence_data.items():
                    st.markdown(f"#### {result_name}")
                    
                    if data["converged_at"]:
                        st.success(f"✅ Converged at **{data['converged_at']:,}** samples (all statistics changed by < {threshold}%)")
                    else:
                        st.warning(f"⚠️ Not fully converged within tested range (up to {max(sample_sizes):,} samples). Consider increasing sample size.")
                    
                    # Create convergence plot
                    if len(data["stats"]["n"]) > 0:
                        fig = go.Figure()
                        
                        # Normalize statistics to show relative changes
                        base_mean = data["stats"]["mean"][0] if data["stats"]["mean"] else 1.0
                        base_sd = data["stats"]["sd"][0] if data["stats"]["sd"] else 1.0
                        base_p50 = data["stats"]["p50"][0] if data["stats"]["p50"] else 1.0
                        
                        # Plot normalized values (as percentage of initial value)
                        if abs(base_mean) > 1e-10:
                            fig.add_trace(go.Scatter(
                                x=data["stats"]["n"],
                                y=[(v / base_mean - 1) * 100 for v in data["stats"]["mean"]],
                                mode='lines+markers',
                                name='Mean (%)',
                                line=dict(color='blue', width=2)
                            ))
                        
                        if abs(base_sd) > 1e-10:
                            fig.add_trace(go.Scatter(
                                x=data["stats"]["n"],
                                y=[(v / base_sd - 1) * 100 for v in data["stats"]["sd"]],
                                mode='lines+markers',
                                name='Std Dev (%)',
                                line=dict(color='green', width=2)
                            ))
                        
                        if abs(base_p50) > 1e-10:
                            fig.add_trace(go.Scatter(
                                x=data["stats"]["n"],
                                y=[(v / base_p50 - 1) * 100 for v in data["stats"]["p50"]],
                                mode='lines+markers',
                                name='P50 (%)',
                                line=dict(color='orange', width=2)
                            ))
                        
                        # Add convergence threshold lines
                        fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                                     annotation_text=f"Threshold ({threshold}%)")
                        fig.add_hline(y=-threshold, line_dash="dash", line_color="red")
                        
                        # Add vertical line at convergence point
                        if data["converged_at"]:
                            fig.add_vline(x=data["converged_at"], line_dash="dot", line_color="green",
                                         annotation_text=f"Converged at {data['converged_at']:,}")
                        
                        fig.update_layout(
                            title=f"Convergence Analysis: {result_name}",
                            xaxis_title="Sample Size (n)",
                            yaxis_title="Relative Change from Initial Value (%)",
                            xaxis_type="log",
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show statistics table
                        conv_df = pd.DataFrame({
                            "Sample Size": data["stats"]["n"],
                            "Mean": [f"{v:,.3g}" for v in data["stats"]["mean"]],
                            "Std Dev": [f"{v:,.3g}" for v in data["stats"]["sd"]],
                            "P10": [f"{v:,.3g}" for v in data["stats"]["p10"]],
                            "P50": [f"{v:,.3g}" for v in data["stats"]["p50"]],
                            "P90": [f"{v:,.3g}" for v in data["stats"]["p90"]],
                        })
                        st.dataframe(conv_df, use_container_width=True, hide_index=True)
                
                # Summary recommendations
                st.markdown("### Recommendations")
                converged_results = [name for name, data in convergence_data.items() if data["converged_at"]]
                if converged_results:
                    max_converged = max([convergence_data[name]["converged_at"] for name in converged_results])
                    st.info(f"""
                    **Recommended minimum sample size: {max_converged:,}**
                    
                    All results have converged at this sample size or smaller. 
                    Using fewer samples may lead to unstable statistics.
                    """)
                else:
                    st.warning(f"""
                    **No convergence detected within tested range.**
                    
                    Consider:
                    - Increasing the maximum sample size tested
                    - Using a higher convergence threshold
                    - Your distributions may have high variance requiring more samples
                    """)
    
    st.markdown("---")
    
    # Helper function to create distribution plot
    def create_result_plot(values, title, color=PALETTE[0]):
        """Create histogram + CDF plot for a result distribution"""
        fig = go.Figure()

        mean_val = float(np.mean(values))
        p10 = float(np.percentile(values, 90))  # High value (exceedance convention)
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
            showlegend=True
        )
        
        # Calculate CDF (exceedance)
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
            showlegend=True
        ))

        # Add markers at P10, P50, Mean, P90
        def cdf_at(val):
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
        
        fig.update_layout(
            xaxis_title=f"{title}",
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
        return fig

    # Result distribution plots
    st.markdown("### Result Distributions")
    if correlation_text:
        st.info(f"Distributions shown WITH correlation applied.{correlation_text}")
    for k in results.keys():
        v_cond = results[k]
        v_uncond = results_unconditional.get(k, v_cond)

        if has_occurrence:
            st.markdown(f"#### {k} - Distribution Plots")
            col1, col2 = st.columns(2)
            with col1:
                fig_cond = create_result_plot(v_cond, f"{k} (Conditional)", color=PALETTE[0])
                st.plotly_chart(fig_cond, use_container_width=True)
            with col2:
                fig_uncond = create_result_plot(v_uncond, f"{k} (Unconditional)", color=PALETTE[1])
                st.plotly_chart(fig_uncond, use_container_width=True)
        else:
            fig = create_result_plot(v_cond, k, color=PALETTE[0])
            st.plotly_chart(fig, use_container_width=True)
    
    # Add expandable box for results WITHOUT correlation (if correlation was used)
    # Moved here so create_result_plot function is available
    if use_correlation and correlation_matrix is not None:
        with st.expander("Results WITHOUT Correlation (Comparison)", expanded=False):
            st.markdown("These results show what the simulation would produce if variables were sampled independently (no correlation). Compare with the main results above to see the impact of correlation.")
            
            # Generate independent samples
            n_samples = len(next(iter(samples.values()))) if samples else 50000
            seed = st.session_state.get("seed", 0)
            
            try:
                # Generate independent samples
                independent_samples, independent_unconditional = simulate_variables(
                    variables_config, n_samples, seed
                )
                
                # Evaluate formulas with independent samples
                env_indep_cond = {**independent_samples}
                env_indep_uncond = {**independent_unconditional} if independent_unconditional else {**independent_samples}
                
                # Add aliases
                for sym, spec in variables_config.items():
                    var_name = (spec.get("name", "") or "").strip()
                    if var_name:
                        alias = _slugify(var_name)
                        if alias and alias not in env_indep_cond and sym in independent_samples:
                            env_indep_cond[alias] = independent_samples.get(sym)
                        if alias and alias not in env_indep_uncond and sym in independent_unconditional:
                            env_indep_uncond[alias] = independent_unconditional.get(sym)
                
                # Build aliases for formulas
                for idx, f in enumerate(st.session_state.formulas, start=1):
                    nm = (f["name"].strip() or "result")
                    ex = f["expr"].strip()
                    try:
                        y_cond_indep = evaluate_expression(ex, env_indep_cond)
                        y_uncond_indep = evaluate_expression(ex, env_indep_uncond)
                        
                        y_cond_indep = np.asarray(y_cond_indep, dtype=float)
                        y_uncond_indep = np.asarray(y_uncond_indep, dtype=float)
                        
                        # Add formula aliases
                        env_indep_cond[f"f{idx}"] = y_cond_indep
                        env_indep_uncond[f"f{idx}"] = y_uncond_indep
                        slug = _slugify(nm)
                        if slug:
                            env_indep_cond[f"res_{slug}"] = y_cond_indep
                            env_indep_uncond[f"res_{slug}"] = y_uncond_indep
                    except Exception as e:
                        pass
                
                # Re-evaluate all formulas with independent samples
                results_indep: Dict[str, np.ndarray] = {}
                results_indep_unconditional: Dict[str, np.ndarray] = {}
                
                for idx, f in enumerate(st.session_state.formulas, start=1):
                    nm = (f["name"].strip() or "result")
                    ex = f["expr"].strip()
                    try:
                        y_cond_indep = evaluate_expression(ex, env_indep_cond)
                        y_uncond_indep = evaluate_expression(ex, env_indep_uncond)
                        
                        y_cond_indep = np.asarray(y_cond_indep, dtype=float)
                        y_uncond_indep = np.asarray(y_uncond_indep, dtype=float)
                        results_indep[nm] = y_cond_indep
                        results_indep_unconditional[nm] = y_uncond_indep
                    except Exception as e:
                        st.error(f"Error evaluating {nm} with independent samples: {e}")
                
                # Show summary table for independent results
                st.markdown("#### Summary Statistics (Without Correlation)")
                summary_rows_indep = []
                for k, v_cond in results_indep.items():
                    v_uncond = results_indep_unconditional.get(k, v_cond)
                    
                    if has_occurrence:
                        row_cond = {"result": f"{k} (conditional)", **summarize(v_cond)}
                        row_uncond = {"result": f"{k} (unconditional)", **summarize(v_uncond)}
                        summary_rows_indep.append(row_cond)
                        summary_rows_indep.append(row_uncond)
                    else:
                        row = {"result": k, **summarize(v_cond)}
                        summary_rows_indep.append(row)
                
                summary_df_indep = pd.DataFrame(summary_rows_indep)
                present_res = [c for c in order_cols if c in summary_df_indep.columns]
                summary_df_indep = summary_df_indep[["result"] + present_res]
                st.dataframe(summary_df_indep, use_container_width=True)
                
                # Show distribution plots for independent results
                st.markdown("#### Distribution Plots (Without Correlation)")
                for k in results_indep.keys():
                    v_cond = results_indep[k]
                    v_uncond = results_indep_unconditional.get(k, v_cond)
                    
                    if has_occurrence:
                        st.markdown(f"##### {k} - Distribution Plots")
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_cond = create_result_plot(v_cond, f"{k} (Conditional, No Correlation)", color=PALETTE[0])
                            st.plotly_chart(fig_cond, use_container_width=True)
                        with col2:
                            fig_uncond = create_result_plot(v_uncond, f"{k} (Unconditional, No Correlation)", color=PALETTE[1])
                            st.plotly_chart(fig_uncond, use_container_width=True)
                    else:
                        fig = create_result_plot(v_cond, f"{k} (No Correlation)", color=PALETTE[0])
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error generating independent samples: {e}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
    
    st.markdown("---")
    
    # Tornado Plot (Sensitivity Analysis)
    st.markdown("### Tornado Plot - Sensitivity Analysis")
    st.markdown("Select a result to analyze how input variables affect it. Analysis is available for both conditional and unconditional results.")
    
    # Check if we have occurrence probability (to determine if unconditional results are meaningful)
    has_occurrence = any(variables_config.get(sym, {}).get("prob", 1.0) < 1.0 for sym in variables_config.keys())
    
    # Select result for tornado analysis
    result_names = list(results.keys())
    if result_names:
        selected_result = st.selectbox("Select result for sensitivity analysis", result_names, key="tornado_result")
        
        if selected_result and selected_result in results:
            # Select analysis type (conditional vs unconditional)
            if has_occurrence and selected_result in results_unconditional:
                analysis_type = st.radio(
                    "Analysis type",
                    ["Conditional", "Unconditional", "Both"],
                    index=0,
                    key="tornado_analysis_type",
                    horizontal=True
                )
            else:
                analysis_type = "Conditional"  # Only conditional available if no occurrence probability
            
            # Function to calculate sensitivity for a given result and samples
            def calculate_sensitivity(result_values, input_samples, result_name_suffix=""):
                sensitivities = []
                for sym in input_samples.keys():
                    if sym in input_samples and len(input_samples[sym]) == len(result_values):
                        # Calculate correlation coefficient
                        corr = np.corrcoef(input_samples[sym], result_values)[0, 1]
                        if not np.isnan(corr):
                            # Calculate impact: difference when variable is at P10 vs P90
                            var_p10 = np.percentile(input_samples[sym], 90)  # High value
                            var_p90 = np.percentile(input_samples[sym], 10)  # Low value
                            
                            # Find indices where variable is near P10 and P90
                            var_sorted_idx = np.argsort(input_samples[sym])
                            n_samples = len(input_samples[sym])
                            p10_indices = var_sorted_idx[int(0.9 * n_samples):]
                            p90_indices = var_sorted_idx[:int(0.1 * n_samples)]
                            
                            result_at_p10 = np.mean(result_values[p10_indices])
                            result_at_p90 = np.mean(result_values[p90_indices])
                            impact = abs(result_at_p10 - result_at_p90)
                            
                            var_name = variables_config.get(sym, {}).get("name", sym)
                            sensitivities.append({
                                "variable": f"{var_name} ({sym})",
                                "correlation": corr,
                                "impact": impact,
                                "result_at_p10": result_at_p10,
                                "result_at_p90": result_at_p90
                            })
                
                # Sort by absolute impact in descending order (largest impact first)
                sensitivities.sort(key=lambda x: abs(x["impact"]), reverse=True)
                return sensitivities
            
            # Function to create tornado plot
            def create_tornado_plot(sensitivities, result_values, title_suffix=""):
                if not sensitivities:
                    return None
                
                base_value = float(np.mean(result_values))
                var_names = [s["variable"] for s in sensitivities]
                impacts_high = [s["result_at_p10"] - base_value for s in sensitivities]
                impacts_low = [s["result_at_p90"] - base_value for s in sensitivities]
                
                fig = go.Figure()
                
                # High impact bars (P10) - using light seaborn-like colors
                fig.add_trace(go.Bar(
                    y=var_names,
                    x=impacts_high,
                    name="P10 Impact",
                    orientation='h',
                    marker_color='#9ECAE1',  # Light seaborn blue
                    base=base_value
                ))
                
                # Low impact bars (P90)
                fig.add_trace(go.Bar(
                    y=var_names,
                    x=impacts_low,
                    name="P90 Impact",
                    orientation='h',
                    marker_color='#FDBB84',  # Light seaborn coral/orange
                    base=base_value
                ))
                
                # Add vertical line at base value
                fig.add_vline(
                    x=base_value,
                    line_dash="dash",
                    line_color="black",
                    annotation_text=f"Base: {base_value:,.3g}"
                )
                
                fig.update_layout(
                    title=f"Tornado Plot: {selected_result}{title_suffix}",
                    xaxis_title=f"Impact on {selected_result}",
                    yaxis_title="Input Variable",
                barmode='overlay',
                    height=max(400, len(var_names) * 30),
                    showlegend=True,
                    yaxis=dict(
                        autorange='reversed'  # Reverse y-axis so largest impact (first in list) appears at top
                    )
                )
                
                return fig
            
            # Calculate and display based on analysis type
            if analysis_type == "Conditional" or (analysis_type == "Both" and not has_occurrence):
                # Conditional analysis
                result_values_cond = results[selected_result]
                sensitivities_cond = calculate_sensitivity(result_values_cond, samples, " (Conditional)")
                
                if sensitivities_cond:
                    fig_cond = create_tornado_plot(sensitivities_cond, result_values_cond, " (Conditional)")
                    if fig_cond:
                        st.plotly_chart(fig_cond, use_container_width=True)
                    
                    # Show sensitivity table
                    st.markdown("**Conditional Sensitivity Table**")
                    sens_df_cond = pd.DataFrame([
                        {
                            "Variable": s["variable"],
                            "Correlation": f"{s['correlation']:.3f}",
                            "Impact (P10-P90)": f"{s['impact']:,.3g}",
                            "Result at P10": f"{s['result_at_p10']:,.3g}",
                            "Result at P90": f"{s['result_at_p90']:,.3g}"
                        }
                        for s in sensitivities_cond
                    ])
                    st.dataframe(sens_df_cond, use_container_width=True)
                else:
                    st.info("No sensitivity data available for conditional results.")
            
            elif analysis_type == "Unconditional":
                # Unconditional analysis
                if selected_result in results_unconditional and unconditional_samples:
                    result_values_uncond = results_unconditional[selected_result]
                    sensitivities_uncond = calculate_sensitivity(result_values_uncond, unconditional_samples, " (Unconditional)")
                    
                    if sensitivities_uncond:
                        fig_uncond = create_tornado_plot(sensitivities_uncond, result_values_uncond, " (Unconditional)")
                        if fig_uncond:
                            st.plotly_chart(fig_uncond, use_container_width=True)
                        
                        # Show sensitivity table
                        st.markdown("**Unconditional Sensitivity Table**")
                        sens_df_uncond = pd.DataFrame([
                            {
                                "Variable": s["variable"],
                                "Correlation": f"{s['correlation']:.3f}",
                                "Impact (P10-P90)": f"{s['impact']:,.3g}",
                                "Result at P10": f"{s['result_at_p10']:,.3g}",
                                "Result at P90": f"{s['result_at_p90']:,.3g}"
                            }
                            for s in sensitivities_uncond
                        ])
                        st.dataframe(sens_df_uncond, use_container_width=True)
                    else:
                        st.info("No sensitivity data available for unconditional results.")
                else:
                    st.warning("Unconditional results not available for this result.")
            
            elif analysis_type == "Both":
                # Both conditional and unconditional
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Conditional Analysis")
                    result_values_cond = results[selected_result]
                    sensitivities_cond = calculate_sensitivity(result_values_cond, samples, " (Conditional)")
                    
                    if sensitivities_cond:
                        fig_cond = create_tornado_plot(sensitivities_cond, result_values_cond, " (Conditional)")
                        if fig_cond:
                            st.plotly_chart(fig_cond, use_container_width=True)
                        
                        # Show sensitivity table
                        st.markdown("**Conditional Sensitivity Table**")
                        sens_df_cond = pd.DataFrame([
                            {
                                "Variable": s["variable"],
                                "Correlation": f"{s['correlation']:.3f}",
                                "Impact (P10-P90)": f"{s['impact']:,.3g}",
                                "Result at P10": f"{s['result_at_p10']:,.3g}",
                                "Result at P90": f"{s['result_at_p90']:,.3g}"
                            }
                            for s in sensitivities_cond
                        ])
                        st.dataframe(sens_df_cond, use_container_width=True)
                    else:
                        st.info("No sensitivity data available for conditional results.")
                
                with col2:
                    st.markdown("#### Unconditional Analysis")
                    if selected_result in results_unconditional and unconditional_samples:
                        result_values_uncond = results_unconditional[selected_result]
                        sensitivities_uncond = calculate_sensitivity(result_values_uncond, unconditional_samples, " (Unconditional)")
                        
                        if sensitivities_uncond:
                            fig_uncond = create_tornado_plot(sensitivities_uncond, result_values_uncond, " (Unconditional)")
                            if fig_uncond:
                                st.plotly_chart(fig_uncond, use_container_width=True)
                            
                            # Show sensitivity table
                            st.markdown("**Unconditional Sensitivity Table**")
                            sens_df_uncond = pd.DataFrame([
                                {
                                    "Variable": s["variable"],
                                    "Correlation": f"{s['correlation']:.3f}",
                                    "Impact (P10-P90)": f"{s['impact']:,.3g}",
                                    "Result at P10": f"{s['result_at_p10']:,.3g}",
                                    "Result at P90": f"{s['result_at_p90']:,.3g}"
                                }
                                for s in sensitivities_uncond
                            ])
                            st.dataframe(sens_df_uncond, use_container_width=True)
                        else:
                            st.info("No sensitivity data available for unconditional results.")
                    else:
                        st.warning("Unconditional results not available for this result.")

    st.markdown("---")
    
    # SimDec (Simulation Decomposition)
    st.markdown("### Simulation Decomposition (SimDec)")
    if not SIMDEC_AVAILABLE:
        st.info("SimDec analysis requires the `simdec` package. Install via `pip install simdec`.")
    else:
        st.caption(
            "SimDec partitions each input variable into equally probable states (ordered from low to high values) and compares their impact on the selected result. "
            "Note that a \"low\" state corresponds to the lower percentile (high probability of exceedance), while a \"high\" state corresponds to the upper percentile (low probability of exceedance)."
        )
        show_simdec = st.checkbox("Show SimDec Analysis", value=False, key="show_simdec")
        if show_simdec:
            label_strategy = st.selectbox(
                "State label style",
                (
                    "Percentile ranges",
                    "Low / Medium / High",
                    "Numeric value ranges",
                ),
                index=1,
                help="Select how the scenario labels should be named in the plots.",
                key="simdec_label_strategy"
            )
            palette_choice = st.selectbox(
                "Color palette",
                SIMDEC_QUALITATIVE_CMAPS,
                index=0,
                help="Choose a qualitative matplotlib/seaborn palette for SimDec scenarios.",
                key="simdec_palette"
            )
            for result_name, result_values in results.items():
                st.markdown(f"#### {result_name}")
                simdec_columns: Dict[str, Any] = {}
                seen_names: Dict[str, int] = {}

                # CRITICAL: Use current variables_config from session_state to ensure it matches samples
                # Rebuild variables_config from session_state to get latest values
                current_variables_config = st.session_state.get("variables_config", {})
                if not current_variables_config:
                    # Fallback: try to rebuild from session_state widget values
                    var_symbols_simdec = st.session_state.get("var_symbols", list(samples.keys()))
                    current_variables_config = {}
                    for sym in var_symbols_simdec:
                        if sym in samples:  # Only include variables that have samples
                            name_key = f"name_{sym}"
                            name_from_state = st.session_state.get(name_key, "").strip()
                            if not name_from_state:
                                name = sym
                            else:
                                name = name_from_state
                            current_variables_config[sym] = {"name": name}

                for sym, values in samples.items():
                    # Use current variables_config to get the correct name
                    base_name = current_variables_config.get(sym, {}).get("name", sym) or sym
                    col_name = base_name
                    if col_name in seen_names:
                        seen_names[col_name] += 1
                        col_name = f"{base_name} ({seen_names[col_name]})"
                    else:
                        seen_names[col_name] = 1
                    simdec_columns[col_name] = values

                result_column = result_name
                if result_column in simdec_columns:
                    suffix = 1
                    while f"{result_column} ({suffix})" in simdec_columns:
                        suffix += 1
                    result_column = f"{result_name} ({suffix})"
                simdec_columns[result_column] = result_values

                simdec_df = pd.DataFrame(simdec_columns)
                if simdec_df.empty or result_column not in simdec_df.columns:
                    st.warning(
                        f"SimDec requires the calculated result '{result_name}' and at least one input variable."
                    )
                    continue

                input_cols = [col for col in simdec_df.columns if col != result_column]
                if not input_cols:
                    st.warning(
                        f"No input variables available for SimDec analysis of '{result_name}'."
                    )
                    continue

                state_choice = st.selectbox(
                    f"States per input for {result_name}",
                    (
                        "Automatic (SimDec default)",
                        "Two states (low/high)",
                        "Three states (low/medium/high)",
                    ),
                    help=(
                        "Choose how many equally probable bins each variable is split into before the decomposition."
                    ),
                    key=f"states_choice_{result_name}"
                )
                states_override: Optional[List[int]] = None
                if state_choice.startswith("Two"):
                    states_override = [2] * len(input_cols)
                elif state_choice.startswith("Three"):
                    states_override = [3] * len(input_cols)

                try:
                    decomposition_result = decompose(
                        simdec_df,
                        inputs=input_cols,
                        output=result_column,
                        bins=10,
                        states=states_override,
                    )
                    decomposition_result.label_strategy = label_strategy  # type: ignore[attr-defined]
                    decomposition_result.palette_name = palette_choice  # type: ignore[attr-defined]
                except Exception as exc:
                    st.error(f"SimDec decomposition failed for {result_name}: {type(exc).__name__}: {exc}")
                else:
                    st.markdown("**Decomposition Bins**")
                    fig_bins = plot_bins(decomposition_result)
                    palette_colors = _get_simdec_palette(
                        decomposition_result.states, palette_choice
                    )
                    if palette_colors is not None:
                        _apply_palette_to_figure(fig_bins, palette_colors)
                    _style_simdec_plot(fig_bins)
                    _add_simdec_legend(decomposition_result, fig_bins)
                    _display_simdec_plot(fig_bins)

                    st.markdown("**Box Plot**")
                    fig_box = plot_box(decomposition_result)
                    palette_colors = _get_simdec_palette(
                        decomposition_result.states, palette_choice
                    )
                    if palette_colors is not None:
                        _apply_palette_to_figure(fig_box, palette_colors)
                    _style_simdec_plot(fig_box)
                    _add_simdec_legend(decomposition_result, fig_box)
                    _display_simdec_plot(fig_box)

                    legend_df = _build_simdec_table(decomposition_result, palette_colors)
                    if legend_df is not None:
                        st.markdown("**Scenario Table**")
                        st.dataframe(legend_df, use_container_width=True)
    
    st.markdown("---")
    
    # Export functionality
    st.markdown("### Export Results")
    
    # Prepare data for export
    trial_df = pd.DataFrame({**samples, **results})
    trial_df.insert(0, 'Trial', range(1, len(trial_df) + 1))
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV",
            data=trial_df.to_csv(index=False).encode("utf-8"),
            file_name="probcalcmc_results.csv",
            mime="text/csv"
        )
    
    with col2:
        if OPENPYXL_AVAILABLE:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Sheet 1: Trial samples and results
                trial_df.to_excel(writer, sheet_name='Trial Samples', index=False)
                
                # Sheet 2: Summary statistics
                summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
            
            st.download_button(
                "Download Excel",
                data=output.getvalue(),
                file_name="probcalcmc_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("Excel export requires openpyxl. Install with: pip install openpyxl")

# --- Build samples for each variable ---
# CRITICAL: Cache key must include parameter values to invalidate when params change
@st.cache_data(show_spinner=False)
def simulate_variables(config: Dict[str, Dict[str, Any]], n: int, seed_local: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Returns (conditional_samples, unconditional_samples) tuples
    
    IMPORTANT: This function is cached. The cache key includes:
    - config (which includes params, type, prob)
    - n (number of samples)
    - seed_local (random seed)
    
    When any parameter value changes, the config dict changes, which invalidates the cache.
    """
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

# Non-cached version for real-time preview plots
# This ensures plots update immediately when parameters change
def _simulate_variables_uncached(config: Dict[str, Dict[str, Any]], n: int, seed_local: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Non-cached version of simulate_variables for real-time preview plots.
    This ensures the distribution plot updates immediately when parameters change.
    """
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

# --- Page Routing ---
if page == "Start":
    render_start_page()
elif page == "Input Variables":
    render_input_variables_page()
elif page == "Dependency":
    render_dependency_page()
elif page == "Formula Definition":
    render_formula_definition_page()
elif page == "Results":
    render_results_page()

