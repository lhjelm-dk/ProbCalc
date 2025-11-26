# ProbCalcMC – Custom Monte Carlo simulation

*by Lars Hjelm*

**ProbCalcMC v0.85** is a Streamlit app for probabilistic modeling and Monte Carlo simulations.

## Features

### Input Variables Page
- Up to **256 variables** (a, b, c, …, aa, ab, …), each with:
  - Optional name, occurrence probability, and one of 20+ built-in distributions
  - Includes **StretchBeta (min–mode–max)** and truncated variants
  - **Persistent parameter values** - your changes are preserved across page navigation
  - **Update Distribution** button for each variable to refresh plots and statistics
  - Real-time distribution preview with histogram and CDF plots
  - Conditional and unconditional distribution statistics

### Dependency Matrix Page
- **Correlation matrix** definition with interactive sliders
- Define correlations between variables (-0.99 to +0.99)
- **Correlation cross plots** showing relationships before and after correlation
- Automatic matrix validation and Higham projection for positive semi-definite matrices
- Cholesky decomposition for generating correlated samples

### Formula Definition Page
- **Formula engine** with chaining:
  - Reference earlier results as `f1`, `f2`, … or by name using `res_<slug>`
  - Example: result name *Net Profit (EUR)* → variable `res_net_profit_eur`
- Multiple formulas with automatic evaluation
- LaTeX formula rendering for mathematical expressions
- Automatic simulation when formulas are defined

### Results Page
- **Results Summary Statistics** with conditional and unconditional results
- **Result Distributions** with interactive plots (histogram + exceedance CDF)
- **Tornado Plot** sensitivity analysis (conditional and unconditional)
- **Simulation Decomposition (SimDec)** analysis:
  - Decomposition bins and box plots
  - Scenario tables
  - Support for both conditional and unconditional analysis
- **Sample Size Convergence Analysis** to determine stable sample sizes
- **Correlation Impact Analysis** comparing results with and without correlations
- Full **CSV/Excel exports**:
  - Trial Samples (conditional & unconditional)
  - Summary Statistics
  - Formulae
  - Derived Map

### Additional Features
- **Percentile convention toggle** in sidebar - choose between standard (P10=low, P90=high) or exceedance convention (P10=high, P90=low)
- **Debug mode** for troubleshooting session state issues
- Support for **occurrence probability** (< 1.0) for conditional events
- Automatic LaTeX rendering of mathematical formulas

## Installation

### Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Distributions

Normal, Lognormal, Uniform, Triangular, PERT, Beta, Gamma, Weibull, Exponential, Poisson, Binomial, Bernoulli, Laplace, Student-t, Cauchy, Pareto, Erlang, TruncNormal, TruncLognormal, and StretchBeta (min, mode, max, λ).

## Formula Syntax

Use `+ - * / ** ()` and functions `abs, sqrt, exp, log, log10, log2, min, max, where, clip, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, floor, ceil, round, sign, pow`.

Constants: `pi, e, inf, nan`.

### Examples

```
Formula 1: profit = revenue - cost
Formula 2: margin = f1 / max(revenue, 1e-9)
Formula 3: kpi    = res_profit / res_margin
```

## Recent Updates (v0.85)

- Fixed session state management - parameter values now persist across page navigation
- Added explicit "Update Distribution" button for each variable
- Moved percentile convention toggle to sidebar
- Fixed distribution statistics calculation (P10/P90 percentiles)
- Improved state management to prevent values from resetting to defaults
- Enhanced debug mode for troubleshooting session state issues
- Added correlation matrix with cross plots
- Implemented Simulation Decomposition (SimDec) analysis
- Added sample size convergence analysis
- Support for conditional and unconditional results throughout
- Improved LaTeX formula rendering

## Deploy on Streamlit Cloud

1. Push the folder to GitHub.
2. Go to https://share.streamlit.io and create a new app.
3. Select the repo, set **Main file path** to `app.py`, click **Deploy**.

## License

MIT (see LICENSE)
