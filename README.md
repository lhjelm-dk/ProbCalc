# ProbCalcMC – Custom Monte Carlo simulation

*by Lars Hjelm*

**ProbCalcMC** is a Streamlit app for probabilistic modeling and Monte Carlo simulations.

### Features
- Up to **256 variables** (a, b, c, …, aa, ab, …), each with:
  - Optional name, occurrence probability, and one of 20+ built-in distributions.
  - Includes **StretchBeta (min–mode–max)** and truncated variants.
- **Formula engine** with chaining:
  - Reference earlier results as `f1`, `f2`, … or by name using `res_<slug>`.
  - Example: result name *Net Profit (EUR)* → variable `res_net_profit_eur`.
- Interactive **histogram + exceedance (CDF)** plots, summary tables, and full **CSV/Excel exports**.
- Excel output includes: Trial Samples (conditional & unconditional), Summaries, Formulae, and Derived Map.

### Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

### Distributions (examples)

Normal, Lognormal, Uniform, Triangular, PERT, Beta, Gamma, Weibull, Exponential, Poisson, Binomial, Bernoulli, Laplace, Student-t, Cauchy, Pareto, Erlang, TruncNormal, TruncLognormal, and StretchBeta (min, mode, max, λ).

### Formula syntax

Use `+ - * / ** ()` and functions `abs, sqrt, exp, log, log10, min, max, where, clip, sin, cos, tan, floor, ceil, round`.
Constants: `pi, e, inf, nan`.

Example:

```
Formula 1: profit = revenue - cost
Formula 2: margin = f1 / max(revenue, 1e-9)
Formula 3: kpi    = res_profit / res_margin
```


### License

MIT (see LICENSE)

