# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Modern Portfolio Optimization Engine

A Python package for institutional-grade asset allocation using Markowitz and Black-Litterman models, with 10-year backtesting and PDF reporting.

## Quick Start

```bash
source venv/bin/activate
pip install -r requirements.txt
python -m src.run  # Full pipeline end-to-end
```

## Project Structure

```
src/
  __init__.py
  data.py           # yfinance + FRED API data fetching, caching
  optimization.py   # Classic Markowitz efficient frontier
  black_litterman.py # Market-implied returns, subjective views, BL blending
  backtest.py       # Static allocation backtest vs 60/40 benchmark
  report.py         # PDF memo generation (reportlab)
  run.py            # Main entry point orchestrating all phases
data/               # Cached raw data (parquet/csv) — DO NOT commit
results/            # Output PDFs and intermediate plots — DO NOT commit
```

## Key Design Decisions

1. **Universe**: 6-8 ETF proxies (SPY, TLT, GLD, VNQ, EFA, + 1-2 others like AGG, EEM).
2. **Covariance**: Ledoit-Wolf shrinkage (via PyPortfolioOpt or sklearn) to stabilize small-sample estimates.
3. **Black-Litterman prior**: Market-cap-weighted reverse optimization (simplified: equal-weight on ETFs as a reasonable proxy).
4. **Constraints**: No shorting (weights ≥ 0), max 40% per asset, weights sum to 1.
5. **Backtest**: Static allocation annually rebalanced over 10 years vs 60/40 (SPY/TLT) benchmark.
6. **Data caching**: Raw prices stored as CSV in `/data` to avoid repeated API calls.

## Critical Implementation Notes

- **FRED API key**: Read from `.env` (`FRED_API_KEY`) at startup.
- **Subjective views** in `black_litterman.py`: Define as top-level list of dicts (ticker pairs + alpha %) for easy editing.
- **Risk-free rate**: Pulled from FRED (3-month T-bill, DGS3MO) and annualized.
- **Plots**: Use matplotlib with consistent styling; save as PNG intermediate, embed in PDF via reportlab.
- **PDF output**: Single-page memo; if content overflows, summarize or use landscape orientation.

## Common Commands

```bash
# Run full pipeline
python -m src.run

# Run only data fetch + caching
python -c "from src.data import fetch_all_data; fetch_all_data()"

# Inspect cached data
python -c "import pandas as pd; print(pd.read_csv('data/prices.csv', index_col=0).head())"

# Quick covariance/correlation table
python -c "from src.data import load_cached_data; from src.optimization import compute_stats; stats = compute_stats(*load_cached_data()); print(stats)"
```

## Testing & Validation

After each phase, run the corresponding stage:
- Phase 1: `python -c "from src.data import fetch_all_data; fetch_all_data()"`
- Phase 2: `python -c "from src.optimization import plot_efficient_frontier; plot_efficient_frontier()"`
- Phase 3: `python -c "from src.black_litterman import bl_posterior_returns; print(bl_posterior_returns())"`
- Phase 4: `python -c "from src.optimization import constrained_optimization; constrained_optimization()"`
- Phase 5: `python -c "from src.backtest import backtest_allocation; backtest_allocation()"`
- Phase 6: `python -m src.run`

## Debugging

- If FRED API fails, check `.env` and network; data.py will cache fallback to yfinance-only stats.
- If optimization diverges, inspect the covariance matrix rank and correlation structure (Ledoit-Wolf should help).
- If plots don't render, ensure matplotlib backend is set (mpl.use('Agg') in headless environments).
