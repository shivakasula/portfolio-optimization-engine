# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Modern Portfolio Optimization Engine

**What it is:** Institutional-grade asset allocation system that combines Markowitz portfolio theory with the Black-Litterman model to generate actionable investment recommendations. Outputs a professional PDF investment memo with performance analysis, efficient frontier visualizations, and constrained portfolio allocations.

**What it does:** Ingests 10 years of daily ETF prices, applies modern portfolio optimization theory, lets users inject subjective market views, optimizes under real-world constraints (no shorting, max 40% per asset), backtests the resulting allocation, and generates a publication-ready PDF memo.

**Current status (Aug 2026):** All 6 phases complete, tested, pushed to GitHub. System is production-ready.

## What Was Built (6 Phases)

### Phase 1: Data & Statistics
- Fetches 10 years of daily adjusted close prices for 7 ETF proxies via yfinance
- Pulls risk-free rate from FRED (3-month T-bill, DGS3MO) with fallback to 4% if API unavailable
- Computes annualized returns, volatility, and **Ledoit-Wolf shrinkage covariance** (addresses small-sample bias)
- Caches data locally as CSV to avoid repeated API calls
- **Output**: Correlation matrix and Sharpe ratios for all assets

### Phase 2: Markowitz Efficient Frontier
- Builds efficient frontier (100 points) by minimizing volatility for each target return
- Identifies **max-Sharpe portfolio** (56% SPY, 44% GLD; Sharpe 0.66)
- Identifies **minimum-volatility portfolio** (92% AGG, 6% SPY, 2% GLD)
- Visualizes with random portfolio cloud shaded by Sharpe ratio + Capital Allocation Line
- **Output**: `results/efficient_frontier.png`

### Phase 3: Black-Litterman Model
- Derives **market-implied equilibrium returns** via reverse optimization (equal-weight prior on ETFs)
- Implements 2–3 user-defined **subjective views** (e.g., "EEM outperforms SPY by 2%")
- Blends prior + views using Bayesian framework with tau parameter (view uncertainty scaling)
- Produces posterior expected returns reflecting both market consensus and subjective insights
- **Output**: `results/bl_comparison.png` (prior vs posterior return comparison)

### Phase 4: Constrained Re-optimization
- Maximizes Sharpe ratio subject to:
  - No shorting (weights ≥ 0)
  - Max 40% per asset (enforced diversification)
  - Weights sum to 1
- Uses SLSQP optimization (scipy)
- **Output**: `results/allocations_comparison.png` (Markowitz vs BL-constrained pie charts)

### Phase 5: Backtesting
- Static allocation strategy, annually rebalanced over 10 years (2016–2026)
- Compares BL-constrained allocation vs 60/40 (SPY/TLT) benchmark
- Computes CAGR, annualized volatility, Sharpe ratio, maximum drawdown
- **Results**:
  - BL-Constrained: 8.49% CAGR, 0.571 Sharpe, -34.9% max drawdown
  - 60/40 Benchmark: 8.69% CAGR, 0.597 Sharpe, -27.2% max drawdown
- **Output**: `results/backtest_comparison.png` (cumulative return curves)

### Phase 6: Institutional Memo
- Generates single-page PDF with reportlab
- Includes methodology summary, asset universe, views, all 4 plots, performance table
- Professional styling and layout
- **Output**: `results/portfolio_memo.pdf` (main deliverable)

## Key Design Decisions & Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| **Asset Universe** | 7 ETFs (SPY, TLT, GLD, VNQ, EFA, EEM, AGG) | Covers equities (US, intl, emerging), bonds (long/agg), commodities, real estate; liquid and diversified |
| **Covariance** | Ledoit-Wolf shrinkage | Stabilizes small-sample estimates; better than raw sample covariance for optimization |
| **BL Prior** | Equal-weight (not market-cap) | Simplified proxy since ETFs, not individual stocks; market-cap weighting over-complicates; equal-weight is reasonable default |
| **BL Blending** | Tau = 0.05, Omega diagonal | Tau scales view uncertainty; Omega reflects confidence levels; tau=0.05 is standard practice |
| **Constraints** | Max 40% per asset, no shorting | Realistic for retail/institutional portfolios; prevents over-concentration; no shorting avoids regulatory/borrowing issues |
| **Rebalancing** | Annual | Balances turnover/transaction costs vs drift; monthly/quarterly would be overkill |
| **Benchmark** | 60/40 (SPY/TLT) | Industry standard; simple to interpret; provides defensible comparison |
| **Data Caching** | CSV (not parquet) | Avoids pyarrow dependency; CSV is human-readable; adequate for this scale |

## Critical Implementation Notes

### Subjective Views (Customization Point)
Located in `src/black_litterman.py`:
```python
VIEWS = [
    {"outperformer": "EEM", "underperformer": "SPY", "alpha": 0.02},  # 2% outperformance
    {"outperformer": "GLD", "underperformer": "AGG", "alpha": 0.01},  # 1% outperformance
]
VIEW_CONFIDENCES = [0.5, 0.4]  # 50%, 40% confidence
```
- Each view compares two assets with an expected alpha (outperformance %)
- Confidence in [0, 1]; higher = stronger belief
- System is **view-sensitive**: wrong views → poor allocations (see "Known Issues" below)

### Risk-Free Rate
- Fetches 3-month T-bill (DGS3MO) from FRED API
- Falls back to 4% if API unavailable or `.env` lacks `FRED_API_KEY`
- Used for Sharpe ratio computation and capital allocation line

### Optimization Details
- **Efficient frontier**: scipy.optimize.minimize with SLSQP (Sequential Least Squares Programming)
- **Max-Sharpe**: Minimizes negative Sharpe ratio
- **Constrained re-optimization**: SLSQP with box bounds on weights
- All optimizations are deterministic (no random seed issues)

## Known Issues & Limitations

### 1. View Sensitivity (Critical)
**Issue**: BL-constrained portfolio underperformed 60/40 benchmark in backtest.

**Root cause**: The view "EEM outperforms SPY by 2%" was **wrong for 2016–2026**. SPY actually returned 15.92% (2.0x the average) while EEM returned 10.22%. The model correctly downweighted SPY based on the view, resulting in a portfolio with 0% SPY and 80% lower-Sharpe international equities. Expected return was 9.75% (higher than 60/40's 9.07%) but realized return was lower due to incorrect view.

**Expected behavior**: This is correct Black-Litterman behavior. When views are wrong, outcomes suffer. The model did exactly what it was supposed to do.

**Fix**: 
1. Remove the EEM view (it's wrong historically)
2. Reduce confidence in the view to 0.2 (you're uncertain)
3. Increase max-weight constraint to 50%+ to allow SPY allocation even if EEM view persists

### 2. No Allocation to Defensive Assets
The BL-constrained portfolio allocated 0% to TLT (Treasuries), which have -0.137 correlation with SPY and provided downside protection in the 60/40 benchmark. The max-40% constraint + international tilt led to an 100% equity-ish portfolio → higher drawdown.

**Fix**: Either add a view favoring bonds or relax constraints.

### 3. FRED API Dependency
Pulling risk-free rate from FRED requires network access and a free API key. If unavailable, system defaults to 4% (reasonable but may be stale).

**Mitigation**: Cache the risk-free rate after first fetch (done in `data/risk_free_rate.csv`).

## Performance Summary

| Metric | BL-Constrained | 60/40 Bench | Notes |
|--------|---|---|---|
| Expected Annual Return | 9.75% | 9.07% | BL higher on paper (better views would confirm) |
| Expected Volatility | 17.21% | 11.58% | BL riskier (no bonds) |
| Expected Sharpe | 0.334 | 0.438 | 60/40 more efficient (bonds help) |
| **Backtest CAGR (2016–2026)** | 8.49% | 8.69% | 60/40 won; BL views were wrong |
| Backtest Sharpe | 0.571 | 0.597 | 60/40 still better |
| Max Drawdown | -34.9% | -27.2% | BL suffered more (no TLT buffer) |

## How to Customize & Extend

### Change Subjective Views
Edit `src/black_litterman.py`, re-run `python -m src.run`. Pipeline regenerates automatically.

### Adjust Constraints
- Max-weight per asset: `src/optimization_constrained.py`, line 22 (`max_weight=0.40`)
- Rebalancing frequency: `src/backtest.py`, line 54 (`rebalance_freq="YE"`)
- Minimum allocation (if you want 10% minimum in each asset): Add lower bound to weights

### Add More Assets
1. Update `TICKERS` list in `src/data.py`
2. Re-run `python -m src.run` (data will re-fetch)
3. Adjust views in `src/black_litterman.py` if needed

### Change Risk-Free Rate
Update `FRED_API_KEY` in `.env` or hardcode fallback in `src/data.py` line 118.

## Common Commands

```bash
# Full pipeline end-to-end
python -m src.run

# Data only
python -c "from src.data import fetch_all_data; fetch_all_data()"

# Inspect cached data
python -c "import pandas as pd; print(pd.read_csv('data/prices.csv', index_col=0).head(10))"

# Test BL model
python -m src.black_litterman

# Test backtest
python -m src.backtest

# View sample stats
python -c "from src.data import load_cached_data, compute_stats; prices, rf = load_cached_data(); annual_returns, cov, corr, rf_rate = compute_stats(prices, rf); print(annual_returns)"
```

## Testing & Validation

All phases run without errors end-to-end. See commit history for phase-by-phase testing.

```bash
python -m src.run  # Should complete in ~30–60 seconds; generates PDF
```

## Debugging

- **FRED API fails**: Check `.env` has `FRED_API_KEY`; data.py falls back to 4%
- **PDF won't generate**: Ensure reportlab is installed; check matplotlib backend
- **Optimization diverges**: Inspect covariance matrix condition number; Ledoit-Wolf should help
- **Cache stale**: Delete `data/` and re-run to re-fetch
- **View didn't change allocation**: Increase `VIEW_CONFIDENCES` or adjust `tau` parameter in `src/black_litterman.py`
