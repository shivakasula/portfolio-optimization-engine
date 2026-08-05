# Modern Portfolio Optimization Engine

**Institutional-grade asset allocation system combining Markowitz and Black-Litterman models with 10-year backtesting.**

**📊 [View Sample Investment Memo](results/portfolio_memo.pdf)** ← Full analysis with charts, performance metrics, and allocations

## What This Does

Generates professional portfolio allocation recommendations by:
1. **Learning from market data** — Ingests 10 years of daily prices (7 ETF asset classes)
2. **Applying institutional theory** — Markowitz efficient frontier + Black-Litterman model for blending market expectations with subjective views
3. **Stress-testing strategies** — Backtests static allocations over the full 10-year period
4. **Producing investment memos** — Generates publication-ready PDF with methodology, visualizations, and performance tables

## Key Numbers

**Sample Portfolio (BL-Constrained):**
- Expected Return: **9.75%** annually
- Expected Volatility: **17.21%**
- Sharpe Ratio: **0.334**
- Max Drawdown: **-34.9%** (10-year backtest)

**Benchmark (60% SPY / 40% TLT):**
- Backtest CAGR: **8.69%**
- Sharpe Ratio: **0.597**
- Max Drawdown: **-27.2%**

Asset universe includes SPY, TLT, GLD, VNQ, EFA, EEM, AGG — covering equities (US, international, emerging), fixed income, alternatives, and real estate.

## Code Quality

- **Modular architecture** — Separate modules for data ingestion, optimization, Black-Litterman, backtesting, and reporting
- **Reproducible** — All data cached; re-run with one command: `python -m src.run`
- **Editable views** — Change market expectations in one place; entire pipeline re-executes automatically
- **Production-grade** — Ledoit-Wolf shrinkage covariance, CVXPY constrained optimization, reportlab PDF generation

## For Developers

See [CLAUDE.md](CLAUDE.md) for architecture details, common commands, and how to extend the system.

To set up locally:
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m src.run  # Generates PDF memo in results/portfolio_memo.pdf
```
