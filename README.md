# Modern Portfolio Optimization Engine

An institutional-grade Python package for asset allocation using Markowitz and Black-Litterman models, built on 10 years of ETF price data with backtesting and PDF reporting.

## Features

- **Classic Markowitz**: Efficient frontier with max-Sharpe and minimum-volatility portfolios
- **Black-Litterman Model**: Market-implied returns blended with subjective views
- **Constrained Optimization**: No-short, diversification constraints (max 40% per asset)
- **Backtesting**: Static allocation vs. 60/40 benchmark over 10 years
- **Institutional Memo**: Single-page PDF summary with methodology, plots, and stats

## Data Universe

6-8 ETF proxies:
- **Equities**: SPY (US large-cap), EFA (International), EEM (Emerging Markets)
- **Fixed Income**: TLT (Long-term Treasury), AGG (Aggregate bonds)
- **Alternatives**: GLD (Gold), VNQ (Real Estate)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your FRED API key:
```
FRED_API_KEY=your_fred_api_key_here
```

(Get a free key at https://fredaccount.stlouisfed.org/apikeys)

## Usage

Run the full pipeline:
```bash
python -m src.run
```

This generates:
- `/data/prices.parquet` — 10-year daily prices (cached)
- `/data/risk_free_rate.csv` — Risk-free rates from FRED
- `/results/portfolio_memo.pdf` — Investment memo with all analysis

## Project Structure

- `src/data.py` — Data fetching and caching
- `src/optimization.py` — Markowitz efficient frontier
- `src/black_litterman.py` — BL model and posterior returns
- `src/backtest.py` — Backtesting and performance metrics
- `src/report.py` — PDF report generation
- `src/run.py` — Main orchestration

## Customization

Edit subjective views in `src/black_litterman.py`:
```python
VIEWS = [
    {"outperformer": "EEM", "underperformer": "SPY", "alpha": 0.02},
    # Add/modify as needed
]
```

## Output

The PDF memo includes:
- Methodology summary
- Efficient frontier plot with random portfolio cloud
- Comparison: Black-Litterman vs. market-implied returns
- Allocation pie charts (Markowitz vs. BL-constrained)
- Backtest performance table and cumulative return curves
