# Interactive Portfolio Optimization Web App

A production-ready Streamlit web app that turns your portfolio optimization engine into an interactive tool.

## Features

### 📊 Tab 1: Data Overview
- View asset statistics (returns, volatility, Sharpe ratios)
- Inspect correlation matrix with heatmap
- Plot normalized historical prices

### 🎯 Tab 2: Markowitz Efficient Frontier
- Generate the efficient frontier dynamically
- Adjust frontier resolution (20-200 points)
- Visualize random portfolio cloud shaded by Sharpe ratio
- Display Capital Allocation Line (CAL)
- Highlight max-Sharpe portfolio
- View allocation pie chart and weights table

### 🔮 Tab 3: Black-Litterman Model
- Define custom subjective views (0-5 views)
- Specify outperformer, underperformer, alpha, and confidence for each view
- Compare market-implied (prior) vs BL-adjusted (posterior) returns
- Constrained re-optimization with adjustable constraints:
  - Max weight per asset (5%-100%)
  - Min weight per asset (0%-20%)
- View optimized allocation with pie chart and metrics

### 📈 Tab 4: Backtesting & Strategy Comparison
- Test custom allocations against the 60/40 benchmark
- Three portfolio options:
  - Max Sharpe (Markowitz) unconstrained
  - BL-Constrained (from Tab 3)
  - Custom weights (drag sliders)
- Performance metrics: CAGR, volatility, Sharpe, max drawdown
- Cumulative return comparison chart

### 💾 Tab 5: Export & Download
- Download price history (CSV)
- Download correlation matrix (CSV)
- Download asset statistics (CSV)
- Copy results summary to clipboard

## Running the App

### Prerequisites
```bash
# Install dependencies (if not already installed)
pip install streamlit plotly -q
```

### Launch the App
```bash
# From the project root directory
streamlit run app.py
```

This will:
1. Open the app in your default browser at `http://localhost:8501`
2. Load cached data automatically (no API calls needed if cache exists)
3. Display a sidebar with data loading controls

### Data Caching
The app uses cached data from `data/` directory (prices.csv, risk_free_rate.csv).
- Click "🔄 Load/Refresh Data" in the sidebar to fetch fresh data
- Fetch takes ~30 seconds (10 years of daily prices for 7 ETFs + risk-free rates)
- Cached data is used automatically for subsequent sessions

## Key Workflows

### Example 1: Analyze Markowitz Frontier
1. Click "Load/Refresh Data" in sidebar
2. Go to **Markowitz** tab
3. Adjust frontier resolution slider (default 100)
4. Toggle "Show Random Portfolios" to visualize distribution
5. View max-Sharpe allocation and metrics

### Example 2: Build BL Portfolio with Custom Views
1. Ensure data is loaded
2. Go to **Black-Litterman** tab
3. Set "Number of views" to 2-3
4. Define views:
   - View 1: "EEM outperforms SPY by 2%" with 50% confidence
   - View 2: "GLD outperforms AGG by 1%" with 40% confidence
5. Adjust max weight constraint (default 40%)
6. View posterior returns comparison and allocation

### Example 3: Backtest Your Strategy
1. Define allocation in BL tab (or use preset)
2. Go to **Backtesting** tab
3. Select portfolio type:
   - "Max Sharpe" → unconstrained Markowitz
   - "Constrained" → BL-optimized with constraints
   - "Custom" → drag sliders to set weights
4. View metrics and cumulative return chart
5. Compare against 60/40 benchmark

## Configuration & Customization

### Change Asset Universe
Edit `src/data.py`:
```python
TICKERS = ["SPY", "TLT", "GLD", "VNQ", "EFA", "EEM", "AGG"]
```

### Change Max Weight Constraint
In **Black-Litterman** tab, use slider (5%-100%)
Or edit `src/optimization_constrained.py` line 16:
```python
def constrained_optimal_portfolio(..., max_weight=0.40):
```

### Change Rebalancing Frequency
In **Backtesting** tab, currently set to annual ("YE").
To change, modify `app.py` line ~620:
```python
backtest_allocation(..., rebalance_freq="YE")  # "YE" = annual, "MS" = monthly, "D" = daily
```

### Adjust BL Parameters
In `src/black_litterman.py`:
- `tau = 0.05` → view uncertainty scaling (lower = less weight to views)
- `omega_diag` → view-specific uncertainty

## Architecture

```
app.py
├── Session State Management
│   └── Caches prices, rf_data, stats, cov_matrix
├── Sidebar
│   └── Data loading & refresh control
└── Tabs
    ├── Tab 1: Data Overview
    │   └── Uses: compute_stats, correlation matrix
    ├── Tab 2: Markowitz
    │   └── Uses: efficient_frontier, compute_sharpe_ratio
    ├── Tab 3: Black-Litterman
    │   └── Uses: market_implied_returns, black_litterman_posterior, constrained_optimal_portfolio
    ├── Tab 4: Backtesting
    │   └── Uses: backtest_allocation
    └── Tab 5: Export
        └── Data download utilities
```

## Performance & Tips

- **First load**: ~30 seconds (fetches 10 years of data)
- **Subsequent loads**: <1 second (uses cache)
- **Efficient frontier**: ~5 seconds (100 points)
- **BL optimization**: ~2 seconds (with views)
- **Backtest**: ~5 seconds (10 years of daily rebalancing)

**Tip**: Use sliders (e.g., max weight) instead of entering text to avoid re-computation delays.

## Troubleshooting

### App won't start
```bash
# Check Python version (3.8+)
python --version

# Verify dependencies
pip list | grep streamlit
pip list | grep plotly
```

### Data won't load
- Check `data/prices.csv` exists
- Check `data/risk_free_rate.csv` exists
- Check `.env` has `FRED_API_KEY` (optional, uses 4% fallback)

### Optimization fails
- Check console output for error messages
- Ensure constraints are feasible (e.g., max_weight ≥ min_weight)
- Try reducing number of views in BL tab

### Performance is slow
- Close other browser tabs
- Use smaller frontier resolution
- Reduce number of views in BL model
- Clear Streamlit cache: `streamlit cache clear`

## Notes

- All optimizations use SLSQP (Sequential Least Squares Programming)
- Covariance uses Ledoit-Wolf shrinkage for stability
- No shorting constraints by default
- Risk-free rate averaged over 10-year period
- Backtest uses annual rebalancing (can be changed)

---

**Built with**: Streamlit, Plotly, SciPy, NumPy, Pandas  
**Data sources**: yfinance (prices), FRED API (risk-free rate)
