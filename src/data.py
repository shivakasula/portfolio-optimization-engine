"""Data fetching, caching, and statistics computation."""

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from dotenv import load_dotenv
from sklearn.covariance import LedoitWolf

load_dotenv()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ETF universe: 6-8 proxies for diversified allocation
TICKERS = ["SPY", "TLT", "GLD", "VNQ", "EFA", "EEM", "AGG"]

PRICES_CACHE = os.path.join(DATA_DIR, "prices.csv")
RF_CACHE = os.path.join(DATA_DIR, "risk_free_rate.csv")


def fetch_prices(start_date=None, end_date=None, use_cache=True):
    """Fetch 10 years of daily adjusted close prices via yfinance."""
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365 * 10)

    if use_cache and os.path.exists(PRICES_CACHE):
        print(f"Loading prices from cache: {PRICES_CACHE}")
        prices = pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True)
        return prices

    print(f"Fetching {len(TICKERS)} tickers from {start_date.date()} to {end_date.date()}...")
    data = yf.download(TICKERS, start=start_date, end=end_date, progress=False)

    # Extract Close prices (yfinance returns MultiIndex: ['Price', 'Ticker'])
    if isinstance(data.columns, pd.MultiIndex):
        # Try 'Adj Close' first, fall back to 'Close'
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"]
        else:
            prices = data["Close"]
    else:
        # Single ticker case
        if "Adj Close" in data.columns:
            prices = data[["Adj Close"]].copy()
            prices.columns = [TICKERS[0]]
        elif "Close" in data.columns:
            prices = data[["Close"]].copy()
            prices.columns = [TICKERS[0]]
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found in yfinance data")

    # Ensure prices is a DataFrame with ticker columns
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna()
    print(f"Fetched {len(prices)} trading days, {len(prices.columns)} tickers")

    prices.to_csv(PRICES_CACHE)
    print(f"Cached to {PRICES_CACHE}")

    return prices


def fetch_risk_free_rate(start_date=None, end_date=None, use_cache=True):
    """Fetch 3-month T-bill rate from FRED (DGS3MO) via API."""
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365 * 10)

    if use_cache and os.path.exists(RF_CACHE):
        print(f"Loading risk-free rate from cache: {RF_CACHE}")
        rf_data = pd.read_csv(RF_CACHE, index_col=0, parse_dates=True)
        return rf_data

    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "your_key_here":
        print("WARNING: FRED_API_KEY not found or invalid in .env. Using fixed 4% annualized rate as fallback.")
        # Create a dummy rate series
        prices = pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True)
        rf_data = pd.DataFrame(
            {"rate": [0.04] * len(prices)},
            index=prices.index
        )
    else:
        print(f"Fetching risk-free rate from FRED (DGS3MO)...")
        url = "https://api.stlouisfed.org/fred/series/data"
        params = {
            "series_id": "DGS3MO",
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date.strftime("%Y-%m-%d"),
            "observation_end": end_date.strftime("%Y-%m-%d"),
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()["observations"]

            df_list = []
            for obs in data:
                try:
                    rate = float(obs["value"]) / 100  # Convert % to decimal, annualized
                    df_list.append({
                        "date": pd.to_datetime(obs["date"]),
                        "rate": rate
                    })
                except (ValueError, KeyError):
                    continue

            rf_data = pd.DataFrame(df_list).set_index("date").sort_index()
            print(f"Fetched {len(rf_data)} risk-free rate observations")
        except Exception as e:
            print(f"FRED API error: {e}. Using fixed 4% fallback.")
            prices = pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True)
            rf_data = pd.DataFrame(
                {"rate": [0.04] * len(prices)},
                index=prices.index
            )

    rf_data.to_csv(RF_CACHE)
    print(f"Cached to {RF_CACHE}")

    return rf_data


def load_cached_data():
    """Load cached prices and risk-free rate."""
    prices = pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True)
    rf_data = pd.read_csv(RF_CACHE, index_col=0, parse_dates=True)
    return prices, rf_data


def compute_stats(prices, rf_data):
    """Compute annualized returns, Ledoit-Wolf covariance, and correlation."""
    # Daily returns
    returns = prices.pct_change().dropna()

    # Annualized returns
    annual_returns = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)

    # Risk-free rate: use average of the period
    rf_rate = rf_data["rate"].mean()

    # Ledoit-Wolf shrinkage covariance
    lw = LedoitWolf()
    cov_matrix, _ = lw.fit(returns.values).covariance_, lw.shrinkage_
    cov_matrix = pd.DataFrame(
        cov_matrix,
        index=returns.columns,
        columns=returns.columns
    )

    # Annualize covariance
    cov_matrix_annual = cov_matrix * 252

    # Correlation
    correlation = returns.corr()

    # Print summary table
    print("\n" + "="*80)
    print("RETURN / VOLATILITY / CORRELATION TABLE")
    print("="*80)

    stats_df = pd.DataFrame({
        "Annual Return": annual_returns,
        "Annual Vol": annual_vol,
        "Sharpe (rf={:.2%})".format(rf_rate): (annual_returns - rf_rate) / annual_vol
    })
    print(stats_df.round(4))

    print("\n" + "-"*80)
    print("CORRELATION MATRIX")
    print("-"*80)
    print(correlation.round(3))

    print("\n" + "-"*80)
    print(f"Risk-Free Rate (Average): {rf_rate:.4f} ({rf_rate*100:.2f}% annualized)")
    print("="*80 + "\n")

    return annual_returns, cov_matrix_annual, correlation, rf_rate


def fetch_all_data(use_cache=True):
    """Orchestrate full data fetch: prices + risk-free rate, return stats."""
    prices = fetch_prices(use_cache=use_cache)
    rf_data = fetch_risk_free_rate(use_cache=use_cache)

    # Align dates
    prices = prices[prices.index.isin(rf_data.index)]
    rf_data = rf_data[rf_data.index.isin(prices.index)]

    print(f"\nFinal aligned data: {len(prices)} trading days")

    annual_returns, cov_matrix, correlation, rf_rate = compute_stats(prices, rf_data)

    return prices, rf_data, annual_returns, cov_matrix, correlation, rf_rate
