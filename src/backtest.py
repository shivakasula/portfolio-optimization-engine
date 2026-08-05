"""Backtest static BL-constrained allocation vs 60/40 benchmark."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data import load_cached_data, compute_stats
from .optimization_constrained import constrained_optimal_portfolio
from .black_litterman import bl_posterior_returns


def backtest_allocation(prices, allocation, rebalance_freq="YE"):
    """
    Backtest a static allocation (rebalanced periodically).

    Parameters:
    - prices: DataFrame with daily adjusted close prices
    - allocation: Dict mapping tickers to weights
    - rebalance_freq: 'Y' for annual, 'M' for monthly, 'D' for daily

    Returns:
    - cumulative_returns: Daily cumulative return series
    """
    # Daily returns
    daily_returns = prices.pct_change().dropna()

    # Initialize portfolio value
    portfolio_value = pd.Series(1.0, index=daily_returns.index)
    current_weights = np.array([allocation.get(t, 0.0) for t in daily_returns.columns])

    rebalance_dates = pd.date_range(start=daily_returns.index[0], end=daily_returns.index[-1], freq=rebalance_freq)

    for i in range(1, len(daily_returns)):
        date = daily_returns.index[i]
        prev_date = daily_returns.index[i - 1]

        # Rebalance if necessary
        if i > 0 and len(rebalance_dates[rebalance_dates <= date]) > len(rebalance_dates[rebalance_dates < prev_date]):
            current_weights = np.array([allocation.get(t, 0.0) for t in daily_returns.columns])

        # Compute daily return of portfolio
        daily_ret = np.sum(current_weights * daily_returns.iloc[i].values)
        portfolio_value.iloc[i] = portfolio_value.iloc[i - 1] * (1 + daily_ret)

    return portfolio_value


def backtest_benchmark_60_40(prices):
    """Backtest 60% SPY, 40% TLT benchmark."""
    allocation = {"SPY": 0.60, "TLT": 0.40}
    return backtest_allocation(prices, allocation)


def backtest_bl_constrained(prices, rf_data):
    """Backtest BL-constrained allocation."""
    annual_returns, cov_matrix, _, rf_rate = compute_stats(prices, rf_data)
    tickers = annual_returns.index.tolist()

    prior, posterior, _, _, _ = bl_posterior_returns(prices, rf_data)
    bl_weights = constrained_optimal_portfolio(
        posterior.values,
        cov_matrix.values,
        rf_rate,
        max_weight=0.40,
    )

    allocation = {t: w for t, w in zip(tickers, bl_weights)}
    return backtest_allocation(prices, allocation)


def compute_backtest_stats(cumulative_returns, annual_return_series=None, rf_rate=0.04):
    """
    Compute key backtest metrics.

    Parameters:
    - cumulative_returns: Daily cumulative return series (starting at 1.0)
    - annual_return_series: Daily returns (needed for volatility)
    - rf_rate: Risk-free rate

    Returns:
    - dict with CAGR, annualized vol, Sharpe, max drawdown
    """
    # CAGR
    start_value = cumulative_returns.iloc[0]
    end_value = cumulative_returns.iloc[-1]
    n_years = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days / 365.25
    cagr = (end_value / start_value) ** (1 / n_years) - 1

    # Annualized volatility
    if annual_return_series is not None:
        annual_vol = annual_return_series.std() * np.sqrt(252)
    else:
        daily_rets = cumulative_returns.pct_change().dropna()
        annual_vol = daily_rets.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe = (cagr - rf_rate) / annual_vol if annual_vol > 0 else 0

    # Max drawdown
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "CAGR": cagr,
        "Annual Vol": annual_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
    }


def plot_backtest_comparison(save_path="results/backtest_comparison.png"):
    """Plot cumulative returns: BL-constrained vs 60/40 benchmark."""
    prices, rf_data = load_cached_data()

    # Compute both strategies
    bl_returns = backtest_bl_constrained(prices, rf_data)
    benchmark_returns = backtest_benchmark_60_40(prices)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(bl_returns.index, bl_returns.values, label="BL-Constrained", linewidth=2.5, color="steelblue")
    ax.plot(benchmark_returns.index, benchmark_returns.values, label="60/40 Benchmark", linewidth=2.5, color="coral")

    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Return", fontsize=12, fontweight="bold")
    ax.set_title("10-Year Backtest: BL-Constrained vs 60/40 Benchmark", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{(x-1)*100:.0f}%'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()

    return bl_returns, benchmark_returns


def print_backtest_summary():
    """Print detailed backtest statistics."""
    prices, rf_data = load_cached_data()
    annual_returns, cov_matrix, _, rf_rate = compute_stats(prices, rf_data)

    # Compute daily returns
    daily_returns = prices.pct_change().dropna()

    # Backtest both strategies
    bl_returns = backtest_bl_constrained(prices, rf_data)
    benchmark_returns = backtest_benchmark_60_40(prices)

    # Compute stats
    bl_stats = compute_backtest_stats(bl_returns, annual_return_series=daily_returns.std(axis=1), rf_rate=rf_rate)
    bench_stats = compute_backtest_stats(benchmark_returns, annual_return_series=daily_returns.std(axis=1), rf_rate=rf_rate)

    print("\n" + "="*80)
    print("10-YEAR BACKTEST SUMMARY (2016-2026)")
    print("="*80)

    summary_df = pd.DataFrame({
        "BL-Constrained": bl_stats,
        "60/40 Benchmark": bench_stats,
    }).T

    print("\n" + summary_df.to_string())

    print("\n" + "-"*80)
    print(f"Final Value (BL-Constrained): ${bl_returns.iloc[-1]:.2f} (from $1.00)")
    print(f"Final Value (60/40 Benchmark): ${benchmark_returns.iloc[-1]:.2f} (from $1.00)")
    print("="*80)


if __name__ == "__main__":
    plot_backtest_comparison()
    print_backtest_summary()
