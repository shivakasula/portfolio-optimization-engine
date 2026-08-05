"""Constrained optimization using Black-Litterman posterior returns."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from .data import load_cached_data, compute_stats
from .black_litterman import bl_posterior_returns


def constrained_optimal_portfolio(
    returns,
    cov_matrix,
    rf_rate,
    max_weight=0.40,
):
    """
    Optimize portfolio subject to:
    - weights sum to 1
    - no shorting (weights >= 0)
    - max weight per asset = max_weight (e.g., 0.40 = 40%)
    - maximize Sharpe ratio
    """
    n_assets = len(returns)

    def neg_sharpe(w):
        ret = np.sum(w * returns)
        vol = np.sqrt(w @ cov_matrix @ w)
        if vol == 0:
            return 0
        return -(ret - rf_rate) / vol

    x0 = np.ones(n_assets) / n_assets
    bounds = [(0, max_weight) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    return result.x


def plot_allocation_comparison(save_path="results/allocations_comparison.png"):
    """Plot allocation pie charts: Markowitz vs BL-constrained."""
    # Load data and compute stats
    prices, rf_data = load_cached_data()
    annual_returns, cov_matrix, correlation, rf_rate = compute_stats(prices, rf_data)
    tickers = annual_returns.index.tolist()

    # Classic Markowitz (unconstrained, max Sharpe)
    def unconstrained_sharpe(w):
        ret = np.sum(w * annual_returns.values)
        vol = np.sqrt(w @ cov_matrix.values @ w)
        if vol == 0:
            return 0
        return -(ret - rf_rate) / vol

    n_assets = len(tickers)
    x0 = np.ones(n_assets) / n_assets
    bounds_unconstrained = [(0, 1) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result_unconstrained = minimize(
        unconstrained_sharpe, x0, method="SLSQP", bounds=bounds_unconstrained, constraints=constraints
    )
    markowitz_weights = result_unconstrained.x

    # BL-constrained (max 40% per asset)
    prior, posterior, _, _, _ = bl_posterior_returns()
    bl_weights = constrained_optimal_portfolio(
        posterior.values,
        cov_matrix.values,
        rf_rate,
        max_weight=0.40,
    )

    # Create pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Filter out near-zero weights for cleaner display
    threshold = 0.005

    def plot_pie(ax, weights, title):
        # Mask small weights
        weights_display = weights.copy()
        weights_display[weights_display < threshold] = 0
        weights_display = weights_display / weights_display.sum()

        colors = plt.cm.Set3(np.linspace(0, 1, len(tickers)))
        wedges, texts, autotexts = ax.pie(
            weights_display,
            labels=tickers,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={"fontsize": 10},
        )
        for autotext in autotexts:
            autotext.set_color("black")
            autotext.set_fontsize(9)
            autotext.set_fontweight("bold")
        ax.set_title(title, fontsize=12, fontweight="bold")

    plot_pie(ax1, markowitz_weights, "Markowitz (Unconstrained)")
    plot_pie(ax2, bl_weights, "Black-Litterman (Constrained, Max 40%)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()

    return markowitz_weights, bl_weights, tickers


def print_allocation_summary():
    """Print allocation weights for both strategies."""
    markowitz_w, bl_w, tickers = plot_allocation_comparison()

    print("\n" + "="*80)
    print("PORTFOLIO ALLOCATIONS")
    print("="*80)

    comparison_df = pd.DataFrame({
        "Markowitz (%)": markowitz_w * 100,
        "BL-Constrained (%)": bl_w * 100,
    }, index=tickers)

    print(comparison_df.round(2).to_string())

    print("\n" + "-"*80)
    print(f"Markowitz weights sum: {markowitz_w.sum():.4f}")
    print(f"BL-Constrained weights sum: {bl_w.sum():.4f}")
    print("="*80)


if __name__ == "__main__":
    print_allocation_summary()
