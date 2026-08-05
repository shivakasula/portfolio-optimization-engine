"""Classic Markowitz efficient frontier optimization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
import cvxpy as cp

from .data import load_cached_data, compute_stats


def compute_portfolio_return(weights, returns):
    """Compute portfolio annual return."""
    return np.sum(weights * returns)


def compute_portfolio_vol(weights, cov_matrix):
    """Compute portfolio annual volatility."""
    return np.sqrt(weights @ cov_matrix @ weights)


def compute_sharpe_ratio(weights, returns, cov_matrix, rf_rate):
    """Compute Sharpe ratio."""
    port_ret = compute_portfolio_return(weights, returns)
    port_vol = compute_portfolio_vol(weights, cov_matrix)
    return (port_ret - rf_rate) / port_vol


def efficient_frontier(returns, cov_matrix, rf_rate, num_points=100):
    """Compute efficient frontier: minimize volatility for each return level."""
    n_assets = len(returns)
    min_ret = returns.min()
    max_ret = returns.max()
    target_returns = np.linspace(min_ret, max_ret, num_points)

    frontier_vols = []
    frontier_rets = []

    for target_ret in target_returns:
        # Minimize volatility subject to: return = target, weights sum to 1, weights >= 0
        def objective(w):
            return compute_portfolio_vol(w, cov_matrix)

        def ret_constraint(w):
            return compute_portfolio_return(w, returns) - target_ret

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # weights sum to 1
            {"type": "eq", "fun": ret_constraint},  # target return
        ]

        x0 = np.ones(n_assets) / n_assets
        bounds = [(0, 1) for _ in range(n_assets)]  # no shorting

        try:
            result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            if result.success:
                frontier_vols.append(result.fun)
                frontier_rets.append(target_ret)
        except:
            pass

    return np.array(frontier_rets), np.array(frontier_vols)


def max_sharpe_portfolio(returns, cov_matrix, rf_rate):
    """Find portfolio that maximizes Sharpe ratio."""
    n_assets = len(returns)

    def neg_sharpe(w):
        return -compute_sharpe_ratio(w, returns, cov_matrix, rf_rate)

    x0 = np.ones(n_assets) / n_assets
    bounds = [(0, 1) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    return result.x, -result.fun  # return weights and Sharpe ratio


def min_volatility_portfolio(returns, cov_matrix):
    """Find minimum volatility portfolio (Global Minimum Variance)."""
    n_assets = len(returns)

    def objective(w):
        return compute_portfolio_vol(w, cov_matrix)

    x0 = np.ones(n_assets) / n_assets
    bounds = [(0, 1) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    return result.x, result.fun  # return weights and volatility


def random_portfolios(returns, cov_matrix, rf_rate, n_samples=5000):
    """Generate random portfolios for visualization."""
    n_assets = len(returns)
    weights_list = []
    sharpes = []
    rets = []
    vols = []

    np.random.seed(42)
    for _ in range(n_samples):
        w = np.random.dirichlet(np.ones(n_assets))
        weights_list.append(w)
        rets.append(compute_portfolio_return(w, returns))
        vols.append(compute_portfolio_vol(w, cov_matrix))
        sharpes.append(compute_sharpe_ratio(w, returns, cov_matrix, rf_rate))

    return np.array(weights_list), np.array(rets), np.array(vols), np.array(sharpes)


def plot_efficient_frontier(prices=None, rf_data=None, save_path="results/efficient_frontier.png"):
    """Plot efficient frontier with random portfolios and key portfolios."""
    if prices is None or rf_data is None:
        prices, rf_data = load_cached_data()

    annual_returns, cov_matrix, correlation, rf_rate = compute_stats(prices, rf_data)

    # Compute efficient frontier
    frontier_rets, frontier_vols = efficient_frontier(annual_returns.values, cov_matrix.values, rf_rate)

    # Find key portfolios
    max_sharpe_w, max_sharpe_s = max_sharpe_portfolio(annual_returns.values, cov_matrix.values, rf_rate)
    min_vol_w, min_vol_v = min_volatility_portfolio(annual_returns.values, cov_matrix.values)
    min_vol_ret = compute_portfolio_return(min_vol_w, annual_returns.values)

    # Generate random portfolios
    rand_w, rand_rets, rand_vols, rand_sharpes = random_portfolios(
        annual_returns.values, cov_matrix.values, rf_rate, n_samples=5000
    )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot of random portfolios, colored by Sharpe ratio
    scatter = ax.scatter(
        rand_vols, rand_rets, c=rand_sharpes, cmap="viridis", alpha=0.5, s=30,
        label="Random Portfolios", edgecolors="none"
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", fontsize=11)

    # Efficient frontier
    ax.plot(frontier_vols, frontier_rets, "r-", linewidth=3, label="Efficient Frontier", zorder=10)

    # Key portfolios
    max_sharpe_vol = compute_portfolio_vol(max_sharpe_w, cov_matrix.values)
    max_sharpe_ret = compute_portfolio_return(max_sharpe_w, annual_returns.values)
    ax.scatter([max_sharpe_vol], [max_sharpe_ret], color="gold", s=400, marker="*",
               label=f"Max Sharpe ({max_sharpe_s:.3f})", edgecolors="black", linewidth=2, zorder=15)

    ax.scatter([min_vol_v], [min_vol_ret], color="cyan", s=300, marker="s",
               label=f"Min Vol ({min_vol_v:.3f})", edgecolors="black", linewidth=2, zorder=15)

    # Risk-free asset
    ax.scatter([0], [rf_rate], color="red", s=200, marker="o", label="Risk-Free Asset",
               edgecolors="black", linewidth=2, zorder=15)

    # Capital Allocation Line (from risk-free to max-Sharpe)
    cal_x = np.linspace(0, max_sharpe_vol * 1.2, 100)
    cal_y = rf_rate + max_sharpe_s * cal_x
    ax.plot(cal_x, cal_y, "r--", linewidth=2, alpha=0.7, label="Capital Allocation Line")

    ax.set_xlabel("Volatility (Annual)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Expected Return (Annual)", fontsize=12, fontweight="bold")
    ax.set_title("Efficient Frontier: Markowitz Portfolio Optimization", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")
    plt.close()

    return {
        "annual_returns": annual_returns,
        "cov_matrix": cov_matrix,
        "rf_rate": rf_rate,
        "max_sharpe_weights": max_sharpe_w,
        "max_sharpe_ratio": max_sharpe_s,
        "min_vol_weights": min_vol_w,
        "min_vol_volatility": min_vol_v,
        "tickers": annual_returns.index,
    }


if __name__ == "__main__":
    result = plot_efficient_frontier()
    print("\n" + "="*80)
    print("MAX SHARPE PORTFOLIO")
    print("="*80)
    for ticker, w in zip(result["tickers"], result["max_sharpe_weights"]):
        if w > 0.001:
            print(f"{ticker}: {w:.2%}")

    print("\n" + "="*80)
    print("MINIMUM VOLATILITY PORTFOLIO")
    print("="*80)
    for ticker, w in zip(result["tickers"], result["min_vol_weights"]):
        if w > 0.001:
            print(f"{ticker}: {w:.2%}")
