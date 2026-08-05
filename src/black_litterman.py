"""Black-Litterman model: market-implied returns + subjective views."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import inv

from .data import load_cached_data, compute_stats


# ============================================================================
# SUBJECTIVE VIEWS (EDITABLE)
# ============================================================================
# Define your market views here. Each view compares two assets:
# "outperformer" will outperform "underperformer" by "alpha" (as decimal, e.g., 0.02 = 2%)
VIEWS = [
    {"outperformer": "EEM", "underperformer": "SPY", "alpha": 0.02},  # Emerging markets outperform US by 2%
    {"outperformer": "GLD", "underperformer": "AGG", "alpha": 0.01},   # Gold outperforms bonds by 1%
]

# Confidence in each view (as decimens between 0 and 1; 0.5 = 50% confidence)
VIEW_CONFIDENCES = [0.5, 0.4]

# ============================================================================


def market_implied_returns(market_weights, cov_matrix, rf_rate, risk_aversion=2.5):
    """
    Derive market-implied equilibrium returns via reverse optimization.

    Assumptions:
    - Market weights represent equilibrium positions
    - Risk aversion = 2.5 (standard assumption for global markets)
    """
    n = len(market_weights)
    pi = risk_aversion * (cov_matrix @ market_weights)
    return pi


def build_view_matrices(tickers, views):
    """
    Build view matrix (P) and view return vector (Q) for Black-Litterman.

    P: Each row represents one view. For a pairwise view, the row has:
       +1 for outperformer, -1 for underperformer, 0 for others.
    Q: Vector of expected outperformance alpha for each view.
    """
    n = len(tickers)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    P_list = []
    Q_list = []

    for view in views:
        row = np.zeros(n)
        row[ticker_to_idx[view["outperformer"]]] = 1.0
        row[ticker_to_idx[view["underperformer"]]] = -1.0
        P_list.append(row)
        Q_list.append(view["alpha"])

    P = np.array(P_list)  # shape: (num_views, n_assets)
    Q = np.array(Q_list)  # shape: (num_views,)

    return P, Q


def black_litterman_posterior(
    prior_returns,
    cov_matrix,
    P,
    Q,
    view_confidences,
    risk_aversion=2.5,
):
    """
    Compute Black-Litterman posterior returns.

    Inputs:
    - prior_returns: Market-implied equilibrium returns
    - cov_matrix: Annualized covariance matrix
    - P: View matrix (num_views x n_assets)
    - Q: View returns vector (num_views,)
    - view_confidences: Confidence in each view (0 to 1)
    - risk_aversion: Risk aversion coefficient

    Returns:
    - posterior_returns: BL-adjusted expected returns
    - posterior_cov: BL-adjusted covariance (same as input, uncertainty captured in V)
    """
    n = len(prior_returns)

    # Omega: Diagonal covariance matrix of view uncertainty
    # tau scales the view uncertainty relative to prior uncertainty
    tau = 0.05  # Typically 0.02 to 0.1; we use 0.05
    omega_diag = np.zeros(len(view_confidences))
    for i, conf in enumerate(view_confidences):
        # Higher confidence = lower uncertainty
        uncertainty = (1.0 - conf) ** 2  # scale to [0, 1]
        omega_diag[i] = tau * (P[i] @ cov_matrix @ P[i]) * uncertainty

    Omega = np.diag(omega_diag)

    # Black-Litterman formula
    # mu_BL = pi + Sigma * P^T * (P * Sigma * P^T + Omega)^{-1} * (Q - P * pi)
    try:
        inv_term = inv(P @ cov_matrix @ P.T + Omega)
        posterior = prior_returns + cov_matrix @ P.T @ inv_term @ (Q - P @ prior_returns)
    except np.linalg.LinAlgError:
        print("WARNING: Singular matrix in BL calculation; returning prior returns")
        posterior = prior_returns

    return posterior


def bl_posterior_returns(prices=None, rf_data=None):
    """Compute Black-Litterman posterior returns."""
    if prices is None or rf_data is None:
        prices, rf_data = load_cached_data()

    annual_returns, cov_matrix, correlation, rf_rate = compute_stats(prices, rf_data)
    tickers = annual_returns.index.tolist()

    # Market-implied equilibrium returns
    # Use equal-weight as proxy for market weights (since these are ETFs, not individual stocks)
    market_weights = np.ones(len(tickers)) / len(tickers)
    prior_returns = market_implied_returns(market_weights, cov_matrix.values, rf_rate)

    # Build view matrices
    P, Q = build_view_matrices(tickers, VIEWS)

    # Compute posterior
    posterior_returns = black_litterman_posterior(
        prior_returns,
        cov_matrix.values,
        P,
        Q,
        VIEW_CONFIDENCES,
    )

    # Return as Series for easier access
    prior_series = pd.Series(prior_returns, index=tickers)
    posterior_series = pd.Series(posterior_returns, index=tickers)

    return prior_series, posterior_series, rf_rate, cov_matrix, tickers


def plot_bl_comparison(save_path="results/bl_comparison.png"):
    """Plot prior (market-implied) vs posterior (BL-adjusted) returns."""
    prior, posterior, rf_rate, cov_matrix, tickers = bl_posterior_returns()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(tickers))
    width = 0.35

    bars1 = ax.bar(x - width/2, prior.values * 100, width, label="Market-Implied (Prior)", alpha=0.8)
    bars2 = ax.bar(x + width/2, posterior.values * 100, width, label="Black-Litterman (Posterior)", alpha=0.8)

    ax.set_xlabel("Asset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Expected Annual Return (%)", fontsize=12, fontweight="bold")
    ax.set_title("Black-Litterman: Prior vs Posterior Returns", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend(fontsize=11)
    ax.axhline(y=rf_rate * 100, color="red", linestyle="--", linewidth=2, label=f"Risk-Free Rate ({rf_rate*100:.1f}%)", alpha=0.7)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()

    return prior, posterior


def print_bl_summary():
    """Print summary of BL model."""
    prior, posterior, rf_rate, cov_matrix, tickers = bl_posterior_returns()

    print("\n" + "="*80)
    print("BLACK-LITTERMAN MODEL SUMMARY")
    print("="*80)

    print("\nSUBJECTIVE VIEWS:")
    for i, (view, conf) in enumerate(zip(VIEWS, VIEW_CONFIDENCES)):
        print(f"  {i+1}. {view['outperformer']} outperforms {view['underperformer']} by {view['alpha']*100:.1f}% (confidence: {conf*100:.0f}%)")

    print("\n" + "-"*80)
    print("RETURNS COMPARISON")
    print("-"*80)

    comparison_df = pd.DataFrame({
        "Market-Implied (%)": prior.values * 100,
        "Black-Litterman (%)": posterior.values * 100,
        "Change (bps)": (posterior.values - prior.values) * 10000,
    }, index=tickers)

    print(comparison_df.to_string())
    print("\n" + "="*80)


if __name__ == "__main__":
    plot_bl_comparison()
    print_bl_summary()
