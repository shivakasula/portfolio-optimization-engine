"""Interactive Streamlit web app for portfolio optimization engine."""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
from io import BytesIO
from scipy.optimize import minimize

from src.data import load_cached_data, compute_stats, fetch_all_data, TICKERS
from src.black_litterman import (
    build_view_matrices,
    black_litterman_posterior,
    market_implied_returns,
)
from src.optimization import (
    efficient_frontier,
    compute_sharpe_ratio,
    compute_portfolio_return,
    compute_portfolio_vol,
)
from src.optimization_constrained import constrained_optimal_portfolio
from src.backtest import backtest_allocation


# ============================================================================
# PAGE CONFIG & SESSION STATE
# ============================================================================

st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 Portfolio Optimization Engine")
st.markdown(
    "Interactive tool for Markowitz & Black-Litterman portfolio optimization"
)


# ============================================================================
# AUTO-LOAD DATA WITH CACHING
# ============================================================================

@st.cache_data(show_spinner=False)
def load_portfolio_data():
    """Load portfolio data with Streamlit caching."""
    return fetch_all_data(use_cache=True)


# Auto-load on startup
if "data_loaded" not in st.session_state:
    with st.spinner("📊 Loading portfolio data..."):
        try:
            (
                prices, rf_data, annual_returns, cov_matrix, correlation, rf_rate
            ) = load_portfolio_data()
            st.session_state.prices = prices
            st.session_state.rf_data = rf_data
            st.session_state.annual_returns = annual_returns
            st.session_state.cov_matrix = cov_matrix
            st.session_state.correlation = correlation
            st.session_state.rf_rate = rf_rate
            st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.session_state.data_loaded = False

# Initialize session state (backup)
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "prices" not in st.session_state:
    st.session_state.prices = None
if "rf_data" not in st.session_state:
    st.session_state.rf_data = None
if "annual_returns" not in st.session_state:
    st.session_state.annual_returns = None
if "cov_matrix" not in st.session_state:
    st.session_state.cov_matrix = None
if "correlation" not in st.session_state:
    st.session_state.correlation = None
if "rf_rate" not in st.session_state:
    st.session_state.rf_rate = None


# ============================================================================
# SIDEBAR: DATA & LOAD
# ============================================================================

with st.sidebar:
    st.header("📋 Data & Settings")

    if st.button("🔄 Load/Refresh Data", use_container_width=True):
        with st.spinner("Fetching data..."):
            try:
                (
                    st.session_state.prices,
                    st.session_state.rf_data,
                    st.session_state.annual_returns,
                    st.session_state.cov_matrix,
                    st.session_state.correlation,
                    st.session_state.rf_rate,
                ) = fetch_all_data(use_cache=True)
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")

    if st.session_state.data_loaded:
        st.success("✅ Data ready")
        prices = st.session_state.prices
        st.info(
            f"📈 {len(prices)} trading days | {len(prices.columns)} assets | "
            f"Period: {prices.index[0].date()} to {prices.index[-1].date()}"
        )

        # Display correlation heatmap
        with st.expander("📊 Correlation Matrix"):
            fig = px.imshow(
                st.session_state.correlation,
                labels=dict(color="Correlation"),
                x=st.session_state.correlation.columns,
                y=st.session_state.correlation.columns,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
            )
            fig.update_layout(height=500, width=500)
            st.plotly_chart(fig, use_container_width=False)
    else:
        st.warning("⚠️ Click 'Load Data' to get started")


# ============================================================================
# MAIN CONTENT: TABS
# ============================================================================

if st.session_state.data_loaded:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Data Overview", "🎯 Markowitz", "🔮 Black-Litterman", "📈 Backtest", "💾 Export"]
    )

    # ========================================================================
    # TAB 1: DATA OVERVIEW
    # ========================================================================
    with tab1:
        st.header("Data Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk-Free Rate", f"{st.session_state.rf_rate:.2%}")
        with col2:
            st.metric("Asset Universe", len(st.session_state.annual_returns))
        with col3:
            st.metric("Observation Period", f"{len(st.session_state.prices)} days")

        # Returns & volatility table
        st.subheader("Asset Statistics")
        stats_df = pd.DataFrame(
            {
                "Annual Return": st.session_state.annual_returns * 100,
                "Annual Vol": (
                    st.session_state.annual_returns.index.map(
                        lambda t: np.sqrt(
                            st.session_state.cov_matrix.loc[t, t]
                        )
                    )
                    * 100
                ),
                "Sharpe Ratio": (
                    (
                        st.session_state.annual_returns
                        - st.session_state.rf_rate
                    )
                    / np.array(
                        [
                            np.sqrt(st.session_state.cov_matrix.loc[t, t])
                            for t in st.session_state.annual_returns.index
                        ]
                    )
                ),
            }
        )
        st.dataframe(stats_df, use_container_width=True)

        # Price chart
        st.subheader("Historical Prices (Normalized)")
        prices_normalized = st.session_state.prices / st.session_state.prices.iloc[0] * 100
        st.line_chart(prices_normalized)

    # ========================================================================
    # TAB 2: MARKOWITZ OPTIMIZATION
    # ========================================================================
    with tab2:
        st.header("Markowitz Efficient Frontier")

        col1, col2 = st.columns(2)
        with col1:
            n_portfolios = st.slider(
                "Frontier Resolution", min_value=20, max_value=200, value=100
            )
        with col2:
            show_random = st.checkbox("Show Random Portfolios", value=True)

        # Compute efficient frontier
        with st.spinner("Computing efficient frontier..."):
            frontier_returns, frontier_vols = efficient_frontier(
                st.session_state.annual_returns.values,
                st.session_state.cov_matrix.values,
                st.session_state.rf_rate,
                num_points=n_portfolios,
            )

            # Find max Sharpe portfolio on frontier
            frontier_sharpes = (frontier_returns - st.session_state.rf_rate) / (frontier_vols + 1e-10)
            max_sharpe_idx = np.argmax(frontier_sharpes)
            max_sharpe_return = frontier_returns[max_sharpe_idx]
            max_sharpe_vol = frontier_vols[max_sharpe_idx]

            # Compute max Sharpe weights
            def neg_sharpe_obj(w):
                return -compute_sharpe_ratio(
                    w, st.session_state.annual_returns.values,
                    st.session_state.cov_matrix.values, st.session_state.rf_rate
                )

            n_assets = len(st.session_state.annual_returns)
            x0 = np.ones(n_assets) / n_assets
            bounds = [(0, 1) for _ in range(n_assets)]
            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            result = minimize(
                neg_sharpe_obj, x0, method="SLSQP", bounds=bounds, constraints=constraints
            )
            max_sharpe_weights = result.x

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Sharpe Ratio", f"{frontier_sharpes[max_sharpe_idx]:.3f}")
        with col2:
            st.metric(
                "Expected Return", f"{max_sharpe_return:.2%}"
            )
        with col3:
            st.metric("Expected Volatility", f"{max_sharpe_vol:.2%}")

        # Plot efficient frontier
        fig = go.Figure()

        # Add frontier line
        fig.add_trace(
            go.Scatter(
                x=frontier_vols * 100,
                y=frontier_returns * 100,
                mode="lines",
                name="Efficient Frontier",
                line=dict(color="blue", width=3),
            )
        )

        # Add max Sharpe point
        fig.add_trace(
            go.Scatter(
                x=[max_sharpe_vol * 100],
                y=[max_sharpe_return * 100],
                mode="markers",
                name="Max Sharpe",
                marker=dict(size=15, color="gold", symbol="star"),
            )
        )

        # Add random portfolios if enabled
        if show_random:
            n_random = 5000
            np.random.seed(42)
            random_weights = np.random.dirichlet(
                np.ones(len(st.session_state.annual_returns)), n_random
            )
            random_returns = random_weights @ st.session_state.annual_returns.values
            random_vols = np.array(
                [
                    np.sqrt(w @ st.session_state.cov_matrix.values @ w)
                    for w in random_weights
                ]
            )
            random_sharpes = (random_returns - st.session_state.rf_rate) / random_vols

            fig.add_trace(
                go.Scatter(
                    x=random_vols * 100,
                    y=random_returns * 100,
                    mode="markers",
                    name="Random Portfolios",
                    marker=dict(
                        size=4,
                        color=random_sharpes,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Sharpe"),
                    ),
                    opacity=0.5,
                )
            )

        # Capital Allocation Line (CAL)
        cal_vols = np.array([0, frontier_vols.max()])
        cal_returns = st.session_state.rf_rate + (
            (max_sharpe_return - st.session_state.rf_rate)
            / max_sharpe_vol
        ) * cal_vols
        fig.add_trace(
            go.Scatter(
                x=cal_vols * 100,
                y=cal_returns * 100,
                mode="lines",
                name="CAL (Capital Allocation Line)",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

        fig.update_layout(
            title="Efficient Frontier & Capital Allocation Line",
            xaxis_title="Expected Volatility (%)",
            yaxis_title="Expected Return (%)",
            hovermode="closest",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Max Sharpe portfolio allocation
        st.subheader("Max Sharpe Allocation")
        allocation_df = pd.DataFrame(
            {
                "Asset": st.session_state.annual_returns.index,
                "Weight": max_sharpe_weights * 100,
            }
        )
        allocation_df = allocation_df[allocation_df["Weight"] > 0.01].sort_values(
            "Weight", ascending=False
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(
                allocation_df,
                names="Asset",
                values="Weight",
                title="Max Sharpe Allocation",
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.dataframe(allocation_df, use_container_width=True, hide_index=True)

    # ========================================================================
    # TAB 3: BLACK-LITTERMAN MODEL
    # ========================================================================
    with tab3:
        st.header("Black-Litterman Model")

        st.markdown(
            """
        Define subjective market views to adjust your portfolio beyond market consensus.
        """
        )

        # View input section
        st.subheader("📝 Subjective Views")

        n_views = st.number_input(
            "Number of views", min_value=0, max_value=5, value=2
        )

        views = []
        view_confidences = []

        view_cols = st.columns([2, 2, 2, 2, 1])
        view_cols[0].write("**Outperformer**")
        view_cols[1].write("**Underperformer**")
        view_cols[2].write("**Alpha (%)**")
        view_cols[3].write("**Confidence (%)**")
        view_cols[4].write("")

        assets = sorted(st.session_state.annual_returns.index.tolist())

        for i in range(n_views):
            cols = st.columns([2, 2, 2, 2, 1])
            with cols[0]:
                outperformer = st.selectbox(
                    "Outperformer",
                    assets,
                    key=f"outperf_{i}",
                    label_visibility="collapsed",
                )
            with cols[1]:
                underperformer = st.selectbox(
                    "Underperformer",
                    assets,
                    key=f"underperf_{i}",
                    label_visibility="collapsed",
                    index=1 if i == 0 else 0,
                )
            with cols[2]:
                alpha = st.number_input(
                    "Alpha",
                    value=2.0,
                    step=0.5,
                    key=f"alpha_{i}",
                    label_visibility="collapsed",
                ) / 100
            with cols[3]:
                confidence = st.number_input(
                    "Confidence",
                    value=50,
                    min_value=0,
                    max_value=100,
                    step=10,
                    key=f"conf_{i}",
                    label_visibility="collapsed",
                ) / 100
            with cols[4]:
                if i > 0:
                    st.write("")

            if outperformer != underperformer:
                views.append(
                    {
                        "outperformer": outperformer,
                        "underperformer": underperformer,
                        "alpha": alpha,
                    }
                )
                view_confidences.append(confidence)

        # Compute BL model
        if views:
            with st.spinner("Computing Black-Litterman posterior..."):
                # Market-implied prior
                market_weights = np.ones(len(assets)) / len(assets)
                prior_returns = market_implied_returns(
                    market_weights,
                    st.session_state.cov_matrix.values,
                    st.session_state.rf_rate,
                )
                prior_series = pd.Series(prior_returns, index=assets)

                # Build view matrices
                P, Q = build_view_matrices(assets, views)

                # Compute posterior
                posterior_returns = black_litterman_posterior(
                    prior_returns,
                    st.session_state.cov_matrix.values,
                    P,
                    Q,
                    view_confidences,
                )
                posterior_series = pd.Series(posterior_returns, index=assets)

            # Plot comparison
            col1, col2 = st.columns(2)

            with col1:
                comparison_df = pd.DataFrame(
                    {
                        "Market-Implied (%)": prior_series * 100,
                        "Black-Litterman (%)": posterior_series * 100,
                    }
                )

                fig = px.bar(
                    comparison_df,
                    barmode="group",
                    title="Prior vs Posterior Returns",
                    labels={"value": "Expected Return (%)", "index": "Asset"},
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(comparison_df, use_container_width=True)

            # Constrained optimization with BL returns
            st.subheader("🎯 Constrained Portfolio (BL-Adjusted)")

            col1, col2 = st.columns(2)
            with col1:
                max_weight = st.slider(
                    "Max Weight per Asset (%)",
                    min_value=5,
                    max_value=100,
                    value=40,
                    step=5,
                ) / 100
            with col2:
                min_weight = st.slider(
                    "Min Weight per Asset (%)",
                    min_value=0,
                    max_value=20,
                    value=0,
                    step=5,
                ) / 100

            with st.spinner("Optimizing constrained portfolio..."):
                bl_weights = constrained_optimal_portfolio(
                    posterior_series.values,
                    st.session_state.cov_matrix.values,
                    st.session_state.rf_rate,
                    max_weight=max_weight,
                )
                bl_return = compute_portfolio_return(bl_weights, posterior_series.values)
                bl_vol = compute_portfolio_vol(bl_weights, st.session_state.cov_matrix.values)
                bl_sharpe = compute_sharpe_ratio(
                    bl_weights, posterior_series.values,
                    st.session_state.cov_matrix.values, st.session_state.rf_rate
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Return", f"{bl_return:.2%}")
            with col2:
                st.metric("Expected Vol", f"{bl_vol:.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{bl_sharpe:.3f}")

            # Allocation chart
            bl_alloc_df = pd.DataFrame(
                {"Asset": assets, "Weight": bl_weights * 100}
            )
            bl_alloc_df = bl_alloc_df[bl_alloc_df["Weight"] > 0.01].sort_values(
                "Weight", ascending=False
            )

            col1, col2 = st.columns(2)
            with col1:
                fig_pie = px.pie(
                    bl_alloc_df,
                    names="Asset",
                    values="Weight",
                    title="BL-Constrained Allocation",
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                st.dataframe(bl_alloc_df, use_container_width=True, hide_index=True)

        else:
            st.info("ℹ️ Enter at least one view to see results")

    # ========================================================================
    # TAB 4: BACKTEST
    # ========================================================================
    with tab4:
        st.header("Backtesting & Strategy Comparison")

        st.markdown(
            """
        Compare a custom allocation strategy against the 60/40 benchmark over the 10-year period.
        """
        )

        # Portfolio input
        st.subheader("Select Portfolio to Backtest")

        portfolio_type = st.radio(
            "Choose allocation",
            ["Max Sharpe (Markowitz)", "Constrained (BL)", "Custom"],
            horizontal=True,
        )

        if portfolio_type == "Max Sharpe (Markowitz)":
            frontier_returns, frontier_vols = efficient_frontier(
                st.session_state.annual_returns.values,
                st.session_state.cov_matrix.values,
                st.session_state.rf_rate,
                num_points=100,
            )
            frontier_sharpes = (frontier_returns - st.session_state.rf_rate) / (frontier_vols + 1e-10)
            max_sharpe_idx = np.argmax(frontier_sharpes)

            # Compute max Sharpe weights
            def neg_sharpe_obj(w):
                return -compute_sharpe_ratio(
                    w, st.session_state.annual_returns.values,
                    st.session_state.cov_matrix.values, st.session_state.rf_rate
                )

            n_assets = len(st.session_state.annual_returns)
            x0 = np.ones(n_assets) / n_assets
            bounds = [(0, 1) for _ in range(n_assets)]
            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            result = minimize(
                neg_sharpe_obj, x0, method="SLSQP", bounds=bounds, constraints=constraints
            )
            test_weights = result.x
            portfolio_name = "Max Sharpe (Markowitz)"

        elif portfolio_type == "Constrained (BL)":
            st.info(
                "ℹ️ Use the Black-Litterman tab to define views and compute the constrained portfolio"
            )
            test_weights = None
            portfolio_name = "BL-Constrained"

        else:  # Custom
            st.subheader("Define Custom Weights")
            assets = sorted(st.session_state.annual_returns.index.tolist())
            custom_weights = {}

            cols = st.columns(len(assets))
            for i, asset in enumerate(assets):
                with cols[i]:
                    custom_weights[asset] = (
                        st.number_input(
                            asset,
                            min_value=0.0,
                            max_value=100.0,
                            value=100 / len(assets),
                            step=5.0,
                            label_visibility="collapsed",
                        )
                        / 100
                    )

            # Normalize
            total = sum(custom_weights.values())
            test_weights = np.array(
                [custom_weights[a] / total for a in assets]
            )
            portfolio_name = "Custom"

        if test_weights is not None:
            with st.spinner(f"Backtesting {portfolio_name}..."):
                # Convert weights array to dict for backtest_allocation
                assets = sorted(st.session_state.annual_returns.index.tolist())
                test_allocation = {assets[i]: test_weights[i] for i in range(len(assets))}

                # Backtest
                bt_portfolio_value = backtest_allocation(
                    st.session_state.prices,
                    test_allocation,
                    rebalance_freq="YE",
                )
                bt_returns = (bt_portfolio_value.pct_change()).dropna()
                bt_cagr = (bt_portfolio_value.iloc[-1] / bt_portfolio_value.iloc[0]) ** (252 / len(st.session_state.prices)) - 1
                bt_vol = bt_returns.std() * np.sqrt(252)
                bt_sharpe = (bt_cagr - st.session_state.rf_rate) / bt_vol
                bt_dd = (bt_portfolio_value / bt_portfolio_value.cummax() - 1).min()

                # Benchmark (60/40 SPY/TLT)
                spy_idx = list(
                    st.session_state.annual_returns.index
                ).index("SPY")
                tlt_idx = list(
                    st.session_state.annual_returns.index
                ).index("TLT")
                bench_allocation = {
                    assets[spy_idx]: 0.6,
                    assets[tlt_idx]: 0.4
                }

                bench_portfolio_value = backtest_allocation(
                    st.session_state.prices,
                    bench_allocation,
                    rebalance_freq="YE",
                )
                bench_returns = (bench_portfolio_value.pct_change()).dropna()
                bench_cagr = (bench_portfolio_value.iloc[-1] / bench_portfolio_value.iloc[0]) ** (252 / len(st.session_state.prices)) - 1
                bench_vol = bench_returns.std() * np.sqrt(252)
                bench_sharpe = (bench_cagr - st.session_state.rf_rate) / bench_vol
                bench_dd = (bench_portfolio_value / bench_portfolio_value.cummax() - 1).min()

            # Comparison metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric(f"{portfolio_name} CAGR", f"{bt_cagr:.2%}")
            with col2:
                st.metric(f"60/40 CAGR", f"{bench_cagr:.2%}")
            with col3:
                st.metric(f"Outperformance", f"{(bt_cagr - bench_cagr):.2%}")
            with col4:
                st.metric(f"{portfolio_name} Sharpe", f"{bt_sharpe:.3f}")
            with col5:
                st.metric(f"60/40 Sharpe", f"{bench_sharpe:.3f}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{portfolio_name} Vol", f"{bt_vol:.2%}")
            with col2:
                st.metric(f"60/40 Vol", f"{bench_vol:.2%}")
            with col3:
                st.metric(f"{portfolio_name} Max DD", f"{bt_dd:.2%}")

            # Plot cumulative returns
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=bt_returns.index,
                    y=(1 + bt_returns).cumprod() * 100,
                    name=portfolio_name,
                    line=dict(width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bench_returns.index,
                    y=(1 + bench_returns).cumprod() * 100,
                    name="60/40 Benchmark",
                    line=dict(width=3),
                )
            )
            fig.update_layout(
                title="Cumulative Return Comparison",
                xaxis_title="Date",
                yaxis_title="Value ($100 initial)",
                hovermode="x unified",
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 5: EXPORT
    # ========================================================================
    with tab5:
        st.header("Export & Download")

        st.markdown(
            """
        Export portfolio allocations, data, and results for further analysis.
        """
        )

        # Export options
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📥 Download Data")
            if st.button("Download Price History (CSV)", use_container_width=True):
                csv = st.session_state.prices.to_csv()
                st.download_button(
                    label="price_history.csv",
                    data=csv,
                    file_name="price_history.csv",
                    mime="text/csv",
                )

            if st.button("Download Correlation Matrix (CSV)", use_container_width=True):
                csv = st.session_state.correlation.to_csv()
                st.download_button(
                    label="correlation_matrix.csv",
                    data=csv,
                    file_name="correlation_matrix.csv",
                    mime="text/csv",
                )

            if st.button(
                "Download Asset Statistics (CSV)", use_container_width=True
            ):
                stats_df = pd.DataFrame(
                    {
                        "Annual Return": st.session_state.annual_returns,
                        "Annual Vol": [
                            np.sqrt(st.session_state.cov_matrix.loc[t, t])
                            for t in st.session_state.annual_returns.index
                        ],
                    }
                )
                csv = stats_df.to_csv()
                st.download_button(
                    label="asset_statistics.csv",
                    data=csv,
                    file_name="asset_statistics.csv",
                    mime="text/csv",
                )

        with col2:
            st.subheader("📊 Generate Report")
            st.info(
                "💡 Use the settings in other tabs to configure your portfolio, "
                "then export results here."
            )

            if st.button("📋 Copy Results Summary", use_container_width=True):
                summary = f"""
Portfolio Optimization Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Data Period: {st.session_state.prices.index[0].date()} to {st.session_state.prices.index[-1].date()}
Risk-Free Rate: {st.session_state.rf_rate:.2%}

Asset Universe: {', '.join(st.session_state.annual_returns.index)}

Top 3 Assets by Return:
{st.session_state.annual_returns.nlargest(3).to_string()}

Top 3 Assets by Volatility:
"""
                for asset in st.session_state.annual_returns.index:
                    vol = np.sqrt(st.session_state.cov_matrix.loc[asset, asset])
                    summary += f"\n{asset}: {vol:.2%}"

                st.text_area(
                    "Copy the text below:",
                    value=summary,
                    height=300,
                    disabled=True,
                )

else:
    st.warning("⚠️ Please load data from the sidebar to begin")
