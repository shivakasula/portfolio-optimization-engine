"""Main entry point: orchestrate full pipeline end-to-end."""

import sys
from .data import fetch_all_data
from .optimization import plot_efficient_frontier
from .black_litterman import plot_bl_comparison, print_bl_summary
from .optimization_constrained import plot_allocation_comparison, print_allocation_summary
from .backtest import plot_backtest_comparison, print_backtest_summary
from .report import create_pdf_memo


def main():
    """Run full pipeline."""
    print("\n" + "="*80)
    print("MODERN PORTFOLIO OPTIMIZATION ENGINE - FULL PIPELINE")
    print("="*80)

    # Phase 1: Data & Stats
    print("\n[PHASE 1] Fetching data and computing statistics...")
    try:
        fetch_all_data(use_cache=True)
        print("✓ Phase 1 complete")
    except Exception as e:
        print(f"✗ Phase 1 failed: {e}")
        sys.exit(1)

    # Phase 2: Markowitz
    print("\n[PHASE 2] Computing efficient frontier...")
    try:
        plot_efficient_frontier()
        print("✓ Phase 2 complete")
    except Exception as e:
        print(f"✗ Phase 2 failed: {e}")
        sys.exit(1)

    # Phase 3: Black-Litterman
    print("\n[PHASE 3] Black-Litterman model...")
    try:
        plot_bl_comparison()
        print_bl_summary()
        print("✓ Phase 3 complete")
    except Exception as e:
        print(f"✗ Phase 3 failed: {e}")
        sys.exit(1)

    # Phase 4: Constrained Optimization
    print("\n[PHASE 4] Constrained re-optimization...")
    try:
        plot_allocation_comparison()
        print_allocation_summary()
        print("✓ Phase 4 complete")
    except Exception as e:
        print(f"✗ Phase 4 failed: {e}")
        sys.exit(1)

    # Phase 5: Backtesting
    print("\n[PHASE 5] Backtesting...")
    try:
        plot_backtest_comparison()
        print_backtest_summary()
        print("✓ Phase 5 complete")
    except Exception as e:
        print(f"✗ Phase 5 failed: {e}")
        sys.exit(1)

    # Phase 6: PDF Report
    print("\n[PHASE 6] Generating PDF report...")
    try:
        create_pdf_memo()
        print("✓ Phase 6 complete")
    except Exception as e:
        print(f"✗ Phase 6 failed: {e}")
        sys.exit(1)

    print("\n" + "="*80)
    print("✓ ALL PHASES COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - data/prices.csv (cached price data)")
    print("  - data/risk_free_rate.csv (cached risk-free rates)")
    print("  - results/efficient_frontier.png")
    print("  - results/bl_comparison.png")
    print("  - results/allocations_comparison.png")
    print("  - results/backtest_comparison.png")
    print("  - results/portfolio_memo.pdf (INVESTMENT MEMO)")
    print("\nNext steps:")
    print("  1. Review results/portfolio_memo.pdf")
    print("  2. Edit views in src/black_litterman.py and re-run")
    print("  3. Adjust constraints in src/optimization_constrained.py as needed")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
