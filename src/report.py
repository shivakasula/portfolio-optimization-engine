"""Generate institutional PDF investment memo."""

import io
from datetime import datetime
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_cached_data, compute_stats
from .black_litterman import bl_posterior_returns, VIEWS, VIEW_CONFIDENCES
from .backtest import backtest_bl_constrained, backtest_benchmark_60_40, compute_backtest_stats
from .optimization import plot_efficient_frontier
from .optimization_constrained import plot_allocation_comparison


def create_pdf_memo(output_path="results/portfolio_memo.pdf"):
    """Create comprehensive PDF investment memo."""
    # Load data
    prices, rf_data = load_cached_data()
    annual_returns, cov_matrix, _, rf_rate = compute_stats(prices, rf_data)
    prior, posterior, _, _, tickers = bl_posterior_returns()

    # Create PDF
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    story = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#1f4788"),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    story.append(Paragraph("Institutional Asset Allocation Engine", title_style))

    # Subtitle
    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#555555"),
        spaceAfter=12,
        alignment=TA_CENTER,
    )
    story.append(Paragraph(f"Modern Portfolio Optimization Report | {datetime.now().strftime('%B %d, %Y')}", subtitle_style))

    story.append(Spacer(1, 0.15 * inch))

    # Methodology Section
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1f4788"),
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    story.append(Paragraph("Methodology", heading_style))

    method_text = """
    This allocation engine combines <b>Markowitz portfolio optimization</b> with the <b>Black-Litterman model</b>.
    The Markowitz frontier identifies efficient risk-return tradeoffs; Black-Litterman blends market-implied returns with
    subjective views to incorporate active insights. The final portfolio is optimized under practical constraints: no
    shorting, maximum 40% allocation per asset, and annual rebalancing.
    """
    story.append(Paragraph(method_text, styles["Normal"]))
    story.append(Spacer(1, 0.1 * inch))

    # Asset Universe
    story.append(Paragraph("Asset Universe", heading_style))
    assets_text = f"Diversified portfolio of {len(tickers)} ETF proxies: {', '.join(tickers)}. Covers equities (US, international, emerging), fixed income (Treasuries, investment-grade bonds), alternatives (gold), and real estate (REITs)."
    story.append(Paragraph(assets_text, styles["Normal"]))
    story.append(Spacer(1, 0.1 * inch))

    # Subjective Views
    story.append(Paragraph("Subjective Views Incorporated", heading_style))
    views_text = "<br/>".join([
        f"• {view['outperformer']} outperforms {view['underperformer']} by {view['alpha']*100:.1f}% (confidence: {conf*100:.0f}%)"
        for view, conf in zip(VIEWS, VIEW_CONFIDENCES)
    ])
    story.append(Paragraph(views_text, styles["Normal"]))
    story.append(Spacer(1, 0.15 * inch))

    # Efficient Frontier Plot
    story.append(Paragraph("Efficient Frontier & Random Portfolios", heading_style))
    if isinstance(plot_efficient_frontier(), dict):
        # Plot already generated, just embed
        img = Image("results/efficient_frontier.png", width=6.5 * inch, height=4.0 * inch)
        story.append(img)
    story.append(Spacer(1, 0.1 * inch))

    # Black-Litterman Comparison
    story.append(Paragraph("Black-Litterman: Prior vs Posterior Returns", heading_style))
    img = Image("results/bl_comparison.png", width=6.5 * inch, height=3.0 * inch)
    story.append(img)
    story.append(Spacer(1, 0.1 * inch))

    # Allocation Comparison
    story.append(Paragraph("Portfolio Allocations", heading_style))
    img = Image("results/allocations_comparison.png", width=6.5 * inch, height=3.0 * inch)
    story.append(img)
    story.append(Spacer(1, 0.1 * inch))

    # Backtest Performance
    story.append(Paragraph("10-Year Backtest Performance (Annual Rebalancing)", heading_style))
    img = Image("results/backtest_comparison.png", width=6.5 * inch, height=3.5 * inch)
    story.append(img)

    # Backtest Stats Table
    bl_returns = backtest_bl_constrained(prices, rf_data)
    bench_returns = backtest_benchmark_60_40(prices)
    daily_returns = prices.pct_change().dropna()

    bl_stats = compute_backtest_stats(bl_returns, annual_return_series=daily_returns.std(axis=1), rf_rate=rf_rate)
    bench_stats = compute_backtest_stats(bench_returns, annual_return_series=daily_returns.std(axis=1), rf_rate=rf_rate)

    story.append(Spacer(1, 0.1 * inch))

    table_data = [
        ["Metric", "BL-Constrained", "60/40 Benchmark"],
        ["CAGR", f"{bl_stats['CAGR']:.2%}", f"{bench_stats['CAGR']:.2%}"],
        ["Annual Volatility", f"{bl_stats['Annual Vol']:.2%}", f"{bench_stats['Annual Vol']:.2%}"],
        ["Sharpe Ratio", f"{bl_stats['Sharpe']:.3f}", f"{bench_stats['Sharpe']:.3f}"],
        ["Max Drawdown", f"{bl_stats['Max Drawdown']:.2%}", f"{bench_stats['Max Drawdown']:.2%}"],
    ]

    table_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4788")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
    ])

    table = Table(table_data, colWidths=[2.0 * inch, 2.0 * inch, 2.0 * inch])
    table.setStyle(table_style)
    story.append(table)

    # Footer
    story.append(Spacer(1, 0.2 * inch))
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#999999"),
        alignment=TA_CENTER,
    )
    story.append(Paragraph(
        "This report is for informational purposes only and does not constitute investment advice. Past performance is not indicative of future results.",
        footer_style
    ))

    # Build PDF
    doc.build(story)
    print(f"✓ PDF memo generated: {output_path}")


if __name__ == "__main__":
    # Ensure plots exist
    plot_efficient_frontier()
    from .black_litterman import plot_bl_comparison
    plot_bl_comparison()
    plot_allocation_comparison()
    from .backtest import plot_backtest_comparison
    plot_backtest_comparison()

    # Generate PDF
    create_pdf_memo()
