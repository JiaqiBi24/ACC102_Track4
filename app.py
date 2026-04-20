from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.stock_analysis import (
    DEFAULT_OUTPUT,
    build_headline_insights,
    load_dataset,
    load_metadata,
    prepare_analysis_bundle,
    rolling_pair_correlation,
)

st.set_page_config(
    page_title="Multi-Stock Interaction Dashboard",
    layout="wide",
    page_icon="chart_with_upwards_trend",
)


def load_local_files() -> tuple[pd.DataFrame | None, dict]:
    data_path = Path(DEFAULT_OUTPUT)
    if not data_path.exists():
        return None, {}
    return load_dataset(data_path), load_metadata()


def percent_format(value: float) -> str:
    return f"{value:.2%}" if pd.notna(value) else "N/A"


def number_format(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:,.0f}"


st.markdown(
    """
    <style>
      .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1320px;
      }
      div[data-testid="metric-container"] {
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: rgba(255, 255, 255, 0.72);
      }
      .insight-box {
        border-left: 3px solid #0f766e;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.65rem;
        background: rgba(15, 118, 110, 0.08);
        border-radius: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Multi-Stock Interaction Dashboard")
st.caption(
    "A Streamlit data product for comparing return co-movement, risk, and lead-lag behaviour across major US technology stocks."
)

dataframe, metadata = load_local_files()
if dataframe is None:
    st.error(
        "Local stock data was not found. Run `python3 scripts/fetch_stock_data.py` first, then refresh the page."
    )
    st.stop()

available_tickers = sorted(dataframe["Ticker"].unique().tolist())
min_date = dataframe["Date"].min().date()
max_date = dataframe["Date"].max().date()

with st.sidebar:
    st.header("Controls")
    selected_tickers = st.multiselect(
        "Choose at least two stocks",
        available_tickers,
        default=available_tickers,
    )
    start_date, end_date = st.date_input(
        "Analysis window",
        value=(max(min_date, pd.Timestamp("2023-01-01").date()), max_date),
        min_value=min_date,
        max_value=max_date,
    )
    benchmark_options = selected_tickers or available_tickers
    default_benchmark_index = benchmark_options.index("MSFT") if "MSFT" in benchmark_options else 0
    benchmark = st.selectbox("Benchmark for beta", benchmark_options, index=default_benchmark_index)
    rolling_window = st.slider("Rolling window (days)", 10, 90, 30, step=5)
    lag_days = st.slider("Maximum lead-lag horizon", 1, 10, 5)

    st.subheader("Dataset")
    st.write(f"Source: {metadata.get('source', 'Yahoo Finance via yfinance')}")
    st.write(f"Access date: {metadata.get('access_date', 'N/A')}")
    st.write(
        "Coverage: "
        f"{metadata.get('date_range', {}).get('start', min_date)} to "
        f"{metadata.get('date_range', {}).get('end', max_date)}"
    )

if len(selected_tickers) < 2:
    st.warning("Please select at least two stocks.")
    st.stop()

bundle = prepare_analysis_bundle(
    dataframe,
    selected_tickers,
    start_date=start_date,
    end_date=end_date,
    benchmark=benchmark,
    rolling_window=rolling_window,
    lag_days=lag_days,
)

headline_pair = bundle.pairwise_relationships.iloc[0]
mean_corr = bundle.correlations.where(~np.eye(len(bundle.correlations), dtype=bool)).stack().mean()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Stocks Compared", str(len(bundle.prices.columns)))
metric_2.metric("Last Trading Day", bundle.prices.index.max().strftime("%Y-%m-%d"))
metric_3.metric("Average Pair Correlation", f"{mean_corr:.2f}")
metric_4.metric("Strongest Same-Day Pair", str(headline_pair["Pair"]))

for insight in build_headline_insights(bundle):
    st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)

tab_overview, tab_interactions, tab_signals, tab_data = st.tabs(
    ["Overview", "Interaction Matrix", "Lead-Lag Signals", "Data Table"]
)

with tab_overview:
    left, right = st.columns([1.6, 1])
    with left:
        normalized_frame = bundle.normalized_prices.reset_index().melt(
            id_vars="Date",
            var_name="Ticker",
            value_name="Indexed Price",
        )
        fig = px.line(
            normalized_frame,
            x="Date",
            y="Indexed Price",
            color="Ticker",
            title="Indexed Price Performance (Start = 100)",
        )
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        summary_display = bundle.summary.copy()
        summary_display["Total Return"] = summary_display["Total Return"].map(percent_format)
        summary_display["Annualised Volatility"] = summary_display["Annualised Volatility"].map(percent_format)
        summary_display["Max Drawdown"] = summary_display["Max Drawdown"].map(percent_format)
        summary_display["Best Day"] = summary_display["Best Day"].map(percent_format)
        summary_display["Worst Day"] = summary_display["Worst Day"].map(percent_format)
        summary_display["Average Daily Volume"] = summary_display["Average Daily Volume"].map(number_format)
        summary_display["Beta vs Benchmark"] = summary_display["Beta vs Benchmark"].map(lambda value: f"{value:.2f}")
        summary_display["Mean Pair Correlation"] = summary_display["Mean Pair Correlation"].map(
            lambda value: f"{value:.2f}"
        )
        st.dataframe(summary_display, use_container_width=True, height=460)

    risk_return = bundle.summary.reset_index(names="Ticker")
    scatter = px.scatter(
        risk_return,
        x="Annualised Volatility",
        y="Total Return",
        size="Average Daily Volume",
        color="Ticker",
        hover_name="Ticker",
        title="Risk-Return Positioning",
        labels={
            "Annualised Volatility": "Annualised Volatility",
            "Total Return": "Total Return",
            "Average Daily Volume": "Average Daily Volume",
        },
        size_max=42,
    )
    scatter.update_xaxes(tickformat=".0%")
    scatter.update_yaxes(tickformat=".0%")
    st.plotly_chart(scatter, use_container_width=True)

with tab_interactions:
    heatmap = px.imshow(
        bundle.correlations,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Tealrose",
        zmin=-1,
        zmax=1,
        title="Correlation Matrix of Daily Returns",
    )
    st.plotly_chart(heatmap, use_container_width=True)

    pair_options = [f"{left} vs {right}" for left, right in combinations(bundle.daily_returns.columns, 2)]
    default_pair = pair_options[0]
    pair_choice = st.selectbox("Rolling correlation pair", pair_options, index=0)
    left_ticker, right_ticker = pair_choice.split(" vs ")
    rolling_series = rolling_pair_correlation(
        bundle.daily_returns,
        left=left_ticker,
        right=right_ticker,
        window=rolling_window,
    )
    rolling_fig = go.Figure()
    rolling_fig.add_trace(
        go.Scatter(
            x=rolling_series.index,
            y=rolling_series.values,
            mode="lines",
            name="Rolling Correlation",
        )
    )
    rolling_fig.add_hline(y=0, line_dash="dash", line_color="#666666")
    rolling_fig.update_layout(
        title=f"{rolling_window}-Day Rolling Correlation: {pair_choice}",
        yaxis_title="Correlation",
    )
    st.plotly_chart(rolling_fig, use_container_width=True)

    vol_frame = bundle.rolling_volatility.reset_index().melt(
        id_vars="Date",
        var_name="Ticker",
        value_name="Rolling Volatility",
    )
    vol_fig = px.line(
        vol_frame,
        x="Date",
        y="Rolling Volatility",
        color="Ticker",
        title=f"{rolling_window}-Day Rolling Volatility",
    )
    vol_fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(vol_fig, use_container_width=True)

with tab_signals:
    relationships = bundle.pairwise_relationships.copy()
    relationships["Same-Day Correlation"] = relationships["Same-Day Correlation"].map(lambda value: f"{value:.2f}")
    relationships["Lead-Lag Correlation"] = relationships["Lead-Lag Correlation"].map(lambda value: f"{value:.2f}")
    st.dataframe(relationships, use_container_width=True, height=320)

    drawdown_frame = bundle.drawdowns.reset_index().melt(
        id_vars="Date",
        var_name="Ticker",
        value_name="Drawdown",
    )
    drawdown_fig = px.area(
        drawdown_frame,
        x="Date",
        y="Drawdown",
        color="Ticker",
        title="Drawdown Paths",
    )
    drawdown_fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(drawdown_fig, use_container_width=True)

with tab_data:
    st.dataframe(
        dataframe[
            dataframe["Ticker"].isin(selected_tickers)
            & (dataframe["Date"].between(pd.to_datetime(start_date), pd.to_datetime(end_date)))
        ],
        use_container_width=True,
        height=520,
    )
