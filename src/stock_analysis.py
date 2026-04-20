from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
DEFAULT_OUTPUT = Path("data/raw/us_tech_stocks.csv")
DEFAULT_METADATA_OUTPUT = Path("data/raw/us_tech_stocks_metadata.json")
SOURCE_NAME = "Yahoo Finance via yfinance"


@dataclass(slots=True)
class AnalysisBundle:
    prices: pd.DataFrame
    volumes: pd.DataFrame
    normalized_prices: pd.DataFrame
    daily_returns: pd.DataFrame
    rolling_volatility: pd.DataFrame
    drawdowns: pd.DataFrame
    summary: pd.DataFrame
    correlations: pd.DataFrame
    pairwise_relationships: pd.DataFrame


def normalize_symbols(symbols: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for symbol in symbols:
        value = str(symbol).strip().upper()
        if value and value not in cleaned:
            cleaned.append(value)
    if len(cleaned) < 2:
        raise ValueError("Please provide at least two stock symbols.")
    return cleaned


def _reshape_download(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("No data was downloaded. Check the symbols or network access.")

    if not isinstance(raw.columns, pd.MultiIndex):
        raw = pd.concat({"Close": raw}, axis=1)

    stack_kwargs = {}
    if "future_stack" in pd.DataFrame.stack.__code__.co_varnames:
        stack_kwargs["future_stack"] = True

    stacked = raw.stack(level=1, **stack_kwargs).reset_index()
    stacked.columns = ["Date", "Ticker", *stacked.columns[2:]]
    rename_map = {"Adj Close": "Close"}
    stacked = stacked.rename(columns=rename_map)

    keep_columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    available = [column for column in keep_columns if column in stacked.columns]
    result = stacked[available].copy()
    result["Ticker"] = result["Ticker"].astype(str)
    result["Date"] = pd.to_datetime(result["Date"])
    numeric_columns = [column for column in result.columns if column not in {"Date", "Ticker"}]
    result[numeric_columns] = result[numeric_columns].apply(pd.to_numeric, errors="coerce")
    result = result.dropna(subset=["Close"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return result


def download_stock_data(
    symbols: Iterable[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    cleaned_symbols = normalize_symbols(symbols)
    raw = yf.download(
        cleaned_symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    return _reshape_download(raw)


def save_dataset(
    dataframe: pd.DataFrame,
    output_path: str | Path = DEFAULT_OUTPUT,
    metadata_path: str | Path = DEFAULT_METADATA_OUTPUT,
    source_name: str = SOURCE_NAME,
) -> tuple[Path, Path]:
    output = Path(output_path)
    metadata = Path(metadata_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(output, index=False)

    payload = {
        "source": source_name,
        "access_date": date.today().isoformat(),
        "row_count": int(len(dataframe)),
        "tickers": sorted(dataframe["Ticker"].unique().tolist()),
        "date_range": {
            "start": dataframe["Date"].min().strftime("%Y-%m-%d"),
            "end": dataframe["Date"].max().strftime("%Y-%m-%d"),
        },
    }
    metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output, metadata


def load_dataset(path: str | Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    dataframe = pd.read_csv(path, parse_dates=["Date"])
    dataframe = dataframe.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return dataframe


def load_metadata(path: str | Path = DEFAULT_METADATA_OUTPUT) -> dict:
    metadata_path = Path(path)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def prepare_analysis_bundle(
    dataframe: pd.DataFrame,
    tickers: Iterable[str],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    benchmark: str | None = None,
    rolling_window: int = 30,
    lag_days: int = 5,
) -> AnalysisBundle:
    selected = normalize_symbols(tickers)
    frame = dataframe[dataframe["Ticker"].isin(selected)].copy()
    frame = frame[(frame["Date"] >= pd.to_datetime(start_date)) & (frame["Date"] <= pd.to_datetime(end_date))]
    if frame.empty:
        raise ValueError("The chosen date range returned no observations.")

    prices = (
        frame.pivot(index="Date", columns="Ticker", values="Close")
        .sort_index()
        .ffill()
        .dropna(axis=1, how="all")
        .dropna(how="any")
    )
    volumes = (
        frame.pivot(index="Date", columns="Ticker", values="Volume")
        .sort_index()
        .reindex(prices.index)
        .ffill()
    )

    if prices.shape[1] < 2:
        raise ValueError("The selected stocks do not share enough common dates for comparison.")

    normalized_prices = prices.div(prices.iloc[0]).mul(100)
    daily_returns = prices.pct_change().dropna(how="any")
    rolling_volatility = daily_returns.rolling(rolling_window).std().mul(np.sqrt(252))
    drawdowns = prices.div(prices.cummax()).sub(1)

    benchmark_symbol = benchmark if benchmark in daily_returns.columns else daily_returns.columns[0]
    benchmark_series = daily_returns[benchmark_symbol]

    summary = pd.DataFrame(index=prices.columns)
    summary["Total Return"] = prices.iloc[-1].div(prices.iloc[0]).sub(1)
    summary["Annualised Volatility"] = daily_returns.std().mul(np.sqrt(252))
    summary["Average Daily Volume"] = volumes.mean()
    summary["Max Drawdown"] = drawdowns.min()
    summary["Best Day"] = daily_returns.max()
    summary["Worst Day"] = daily_returns.min()
    summary["Beta vs Benchmark"] = daily_returns.apply(
        lambda column: _calculate_beta(column, benchmark_series)
    )
    correlations = daily_returns.corr()
    summary["Mean Pair Correlation"] = correlations.apply(
        lambda row: row.drop(index=row.name).mean(),
        axis=1,
    )
    summary = summary.sort_values("Total Return", ascending=False)

    pairwise_relationships = compute_pairwise_relationships(daily_returns, max_lag=lag_days)

    return AnalysisBundle(
        prices=prices,
        volumes=volumes,
        normalized_prices=normalized_prices,
        daily_returns=daily_returns,
        rolling_volatility=rolling_volatility,
        drawdowns=drawdowns,
        summary=summary,
        correlations=correlations,
        pairwise_relationships=pairwise_relationships,
    )


def _calculate_beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.shape[0] < 2:
        return np.nan
    covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    variance = aligned.iloc[:, 1].var()
    if variance == 0:
        return np.nan
    return covariance / variance


def compute_pairwise_relationships(returns: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left, right in combinations(returns.columns, 2):
        aligned = returns[[left, right]].dropna()
        same_day_corr = aligned[left].corr(aligned[right])
        best = {
            "Pair": f"{left} vs {right}",
            "Same-Day Correlation": same_day_corr,
            "Best Lead": "Same day",
            "Follower": "Same day",
            "Lag (days)": 0,
            "Lead-Lag Correlation": same_day_corr,
            "Absolute Best Correlation": abs(same_day_corr),
        }

        for lag in range(1, max_lag + 1):
            left_leads = aligned[left].iloc[:-lag].corr(aligned[right].iloc[lag:])
            right_leads = aligned[right].iloc[:-lag].corr(aligned[left].iloc[lag:])

            if pd.notna(left_leads) and abs(left_leads) > best["Absolute Best Correlation"]:
                best.update(
                    {
                        "Best Lead": left,
                        "Follower": right,
                        "Lag (days)": lag,
                        "Lead-Lag Correlation": left_leads,
                        "Absolute Best Correlation": abs(left_leads),
                    }
                )
            if pd.notna(right_leads) and abs(right_leads) > best["Absolute Best Correlation"]:
                best.update(
                    {
                        "Best Lead": right,
                        "Follower": left,
                        "Lag (days)": lag,
                        "Lead-Lag Correlation": right_leads,
                        "Absolute Best Correlation": abs(right_leads),
                    }
                )

        rows.append(best)

    result = pd.DataFrame(rows).sort_values(
        ["Absolute Best Correlation", "Same-Day Correlation"], ascending=False
    )
    return result.reset_index(drop=True)


def rolling_pair_correlation(
    returns: pd.DataFrame,
    left: str,
    right: str,
    window: int = 30,
) -> pd.Series:
    aligned = returns[[left, right]].dropna()
    return aligned[left].rolling(window).corr(aligned[right]).dropna()


def build_headline_insights(bundle: AnalysisBundle) -> list[str]:
    summary = bundle.summary
    relationships = bundle.pairwise_relationships
    if relationships.empty:
        return []

    best_pair = relationships.iloc[0]
    most_volatile = summary["Annualised Volatility"].idxmax()
    best_performer = summary["Total Return"].idxmax()
    weakest_pair = relationships.iloc[-1]

    insights = [
        (
            f"The strongest interaction was {best_pair['Pair']} with a same-day return correlation of "
            f"{best_pair['Same-Day Correlation']:.2f}."
        ),
        (
            f"{best_performer} delivered the highest total return in the selected window, while "
            f"{most_volatile} was the most volatile stock."
        ),
        (
            f"The weakest pair was {weakest_pair['Pair']}, suggesting it contributed the most diversification "
            "within this shortlist."
        ),
    ]

    if best_pair["Lag (days)"] != 0:
        insights.append(
            f"The lead-lag scan suggests {best_pair['Best Lead']} tended to move about "
            f"{int(best_pair['Lag (days)'])} day(s) before {best_pair['Follower']}."
        )
    return insights

