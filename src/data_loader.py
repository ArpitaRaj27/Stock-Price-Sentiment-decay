"""Data loading utilities.

Two modes:
  - real: pulls from yfinance + Finnhub (requires internet, API key for news)
  - demo: generates synthetic data for offline reproducibility and testing

The demo generator is calibrated to produce realistic-looking returns
(daily stdev ~1.5%, occasional fat-tail events) and news headlines
distributed across tiers and sentiments, so downstream analysis code
can be validated end-to-end without any API access.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def load_prices_real(
    tickers: list[str],
    start: str = config.START_DATE,
    end: str = config.END_DATE,
) -> pd.DataFrame:
    """Pull adjusted close prices from yfinance.

    Returns a long DataFrame with columns: [date, ticker, adj_close, return].
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance not installed. Run: pip install yfinance") from e

    log.info(f"Downloading prices for {len(tickers)} tickers from yfinance...")
    # Include market proxy
    all_tix = list(set(tickers + [config.MARKET_PROXY]))
    df = yf.download(all_tix, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns a multi-index column frame; we want long format
    if isinstance(df.columns, pd.MultiIndex):
        closes = df["Close"]
    else:
        closes = df[["Close"]].rename(columns={"Close": all_tix[0]})

    long = closes.reset_index().melt(id_vars="Date", var_name="ticker", value_name="adj_close")
    long = long.rename(columns={"Date": "date"}).dropna()
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None)
    long = long.sort_values(["ticker", "date"])
    long["return"] = long.groupby("ticker")["adj_close"].pct_change()
    return long.reset_index(drop=True)


def load_prices_demo(
    tickers: list[str],
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic price data with realistic properties.

    - Geometric Brownian motion + occasional jumps
    - Market factor (SPY) drives cross-sectional correlation
    - Sector-level beta variation
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n_days = len(dates)

    # Simulate market return first
    market_ret = rng.normal(0.0004, 0.011, n_days)  # ~10% annual drift, 17% annual vol

    rows = []
    all_tix = list(set(tickers + [config.MARKET_PROXY]))
    sector_map = config.ticker_to_sector()

    for tkr in all_tix:
        if tkr == config.MARKET_PROXY:
            ret = market_ret
        else:
            # Sector-dependent beta
            sector = sector_map.get(tkr, "Technology")
            beta = {
                "Technology": 1.2, "Financials": 1.1, "Healthcare": 0.8,
                "Energy": 1.1, "Consumer Discretionary": 1.15, "Utilities": 0.6,
            }.get(sector, 1.0)
            idiosyncratic = rng.normal(0, 0.012, n_days)
            # Add occasional jumps (news-like events)
            jumps = rng.choice([0, 1], size=n_days, p=[0.98, 0.02]) * rng.normal(0, 0.04, n_days)
            ret = beta * market_ret + idiosyncratic + jumps

        price = 100 * np.cumprod(1 + ret)
        for d, p, r in zip(dates, price, ret):
            rows.append({"date": d, "ticker": tkr, "adj_close": p, "return": r})

    df = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    # First return is NaN per ticker
    df.loc[df.groupby("ticker").head(1).index, "return"] = np.nan
    return df


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

SAMPLE_HEADLINES_POS = [
    "{ticker} beats earnings estimates as revenue jumps",
    "Analysts upgrade {ticker} citing strong growth outlook",
    "{ticker} announces major expansion into new markets",
    "{ticker} reports record quarterly profit",
    "Institutional investors increase stake in {ticker}",
    "{ticker} launches breakthrough product line",
    "{ticker} raises full-year guidance",
]
SAMPLE_HEADLINES_NEG = [
    "{ticker} misses revenue targets, shares slide",
    "Regulatory probe weighs on {ticker}",
    "{ticker} cuts forecast citing weak demand",
    "Analysts downgrade {ticker} on margin concerns",
    "{ticker} faces lawsuit over product defect",
    "{ticker} CEO departs amid strategic disagreement",
    "Supply chain disruption hits {ticker} production",
]
SAMPLE_HEADLINES_NEU = [
    "{ticker} to present at industry conference next week",
    "{ticker} schedules earnings call for next month",
    "{ticker} files quarterly 10-Q with SEC",
    "{ticker} announces board member rotation",
    "{ticker} maintains dividend at current level",
]
SAMPLE_SOURCES = [
    ("Reuters", "tier_1"), ("Bloomberg", "tier_1"), ("Wall Street Journal", "tier_1"),
    ("Seeking Alpha", "tier_2"), ("Benzinga", "tier_2"), ("CNBC", "tier_2"),
    ("Yahoo Finance", "tier_3"), ("Motley Fool", "tier_3"), ("Zacks", "tier_3"),
]


def load_news_demo(
    tickers: list[str],
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic news with timestamps, sources, and pre-labeled sentiment.

    The pre-labeled 'true_sentiment' is used only for the demo pipeline's
    sentiment scorer shortcut. In real-data mode, sentiment is computed by FinBERT.
    """
    rng = np.random.default_rng(seed + 1)
    dates = pd.bdate_range(start, end)
    rows = []

    for tkr in tickers:
        # ~50-150 articles per ticker over 2 years
        n_articles = rng.integers(50, 150)
        article_dates = rng.choice(dates, size=n_articles, replace=True)

        for d in article_dates:
            # Random hour during trading + some after-hours
            hour = rng.integers(6, 20)
            minute = rng.integers(0, 60)
            timestamp = pd.Timestamp(d) + pd.Timedelta(hours=int(hour), minutes=int(minute))

            # Pick sentiment bucket with realistic prior
            bucket = rng.choice(["pos", "neg", "neu"], p=[0.35, 0.25, 0.40])
            if bucket == "pos":
                headline = rng.choice(SAMPLE_HEADLINES_POS).format(ticker=tkr)
                true_sent = rng.uniform(0.3, 0.95)
            elif bucket == "neg":
                headline = rng.choice(SAMPLE_HEADLINES_NEG).format(ticker=tkr)
                true_sent = -rng.uniform(0.3, 0.95)
            else:
                headline = rng.choice(SAMPLE_HEADLINES_NEU).format(ticker=tkr)
                true_sent = rng.uniform(-0.15, 0.15)

            source, tier = SAMPLE_SOURCES[rng.integers(0, len(SAMPLE_SOURCES))]

            rows.append({
                "ticker": tkr,
                "datetime": timestamp,
                "date": pd.Timestamp(d).normalize(),
                "headline": headline,
                "source": source,
                "source_tier": tier,
                "true_sentiment": true_sent,  # demo-only
            })

    df = pd.DataFrame(rows).sort_values(["ticker", "datetime"]).reset_index(drop=True)
    return df


def load_news_real(
    tickers: list[str],
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Pull company news from Finnhub.

    Note: Finnhub's free-tier `/company-news` endpoint returns up to ~1 year of data
    per call. For longer windows, chunk the date range.
    """
    import os
    import time
    import requests

    key = api_key or os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise ValueError(
            "FINNHUB_API_KEY not set. Copy .env.example to .env and add your key, "
            "or pass api_key= explicitly. Get a free key at https://finnhub.io/."
        )

    all_rows = []
    for tkr in tickers:
        url = "https://finnhub.io/api/v1/company-news"
        params = {"symbol": tkr, "from": start, "to": end, "token": key}
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            articles = r.json()
        except Exception as e:
            log.warning(f"Failed to fetch news for {tkr}: {e}")
            continue

        for a in articles:
            ts = pd.Timestamp(a.get("datetime", 0), unit="s")
            all_rows.append({
                "ticker": tkr,
                "datetime": ts,
                "date": ts.normalize(),
                "headline": a.get("headline", ""),
                "summary": a.get("summary", ""),
                "source": a.get("source", ""),
                "source_tier": config.classify_source(a.get("source", "")),
                "url": a.get("url", ""),
            })
        time.sleep(1.1)  # Rate limit: 60 calls/min on free tier

    df = pd.DataFrame(all_rows)
    if df.empty:
        log.warning("No news returned. Check your API key and date range.")
        return df
    df = df.drop_duplicates(subset=["ticker", "headline"]).sort_values(["ticker", "datetime"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_prices(demo: bool = True, **kwargs) -> pd.DataFrame:
    tickers = kwargs.pop("tickers", config.all_tickers())
    if demo:
        return load_prices_demo(tickers, **kwargs)
    return load_prices_real(tickers, **kwargs)


def load_news(demo: bool = True, **kwargs) -> pd.DataFrame:
    tickers = kwargs.pop("tickers", config.all_tickers())
    if demo:
        return load_news_demo(tickers, **kwargs)
    return load_news_real(tickers, **kwargs)
