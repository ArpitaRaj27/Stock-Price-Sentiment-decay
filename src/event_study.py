"""Event study: cumulative abnormal returns (CAR).

Method (market model):
  1. For each (ticker, event_date), estimate beta and alpha from the
     ESTIMATION_WINDOW (e.g. days [-60, -11] relative to event), regressing
     ticker returns on market returns.
  2. In the EVENT_WINDOW (e.g. [-5, +10]), predict "normal" return using
     alpha + beta * market_return.
  3. Abnormal return AR_t = actual_return_t - predicted_return_t.
  4. Cumulative abnormal return CAR(t) = sum of AR from t=0 up to t.
 
This is the standard event-study approach documented in MacKinlay (1997).

Important: the estimation window ends BEFORE the event window starts, which
prevents leakage — a subtle mistake that sinks most student finance projects.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)


def build_trading_day_index(prices: pd.DataFrame) -> pd.DataFrame:
    """Assign sequential trading-day index per ticker for window math."""
    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["t_idx"] = prices.groupby("ticker").cumcount()
    return prices


def _estimate_market_model(
    ticker_returns: pd.Series,
    market_returns: pd.Series,
) -> tuple[float, float]:
    """OLS: R_i = alpha + beta * R_m + epsilon. Returns (alpha, beta)."""
    x = market_returns.values
    y = ticker_returns.values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 20:  # need enough points
        return 0.0, 1.0
    x, y = x[mask], y[mask]
    # Closed-form OLS
    beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    alpha = y.mean() - beta * x.mean()
    return float(alpha), float(beta)


def compute_car_for_event(
    prices_long: pd.DataFrame,
    ticker: str,
    event_date: pd.Timestamp,
    pre: int = config.EVENT_WINDOW_PRE,
    post: int = config.EVENT_WINDOW_POST,
    est_start: int = config.ESTIMATION_WINDOW_START,
    est_end: int = config.ESTIMATION_WINDOW_END,
    market_proxy: str = config.MARKET_PROXY,
) -> pd.DataFrame | None:
    """Compute AR and CAR for a single event.

    Returns a DataFrame with columns [tau, date, AR, CAR], where tau is
    the trading-day offset relative to event_date (tau=0 is event day).
    Returns None if insufficient data.
    """
    tkr_df = prices_long[prices_long["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    mkt_df = prices_long[prices_long["ticker"] == market_proxy].sort_values("date").reset_index(drop=True)

    # Find the index of the event date (first trading day >= event_date)
    future = tkr_df[tkr_df["date"] >= event_date]
    if future.empty:
        return None
    t0 = future.index[0]

    # Check window bounds
    if t0 + est_start < 0 or t0 + post >= len(tkr_df):
        return None

    # Estimation window (pre-event, excludes event window)
    est = tkr_df.iloc[t0 + est_start : t0 + est_end + 1]
    mkt_est = mkt_df.set_index("date").reindex(est["date"])["return"]
    alpha, beta = _estimate_market_model(est["return"], mkt_est)

    # Event window
    evt = tkr_df.iloc[t0 + pre : t0 + post + 1].copy()
    evt["tau"] = range(pre, post + 1)
    mkt_evt = mkt_df.set_index("date").reindex(evt["date"])["return"].values
    predicted = alpha + beta * mkt_evt
    evt["AR"] = evt["return"].values - predicted

    # CAR from t=0 onwards (conventional; some studies cumulate from pre)
    evt["CAR"] = 0.0
    post_mask = evt["tau"] >= 0
    evt.loc[post_mask, "CAR"] = evt.loc[post_mask, "AR"].cumsum().values

    return evt[["tau", "date", "AR", "CAR"]].reset_index(drop=True)


def compute_all_cars(
    prices_long: pd.DataFrame,
    events: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Compute CAR for every event. Returns long DataFrame with event metadata."""
    out = []
    for _, ev in events.iterrows():
        car = compute_car_for_event(prices_long, ev["ticker"], ev["date"], **kwargs)
        if car is None:
            continue
        car["ticker"]       = ev["ticker"]
        car["event_date"]   = ev["date"]
        car["sector"]       = ev["sector"]
        car["direction"]    = ev["direction"]
        car["event_type"]   = ev["event_type"]
        car["source_tier"]  = ev["dominant_tier"]
        car["sentiment"]    = ev["max_abs_sent"]
        out.append(car)

    if not out:
        log.warning("No CARs computed — check event window against data range.")
        return pd.DataFrame()

    result = pd.concat(out, ignore_index=True)
    return result


def mean_car_curve(
    cars: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Average CAR across events, grouped by optional dims (e.g. sector, tier)."""
    group = ["tau"] + (group_cols or [])
    return (
        cars.groupby(group)
        .agg(mean_CAR=("CAR", "mean"), sem_CAR=("CAR", lambda x: x.std() / np.sqrt(len(x))),
             n=("CAR", "count"))
        .reset_index()
    )
