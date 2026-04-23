"""Central configuration for the project.

Keep all magic numbers, tickers, and paths here so they're easy to change.
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure dirs exist
for d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Universe: 30 tickers, 6 sectors x 5 each
# Chosen for liquidity, sector coverage, and sufficient news volume.
UNIVERSE = {
    "Technology":             ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "Financials":             ["JPM", "BAC", "GS", "MS", "WFC"],
    "Healthcare":             ["JNJ", "PFE", "UNH", "MRK", "ABBV"],
    "Energy":                 ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
    "Utilities":              ["NEE", "DUK", "SO", "AEP", "EXC"],
}

def all_tickers() -> list[str]:
    return [t for tickers in UNIVERSE.values() for t in tickers]

def ticker_to_sector() -> dict[str, str]:
    return {t: sector for sector, tickers in UNIVERSE.items() for t in tickers}

# Date range
START_DATE = "2024-01-01"
END_DATE   = "2025-12-31"

# Market proxy for abnormal-return computation
MARKET_PROXY = "SPY"

# Event study window (trading days relative to event)
EVENT_WINDOW_PRE  = -5
EVENT_WINDOW_POST = 10

# Estimation window for beta (used in market-model abnormal returns)
ESTIMATION_WINDOW_START = -60
ESTIMATION_WINDOW_END   = -11  # gap before event to avoid leakage

# Event-definition thresholds
SENTIMENT_THRESHOLD    = 0.7   # |compound| above this = significant
VOLUME_SPIKE_SIGMA     = 3.0   # stddevs above 30-day rolling mean
VOLUME_ROLLING_WINDOW  = 30

# FinBERT
FINBERT_MODEL = "ProsusAI/finbert"

# Bootstrap
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED       = 42

# News source tiers (for tier comparison)
SOURCE_TIERS = {
    "tier_1": ["Reuters", "Bloomberg", "Wall Street Journal", "Financial Times"],
    "tier_2": ["Seeking Alpha", "Benzinga", "MarketWatch", "CNBC"],
    "tier_3": ["Yahoo", "Zacks", "Motley Fool", "InvestorPlace"],
}

def classify_source(source: str) -> str:
    """Map an arbitrary source string to tier_1/2/3/unknown."""
    if not source:
        return "unknown"
    s = source.lower()
    for tier, names in SOURCE_TIERS.items():
        if any(name.lower() in s for name in names):
            return tier
    return "unknown"

