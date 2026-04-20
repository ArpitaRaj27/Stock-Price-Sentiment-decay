# News Sentiment Decay in US Equity Markets

> How quickly does the stock market digest financial news? A sector-level event-study analysis.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in--progress-yellow.svg)

## Research Question

**After a significant news event, how quickly is the sentiment reflected in the stock price, and does this speed vary by sector, market cap, and news source?**



## What's in this repo

```
sentiment-decay/
├── data/                 # raw + processed data (gitignored except schemas)
├── notebooks/            # guided tutorial notebooks, numbered in order
├── src/                  # reusable Python modules
├── dashboard/            # Streamlit app
├── reports/              # PDF writeup + figures
├── docs/                 # week-by-week guide, methodology notes
└── tests/                # unit tests for core modules
```

## Quick start

```bash
# 1. Clone + install
git clone <your-repo-url>
cd sentiment-decay
python -m venv venv && source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run with synthetic data (no API keys needed)
python -m src.pipeline --demo

# 3. Launch dashboard
streamlit run dashboard/app.py

# 4. Run notebooks in order
jupyter lab notebooks/
```

## Methodology (in one paragraph)

For each ticker-event pair, we compute cumulative abnormal returns (CAR) over a [-5, +10] day window, where abnormal return = stock return minus SPY return. Events are identified from financial news headlines scored via FinBERT, filtered to |compound sentiment| > 0.7 OR news-volume spikes > 3σ above rolling mean. We then fit an exponential decay `CAR(t) = A·exp(-λt) + C` to the mean post-event CAR, grouped by sector and source. Half-life = ln(2)/λ is our headline metric. Bootstrap (n=1000) gives 95% confidence intervals.

## Tech stack

**Data**: yfinance, Finnhub API, FRED • **NLP**: FinBERT (HuggingFace) • **Modeling**: statsmodels, scipy • **Viz**: matplotlib, plotly • **Dashboard**: Streamlit • **Testing**: pytest



## Author

**Arpita Rajapkar** — M.S. Computer Science, DePaul University

## Citations

- MacKinlay, A.C. (1997). *Event Studies in Economics and Finance.* Journal of Economic Literature.
- Bernard, V. & Thomas, J. (1989). *Post-Earnings-Announcement Drift.* Journal of Accounting Research.
- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv:1908.10063.
