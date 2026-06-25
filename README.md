# StockIntel

StockIntel is an automated stock intelligence platform that scans US equities, generates AI-powered daily recommendations with conviction-weighted portfolio allocations, tracks multi-day position performance, and delivers a professional HTML email report every morning.

The system runs as a scheduled Windows task immediately after US market close (2:00 AM IST / 4:30 PM ET) and completes a full pipeline — data collection, scoring, allocation, evaluation, research, and email dispatch — before the next trading day begins.

## Features

- **Daily Stock Scanning** — Scans 5,000+ US stocks (S&P 500, NASDAQ-100, Russell 2000) for unusual volume, pre-market movers, and technical setups via Yahoo Finance and FinViz
- **Multi-Factor AI Scoring** — Ranks stocks across 9 weighted factors: volume surge, news catalysts, momentum, low float, earnings momentum, social sentiment, technical setup, pre-market activity, and insider transactions
- **Portfolio Allocation Engine** — Computes conviction-weighted position sizes with 4 tiers (Very High / High / Medium / Speculative) and dynamic cash allocation based on average conviction strength
- **Buy/Sell Timing** — Signal-specific intraday buy windows (9:30–10:00 AM ET) and sell windows (3:30–4:00 PM ET) with configurable holding periods (1–5 trading days)
- **Multi-Day Position Tracking** — Tracks open positions with planned exit dates, unrealized P&L, days held/remaining
- **Performance Evaluation** — Evaluates predictions at +1, +2, and +5 trading day horizons with win rates, average returns, Sharpe ratio, max drawdown, and risk/reward ratio
- **Benchmark Comparison** — Compares strategy returns against SPY benchmark and reports alpha
- **Signal Attribution** — Win rate, average return, and precision/recall per signal type
- **Confidence Calibration** — Binned reliability analysis with Expected Calibration Error (ECE) and reliability diagrams
- **Post-Mortem Analysis** — Categorizes failed predictions into 9 failure types with news context lookup
- **Research Engine** — Bootstrap confidence intervals, permutation tests, Cohen's h/d effect sizes, and out-of-sample validation for weight optimization
- **Self-Improvement** — Generates strategy proposals to adjust factor weights based on statistical evidence, with approval/rejection workflow
- **HTML Dashboard** — Generates a standalone dashboard with charts, calibration diagrams, and factor performance tables
- **Rich Email Reports** — Professional HTML email with 8 sections: executive summary, top picks, portfolio allocation, open positions, closed positions, performance summary, factor attribution, and calibration insights
- **SQLite Database** — 16-table ORM schema storing snapshots, recommendations, positions, signals, news, sentiment, insider trades, earnings, performance summaries, and research artifacts
- **Windows Task Scheduler Integration** — Automated daily execution via scheduled PowerShell script

## Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │         Windows Task Scheduler              │
                          │     (Tue–Sat 2:00 AM IST = 4:30 PM ET)      │
                          └──────────────────┬──────────────────────────┘
                                             │
                                             ▼
                    ┌────────────────────────────────────────┐
                    │         StockIntelPipeline.run_full()   │
                    └────────────────────────────────────────┘
                                             │
     ┌───────────────────────────────────────┼──────────────────────────────────────────┐
     │                                       │                                          │
     ▼                                       ▼                                          ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌─────────────────────┐
│  Scanner    │  │ News Analyst │  │ Sentiment Analyst│  │Scoring Engine│  │ Performance Tracker │
│ Yahoo/FinViz │  │ yfinance     │  │ Reddit + VADER   │  │ 9-factor      │  │ Eval + Portfolio    │
│ 5145 stocks │  │ catalysts    │  │ aggregate scores │  │ weighted sum  │  │ Position tracking   │
└──────┬──────┘  └──────┬───────┘  └────────┬─────────┘  └──────┬───────┘  └─────────┬───────────┘
       │                │                   │                    │                    │
       ▼                ▼                   ▼                    ▼                    ▼
       └────────────────┴───────────────────┴────────────────────┴────────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │     Allocation Engine       │
                              │  (Kelly-derived weighting)  │
                              │  Conviction + Volatility    │
                              │  Cash Reserve Logic         │
                              └─────────────┬───────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │     Attribution Engine      │
                              │     Calibration Analyzer    │
                              │     Post-Mortem Engine      │
                              └─────────────┬───────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │   Research / Pattern Disc.  │
                              │   Improvement Engine        │
                              │   (proposal generation)     │
                              └─────────────┬───────────────┘
                                             │
                                             ▼
                    ┌────────────────────────────────────────┐
                    │  Dashboard + EmailReporter             │
                    │  (HTML report + SMTP dispatch)         │
                    └────────────────────────────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │      SQLite Database         │
                              │  16 tables, ORM (SQLAlchemy) │
                              └─────────────────────────────┘
```

## Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.9+ (developed on 3.14) |
| **Market Data** | Yahoo Finance (`yfinance`), FinViz (scraped via `requests` + `BeautifulSoup`) |
| **Alternative Data** | Reddit (`praw`), SEC EDGAR (`sec_api`), NASDAQ/NYSE FTP |
| **Sentiment** | NLTK VADER |
| **ORM** | SQLAlchemy 2.0+ with SQLite |
| **Data** | pandas, numpy |
| **Email** | smtplib (SMTP TLS, port 587) |
| **Scheduling** | Windows Task Scheduler (PowerShell `schtasks`) |
| **Configuration** | JSON (`config.json`) with `python-dotenv` for secrets |
| **Scoring Model** | 9-factor weighted linear combination with conviction tiers |
| **Statistical Testing** | Bootstrap CI, permutation tests, Cohen's h/d, out-of-sample validation |
| **AI/Learning** | Statistical self-improvement engine (research → proposals → approval) |

## Project Structure

```
stock_intel/
├── main.py                         # Pipeline orchestrator (StockIntelPipeline)
├── run.py                          # CLI entry point with argparse (20+ subcommands)
├── config/
│   └── settings.py                 # 9 dataclasses: Database, Scanner, Sentiment,
│                                   #   ScoringWeights, Email, Collector, App configs
├── db/
│   ├── models.py                   # 16 SQLAlchemy ORM models
│   └── database.py                 # DatabaseManager (~90+ CRUD methods, schema migration)
├── agents/
│   ├── base_agent.py               # Abstract base agent with logging
│   ├── scanner_agent.py            # Universe scan, technical indicators, FinViz screeners
│   ├── news_analyst.py             # yfinance news + catalyst detection
│   ├── sentiment_analyst.py        # Reddit + news sentiment aggregation
│   ├── scoring_engine.py           # 9-factor scoring, ranking, conviction
│   ├── performance_tracker.py      # EOD evaluation, position tracking, portfolio metrics
│   ├── attribution_engine.py       # Factor-level win rates, precision/recall/F1
│   ├── calibration.py              # Binned reliability, ECE, reliability diagrams
│   ├── post_mortem.py              # Failure categorization (9 types) + news context
│   ├── periodic_report.py          # Weekly/monthly report generation
│   ├── research_agent.py           # Statistical analysis, bootstrap, permutation tests
│   ├── pattern_discovery.py        # Signal combos, day-of-week, temporal patterns
│   └── improvement_engine.py       # Strategy proposal generation
├── collectors/
│   ├── yahoo_collector.py          # Yahoo Finance data: OHLCV, technicals, insider, earnings
│   ├── finviz_collector.py         # 9 FinViz screeners (unusual volume, gaps, momentum)
│   ├── reddit_collector.py         # Reddit API via praw, VADER sentiment
│   └── sec_collector.py            # SEC EDGAR Form 4 insider filings
├── utils/
│   ├── intraday.py                 # TradeSpec, buy/sell windows, entry/exit prices, market dates
│   ├── allocator.py                # Portfolio allocation engine, benchmark return
│   ├── emailer.py                  # HTML email builder, SMTP dispatcher
│   ├── dashboard.py                # Standalone HTML/JSON dashboard generator
│   └── stat_utils.py               # Statistical guards, bootstrap, validation helpers
└── data/
    ├── config.json                 # Live configuration (gitignored for secrets)
    ├── universe/
    │   ├── rebuild_universe.py     # FTP universe download (NASDAQ/NYSE)
    │   └── generate_universe.py    # Hardcoded fallback ticker lists
    └── schedule_daily.ps1          # PowerShell script to register Windows scheduled task
```

## Installation

### Prerequisites

- Python 3.9+
- Windows 10/11 (for Task Scheduler integration; Unix users can substitute cron)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/stockintel.git
cd stockintel

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK VADER lexicon (required for Reddit sentiment)
python -c "import nltk; nltk.download('vader_lexicon')"

# 5. Configure settings
# Edit stock_intel/data/config.json with your preferences

# 6. Build the stock universe
python -m stock_intel.run --scan
```

## Configuration

### config.json

All settings live in `stock_intel/data/config.json`. Key sections:

```json
{
  "scanner": {
    "min_price": 2.0,
    "max_price": 500.0,
    "min_volume": 100000,
    "max_stocks_to_scan": 6000
  },
  "weights": {
    "volume_score": 0.25,
    "news_catalyst_score": 0.20,
    "momentum_score": 0.15,
    "float_score": 0.10,
    "earnings_momentum_score": 0.10,
    "sentiment_score": 0.08,
    "technical_score": 0.07,
    "premarket_score": 0.03,
    "insider_score": 0.02
  },
  "email": {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your-email@gmail.com",
    "recipient_email": "your-email@gmail.com",
    "enabled": true
  }
}
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `STOCKINTEL_EMAIL_PASSWORD` | If `email.sender_password` empty | Gmail app password for SMTP |

A `.env` file is supported via `python-dotenv`.

### Scoring Weights

The platform scores stocks across 9 factors (default weights shown):

| Factor | Weight | Data Source |
|---|---|---|
| Volume Surge | 25% | Yahoo Finance volume ratio |
| News Catalyst | 20% | yfinance news, catalyst detection |
| Short-Term Momentum | 15% | Price momentum over lookback |
| Low Float | 10% | Shares outstanding vs float |
| Earnings Momentum | 10% | EPS surprise / earnings dates |
| Social Sentiment | 8% | Reddit + news VADER scores |
| Technical Setup | 7% | RSI, ATR, Bollinger Bands |
| Premarket Activity | 3% | Premarket change % |
| Insider Activity | 2% | SEC Form 4 filings |

Weights can be adjusted manually or via the research engine's self-improvement proposals.

## Usage

### CLI Commands

```bash
# Full pipeline (recommended daily run)
python -m stock_intel.run --full

# Individual phases
python -m stock_intel.run --scan          # Scan universe + technicals
python -m stock_intel.run --score         # Scan + score (generate recommendations)
python -m stock_intel.run --evaluate      # Post-market evaluation
python -m stock_intel.run --attribution   # Signal attribution report
python -m stock_intel.run --calibration   # Confidence calibration
python -m stock_intel.run --postmortem    # Failure analysis

# Reports
python -m stock_intel.run --dashboard     # Generate HTML dashboard
python -m stock_intel.run --weekly-report # Weekly performance
python -m stock_intel.run --monthly-report

# Research & improvement
python -m stock_intel.run --research      # Statistical research
python -m stock_intel.run --proposals     # List strategy proposals

# Proposal management
python -m stock_intel.run --approve 3                       # Accept proposal #3
python -m stock_intel.run --reject 3 --reason "Insufficient data"
python -m stock_intel.run --argue 3 --stance against --argument "Too risky"

# Utilities
python -m stock_intel.run --test-email      # Verify SMTP configuration
python -m stock_intel.run --default-weights # Reset weights to defaults
python -m stock_intel.run --verbose         # Debug logging

# Scheduled mode (for Task Scheduler / cron)
python -m stock_intel.run --schedule
```

### Automated Daily Execution

```powershell
# Run this once as Administrator to register the task:
powershell -ExecutionPolicy Bypass -File stock_intel\data\schedule_daily.ps1
```

This creates a Windows scheduled task (`StockIntelDaily`) that runs the full pipeline on Tue–Sat at 2:00 AM IST (= Mon–Fri 4:30 PM ET, immediately after US market close).

## Example Output

### Email Report Sections

The daily email contains 8 sections:

1. **Executive Summary** — Pipeline timestamp, stocks scanned, recommendations, runtime
2. **NEW PICKS** — Ranked table of top recommendations with ticker, score, direction, signals, buy/sell timing
3. **PORTFOLIO ALLOCATION** — Ticker, score, conviction label, and allocation % with dynamic cash reserve
4. **OPEN POSITIONS** — Current holdings with unrealized P&L, days held, days remaining to planned exit
5. **CLOSED POSITIONS** — Recently closed trades with realized P&L
6. **Portfolio Summary** — Cumulative return, benchmark (SPY) return, alpha, open/closed counts
7. **Performance Summary** — Win rates at +1d/+2d/+5d, avg returns, Sharpe, max drawdown, SPY comparison
8. **Factor Performance** — Win rate and avg return for each signal type

### Allocation Engine Output

```
PORTFOLIO ALLOCATION
┌────────┬───────┬────────────┬──────────────┐
│ Ticker │ Score │ Conviction │ Allocation % │
├────────┼───────┼────────────┼──────────────┤
│ AAPL   │ 0.823 │ Very High  │ 18.5%        │
│ NVDA   │ 0.771 │ Very High  │ 17.3%        │
│ AMD    │ 0.654 │ High       │ 14.7%        │
│ MSFT   │ 0.601 │ High       │ 13.5%        │
│ CRM    │ 0.552 │ High       │ 12.4%        │
│ SNOW   │ 0.487 │ Medium     │ 10.9%        │
│ PLTR   │ 0.423 │ Medium     │ 9.5%         │
│ CASH   │  —    │ Reserve    │ 3.2%         │
└────────┴───────┴────────────┴──────────────┘
```

## Performance Tracking

### Evaluation Methodology

1. **Entry**: For each recommendation, the first available trading day on or after the recommendation date is used as entry. Intraday data is preferred (buy-window average); daily close is used as fallback.
2. **Horizons**: Returns are measured at +1, +2, and +5 trading days from entry. A prediction is accurate if the best return across all three horizons exceeds the predicted direction threshold.
3. **Benchmark**: SPY return is measured over the same 5-trading-day window for comparison.

### Portfolio Metrics

- **Allocation-Weighted Return**: Each recommendation gets a conviction-weighted allocation % (summing to 100% minus cash reserve). Portfolio return is the weighted average of individual position returns.
- **Alpha**: Portfolio return minus SPY benchmark return over the same period.
- **Cash Reserve**: When average conviction is below 0.40, 15% cash reserve is held; when below 0.55, 7.5% cash reserve is held.

### Position Management

- Open positions have planned exit dates based on signal-specific holding periods (1–5 trading days, skipping weekends).
- Positions auto-close on their planned exit date.
- Closed positions are reported in the daily email and contribute to realized P&L.

### Database Storage

All evaluation results are stored in SQLite across 16 tables — every recommendation, position, snapshot, signal performance, and portfolio metric is queryable for backtesting and analysis.

## Roadmap

- [ ] Add support for additional exchanges (NYSE, AMEX)
- [ ] Implement paper trading with real-time position tracking
- [ ] Build a web dashboard (FastAPI + React)
- [ ] Add options flow data integration
- [ ] Implement machine learning models (XGBoost, LSTM) as alternative scoring engines
- [ ] Add portfolio-level risk management (VaR, stop-loss, position limits)
- [ ] Support for short-selling and inverse ETF recommendations
- [ ] Multi-user support with separate portfolio tracking
- [ ] Docker containerization for cross-platform deployment
- [ ] API-first design for third-party integration

## Disclaimer

**This project is for educational and research purposes only. It does not constitute financial advice, solicitation, or recommendation to buy or sell any security. Past performance is not indicative of future results. Stock trading involves substantial risk of loss. The authors assume no liability for any financial losses incurred from using this software. Always consult a qualified financial advisor before making investment decisions.**

## License

MIT
