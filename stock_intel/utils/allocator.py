import datetime
import logging
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CONVICTION_THRESHOLDS = [
    ("Very High", 0.75),
    ("High", 0.55),
    ("Medium", 0.35),
    ("Speculative", 0.0),
]

MAX_SIGNAL_BONUS = 0.15
VOLATILITY_WEIGHT = 0.15
SCORE_WEIGHT = 0.55
SIGNAL_WEIGHT = 0.15
CONVICTION_CASH_THRESHOLD = 0.40
CASH_ALLOCATION_BASE = 15.0


def _get_volatility(ticker: str) -> Optional[float]:
    try:
        hist = yf.download(ticker, period="1mo", progress=False)
        if hist.empty or len(hist) < 5:
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            close_series = hist["Close"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series[ticker]
        else:
            close_series = hist["Close"]
        daily_rets = close_series.pct_change().dropna()
        if len(daily_rets) < 4:
            return None
        return float(np.std(daily_rets))
    except Exception:
        return None


def _signal_strength(signals: Optional[List[str]]) -> float:
    if not signals:
        return 0.0
    count = len(signals)
    if count >= 4:
        return 1.0
    if count >= 3:
        return 0.8
    if count >= 2:
        return 0.5
    return 0.2


def _conviction_label(composite: float) -> str:
    for label, threshold in CONVICTION_THRESHOLDS:
        if composite >= threshold:
            return label
    return "Speculative"


def compute_allocations(
    recommendations: List[Dict],
    today: Optional[datetime.date] = None,
) -> Tuple[List[Dict], float, str]:
    today = today or datetime.date.today()
    scored = []

    for rec in recommendations:
        ticker = rec.get("ticker", "?")
        total_score = rec.get("total_score", 0) or 0
        signals = rec.get("signals") or []

        vol = _get_volatility(ticker)
        if vol is not None:
            vol_factor = max(0.3, min(1.0, 1.0 - vol * 5.0))
        else:
            vol_factor = 0.7

        sig_str = _signal_strength(signals)
        sig_bonus = sig_str * MAX_SIGNAL_BONUS

        composite = total_score * SCORE_WEIGHT + sig_bonus + vol_factor * VOLATILITY_WEIGHT
        composite = min(max(composite, 0.0), 1.0)

        label = _conviction_label(composite)

        scored.append({
            "ticker": ticker,
            "total_score": total_score,
            "signal_strength": round(sig_str, 2),
            "volatility": vol,
            "conviction": label,
            "composite": composite,
            "_rec_ref": rec,
        })

    total_composite = max(sum(s["composite"] for s in scored), 0.01)
    avg_conviction = sum(s["composite"] for s in scored) / len(scored) if scored else 0.0

    cash_pct = 0.0
    if avg_conviction < CONVICTION_CASH_THRESHOLD:
        cash_pct = CASH_ALLOCATION_BASE
    elif avg_conviction < CONVICTION_CASH_THRESHOLD + 0.15:
        cash_pct = CASH_ALLOCATION_BASE * 0.5

    allocable = 100.0 - cash_pct
    for s in scored:
        raw = (s["composite"] / total_composite) * allocable
        s["allocation_pct"] = round(raw, 1)
    allocated_sum = sum(s["allocation_pct"] for s in scored)

    if allocated_sum != allocable and len(scored) > 0:
        diff = round(allocable - allocated_sum, 1)
        scored[-1]["allocation_pct"] = round(scored[-1]["allocation_pct"] + diff, 1)

    allocations = []
    for s in scored:
        allocations.append({
            "ticker": s["ticker"],
            "total_score": s["total_score"],
            "signal_strength": s["signal_strength"],
            "volatility": s["volatility"],
            "conviction": s["conviction"],
            "allocation_pct": s["allocation_pct"],
        })

    return allocations, cash_pct, today.isoformat()


def compute_portfolio_return(allocations: List[Dict], positions: List) -> float:
    total_weight = sum(a["allocation_pct"] for a in allocations)
    if total_weight == 0:
        return 0.0
    weighted_sum = 0.0
    for a in allocations:
        pos = next((p for p in positions if p.get("ticker") == a["ticker"]), None)
        ret = pos.get("current_return", 0) if pos else 0
        if ret is None:
            ret = 0
        weighted_sum += a["allocation_pct"] * ret
    return weighted_sum / total_weight


def fetch_benchmark_return(benchmark: str = "SPY", days: int = 5) -> Optional[float]:
    try:
        hist = yf.download(benchmark, period=f"{max(days, 5)}d", progress=False)
        if hist.empty or len(hist) < 2:
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            close_series = hist["Close"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series[benchmark]
        else:
            close_series = hist["Close"]
        start = float(close_series.iloc[0])
        end = float(close_series.iloc[-1])
        return round((end - start) / start * 100, 2)
    except Exception:
        return None
