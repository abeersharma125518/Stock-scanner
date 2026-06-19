import datetime
import logging
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy import desc
from stock_intel.agents.base_agent import BaseAgent
from stock_intel.db.database import DatabaseManager
from stock_intel.db.models import Recommendation, Stock, NewsArticle

logger = logging.getLogger(__name__)


FAILURE_CATEGORIES = {
    "unexpected_news": "Unexpected news (headline risk) moved the stock opposite to prediction",
    "premarket_reversal": "Premarket signal reversed — stock opened in the wrong direction and never recovered",
    "sector_weakness": "Sector-wide selloff dragged the stock down despite positive signals",
    "market_downturn": "Broad market downturn (SPY dropped significantly) overwhelmed stock-specific signals",
    "earnings_surprise": "Unexpected earnings result or guidance change caused the reversal",
    "gap_fade": "Stock gapped up/down at open but faded through the day (weak follow-through)",
    "low_confidence": "Low confidence score — the system was uncertain about this prediction",
    "technical_failure": "Key technical level failed — stock broke support/resistance opposite to prediction",
    "unknown": "Could not determine specific cause — may be random noise or unmodeled factor",
}


class PostMortemEngine(BaseAgent):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        super().__init__(db, config)

    def validate(self) -> bool:
        return True

    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.log_start()
        days = (context or {}).get("lookback_days", 90)
        results = {"analyzed": 0, "failed": 0, "categories": {}, "post_mortems": []}

        with self.db.session() as session:
            today = datetime.date.today()
            cutoff = today - datetime.timedelta(days=days)
            recs = session.query(Recommendation).filter(
                Recommendation.evaluated == True,
                Recommendation.prediction_accurate == False,
                Recommendation.failure_reason.is_(None),
                Recommendation.date >= cutoff,
                Recommendation.actual_close_pct.isnot(None),
            ).all()

            results["failed"] = len(recs)
            for rec in recs:
                reason, category = self._analyze_failure(session, rec)
                rec.failure_reason = reason
                rec.failure_category = category
                results["categories"][category] = results["categories"].get(category, 0) + 1
                results["post_mortems"].append(self._build_entry(rec, reason, category))
                results["analyzed"] += 1

            session.commit()

        for cat, count in sorted(results["categories"].items(), key=lambda x: x[1], reverse=True):
            pct = count / results["analyzed"] * 100 if results["analyzed"] > 0 else 0
            logger.info(f"  {FAILURE_CATEGORIES.get(cat, cat)}: {count} ({pct:.0f}%)")

        self.results = results
        self.log_end()
        return results

    def get_all_post_mortems(self, days: int = 90) -> List[Dict]:
        with self.db.session() as session:
            today = datetime.date.today()
            cutoff = today - datetime.timedelta(days=days)
            recs = session.query(Recommendation).filter(
                Recommendation.evaluated == True,
                Recommendation.prediction_accurate == False,
                Recommendation.date >= cutoff,
            ).order_by(desc(Recommendation.date)).all()
            return [self._build_entry(r, r.failure_reason or "Pending analysis", r.failure_category or "unknown") for r in recs if r.actual_close_pct is not None]

    def _analyze_failure(self, session, rec: Recommendation) -> Tuple[str, str]:
        stock = session.query(Stock).filter_by(id=rec.stock_id).first()
        ticker = stock.ticker if stock else "?"
        predicted_up = rec.predicted_direction == "up"
        actual_return = rec.actual_close_pct or 0
        total_score = rec.total_score or 0
        reasons = []

        if total_score < 0.2:
            reasons.append(f"Low confidence score ({total_score:.2f})")
            return "; ".join(reasons), "low_confidence"

        news_reason = self._check_news(session, rec, predicted_up)
        if news_reason:
            reasons.append(news_reason)

        gap_reason = self._check_gap_reversal(rec, predicted_up, actual_return)
        if gap_reason:
            reasons.append(gap_reason)

        sector_reason = self._check_sector(session, rec, predicted_up)
        if sector_reason:
            reasons.append(sector_reason)

        market_reason = self._check_market(rec, predicted_up)
        if market_reason:
            reasons.append(market_reason)

        if total_score < 0.35:
            reasons.append(f"Moderate confidence only ({total_score:.2f})")
            return "; ".join(reasons), "low_confidence"

        if not reasons:
            if actual_return < -3:
                reasons.append(f"Sharp unexpected move ({actual_return:+.1f}%) with no clear catalyst")
                return "; ".join(reasons), "unknown"
            reasons.append("No clear cause identified — may be random noise or unmodeled factor")
            return "; ".join(reasons), "unknown"

        return "; ".join(reasons), self._pick_category(reasons)

    def _check_news(self, session, rec: Recommendation, predicted_up: bool) -> Optional[str]:
        articles = session.query(NewsArticle).filter(
            NewsArticle.stock_id == rec.stock_id,
            NewsArticle.published_at >= rec.date,
            NewsArticle.published_at < rec.date + datetime.timedelta(days=1),
        ).all()
        if not articles:
            return None
        strong_negative = any(
            a.sentiment_score is not None and a.sentiment_score < -0.3 for a in articles
        )
        strong_positive = any(
            a.sentiment_score is not None and a.sentiment_score > 0.3 for a in articles
        )
        catalyst_titles = [a.title for a in articles if a.is_catalyst]
        if predicted_up and strong_negative:
            detail = f"Negative news after open: '{catalyst_titles[0][:80]}'" if catalyst_titles else "Unexpected negative news after market open"
            return detail
        if not predicted_up and strong_positive:
            detail = f"Positive news after open: '{catalyst_titles[0][:80]}'" if catalyst_titles else "Unexpected positive news after market open"
            return detail
        return None

    def _check_gap_reversal(self, rec: Recommendation, predicted_up: bool, actual_return: float) -> Optional[str]:
        pre = rec.premarket_score or 0
        gap = rec.actual_gap_pct
        if gap is None:
            return None
        if predicted_up and pre > 0.3 and gap < -1.5:
            return f"Premarket indicated up ({pre:.0%} score) but stock gapped down {gap:+.1f}% — premarket signal reversed"
        if not predicted_up and pre > 0.3 and gap > 1.5:
            return f"Premarket indicated down ({pre:.0%} score) but stock gapped up {gap:+.1f}% — premarket signal reversed"
        if predicted_up and actual_return < 0 and (rec.actual_gap_pct or 0) > 2:
            return f"Stock gapped up {gap:+.1f}% but closed {actual_return:+.1f}% — gap faded through the day"
        return None

    def _check_sector(self, session, rec: Recommendation, predicted_up: bool) -> Optional[str]:
        stock = session.query(Stock).filter_by(id=rec.stock_id).first()
        if not stock or not stock.sector:
            return None
        sector_recs = session.query(
            Recommendation
        ).join(
            Stock, Recommendation.stock_id == Stock.id
        ).filter(
            Stock.sector == stock.sector,
            Recommendation.date == rec.date,
            Recommendation.evaluated == True,
            Recommendation.actual_close_pct.isnot(None),
            Recommendation.stock_id != rec.stock_id,
        ).all()
        if not sector_recs:
            return None
        sector_returns = [r.actual_close_pct or 0 for r in sector_recs]
        avg_sector_return = sum(sector_returns) / len(sector_returns)
        if avg_sector_return < -1.5 and predicted_up:
            return f"Sector ({stock.sector}) averaged {avg_sector_return:+.1f}% that day — sector-wide weakness dragged the stock"
        if avg_sector_return > 1.5 and not predicted_up:
            return f"Sector ({stock.sector}) averaged {avg_sector_return:+.1f}% that day — sector strength lifted the stock despite signals"
        return None

    def _check_market(self, rec: Recommendation, predicted_up: bool) -> Optional[str]:
        try:
            import yfinance as yf
            spy = yf.download("SPY", start=rec.date, end=rec.date + datetime.timedelta(days=2), progress=False)
            if spy.empty:
                return None
            spy_return = ((spy["Close"].iloc[-1] - spy["Open"].iloc[0]) / spy["Open"].iloc[0]) * 100
            if spy_return < -1.5 and predicted_up:
                return f"Broad market (SPY) dropped {spy_return:+.1f}% — market-wide selloff overwhelmed stock-specific signals"
            if spy_return > 1.5 and not predicted_up:
                return f"Broad market (SPY) rallied {spy_return:+.1f}% — market-wide rally lifted the stock despite signals"
        except Exception:
            pass
        return None

    def _pick_category(self, reasons: List[str]) -> str:
        for cat_key in ["unexpected_news", "premarket_reversal", "sector_weakness", "market_downturn", "gap_fade", "earnings_surprise"]:
            if any(cat_key.replace("_", " ") in r.lower() for r in reasons):
                return cat_key
        return "unknown"

    def _build_entry(self, rec: Recommendation, reason: str, category: str) -> Dict:
        with self.db.session() as session:
            stock = session.query(Stock).filter_by(id=rec.stock_id).first()
            ticker = stock.ticker if stock else "?"
            sector = stock.sector if stock else "?"
        return {
            "date": rec.date.isoformat() if hasattr(rec.date, "isoformat") else str(rec.date),
            "ticker": ticker,
            "sector": sector,
            "predicted_direction": rec.predicted_direction,
            "expected_return": rec.predicted_gap_pct,
            "actual_return": rec.actual_close_pct,
            "total_score": rec.total_score,
            "failure_reason": reason,
            "failure_category": category,
        }

    def print_post_mortems(self, days: int = 90):
        mortems = self.get_all_post_mortems(days)
        if not mortems:
            print("\n  No failed predictions found in this period.")
            return
        print("\n" + "=" * 80)
        print(f"  WHY WAS I WRONG? — Post-Mortem Analysis (Last {days} Days)")
        print("=" * 80)
        from collections import Counter
        cats = Counter(m["failure_category"] for m in mortems)
        for cat, count in cats.most_common():
            pct = count / len(mortems) * 100
            print(f"  {FAILURE_CATEGORIES.get(cat, cat):<55} {count:>3} ({pct:>5.1f}%)")
        print("  " + "-" * 80)
        for m in mortems[:10]:
            ticker = m["ticker"]
            exp = m.get("expected_return") or 0
            act = m.get("actual_return") or 0
            cat = m["failure_category"]
            reason = m["failure_reason"][:100] if m["failure_reason"] else "?"
            print(f"\n  {ticker:<6} Exp: {exp:>+6.1f}%  Act: {act:>+6.1f}%  [{cat:<20}]")
            print(f"         {reason}")
        print("=" * 80)
