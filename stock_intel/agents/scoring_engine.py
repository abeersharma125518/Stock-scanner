import datetime
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np
from stock_intel.agents.base_agent import BaseAgent
from stock_intel.db.database import DatabaseManager
from stock_intel.config.settings import CONFIG

logger = logging.getLogger(__name__)


class ScoringEngine(BaseAgent):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        super().__init__(db, config)
        self.weights = CONFIG.weights

    def validate(self) -> bool:
        return True

    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.log_start()
        context = context or {}
        today = datetime.date.today()
        scanner_results = context.get("scanner_results", {})
        news_results = context.get("news_results", {})
        sentiment_results = context.get("sentiment_results", {})
        all_snapshots = scanner_results.get("all_snapshots", {})

        scores = {}
        for ticker, snapshot in all_snapshots.items():
            try:
                score = self._score_stock(ticker, snapshot, scanner_results, news_results, sentiment_results)
                if score is not None and score["total_score"] > 0:
                    scores[ticker] = score
            except Exception as e:
                logger.warning(f"Scoring error for {ticker}: {e}")
                continue

        ranked = sorted(scores.values(), key=lambda x: x["total_score"], reverse=True)
        top_n = ranked[:CONFIG.top_n_recommendations]
        recommendations = []

        for i, score_data in enumerate(top_n):
            ticker = score_data["ticker"]
            stock_id = self.db.get_or_create_stock_id(ticker)

            signals = score_data.get("signals", [])
            key_drivers = self._generate_key_drivers(score_data)
            explanation = self._generate_explanation(score_data)

            rec_data = {
                "total_score": score_data["total_score"],
                "volume_score": score_data.get("volume_score", 0),
                "premarket_score": score_data.get("premarket_score", 0),
                "sentiment_score": score_data.get("sentiment_score", 0),
                "news_catalyst_score": score_data.get("news_catalyst_score", 0),
                "insider_score": score_data.get("insider_score", 0),
                "earnings_score": score_data.get("earnings_score", 0),
                "technical_score": score_data.get("technical_score", 0),
                "momentum_score": score_data.get("momentum_score", 0),
                "float_score": score_data.get("float_score", 0),
                "signals": signals, "key_drivers": key_drivers,
                "explanation": explanation,
                "predicted_direction": "up" if score_data.get("direction_score", 0) > 0 else "down",
            }

            predicted_pct = score_data.get("total_score", 0) * 0.5
            if score_data.get("premarket_change", 0):
                predicted_pct = max(predicted_pct, abs(score_data.get("premarket_change", 0)) * 0.3)
            rec_data["predicted_gap_pct"] = round(predicted_pct, 2)

            self.db.save_recommendation(stock_id, today, i + 1, rec_data)
            self.db.save_signal(stock_id, today, "recommendation", float(score_data["total_score"]),
                                {"rank": i + 1, "score": score_data["total_score"], "explanation": explanation})

            recommendations.append({
                "rank": i + 1, "ticker": ticker, "total_score": score_data["total_score"],
                "explanation": explanation, "key_drivers": key_drivers, "signals": signals,
                "component_scores": {k: score_data[k] for k in ["volume_score","premarket_score","sentiment_score",
                    "news_catalyst_score","insider_score","earnings_score","technical_score","momentum_score","float_score"]},
            })

        self.results = {
            "recommendations": recommendations, "total_scored": len(scores),
            "ranked_list": ranked, "date": today.isoformat(),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        logger.info(f"Scored {len(scores)} stocks, top {len(recommendations)} recommended")
        self.log_end()
        return self.results

    def _score_stock(self, ticker: str, snapshot: Dict, scanner_results: Dict,
                      news_results: Dict, sentiment_results: Dict) -> Optional[Dict]:
        volume_score = self._calc_volume_score(snapshot)
        premarket_score = self._calc_premarket_score(snapshot)
        sentiment_score = self._calc_sentiment_score(ticker, sentiment_results)
        news_score = self._calc_news_score(ticker, news_results)
        insider_score = self._calc_insider_score(ticker)
        earnings_score = self._calc_earnings_score(ticker)
        technical_score = self._calc_technical_score(snapshot)
        momentum_score = self._calc_momentum_score(snapshot)
        float_score = self._calc_float_score(snapshot)

        direction_score = (
            volume_score * 0.1 + premarket_score * 0.2 + sentiment_score * 0.2 +
            news_score * 0.15 + insider_score * 0.1 + earnings_score * 0.05 +
            technical_score * 0.1 + momentum_score * 0.1
        )

        total_score = (
            volume_score * self.weights.volume_score +
            premarket_score * self.weights.premarket_score +
            sentiment_score * self.weights.sentiment_score +
            news_score * self.weights.news_catalyst_score +
            insider_score * self.weights.insider_score +
            earnings_score * self.weights.earnings_momentum_score +
            technical_score * self.weights.technical_score +
            momentum_score * self.weights.momentum_score +
            float_score * self.weights.float_score
        )

        signals = self._collect_signals(snapshot, volume_score, premarket_score, sentiment_score,
                                         news_score, insider_score, earnings_score, technical_score,
                                         momentum_score, float_score)

        return {
            "ticker": ticker, "total_score": round(total_score, 4),
            "volume_score": round(volume_score, 4), "premarket_score": round(premarket_score, 4),
            "premarket_change": snapshot.get("premarket_change_pct"),
            "sentiment_score": round(sentiment_score, 4),
            "news_catalyst_score": round(news_score, 4), "insider_score": round(insider_score, 4),
            "earnings_score": round(earnings_score, 4), "technical_score": round(technical_score, 4),
            "momentum_score": round(momentum_score, 4), "float_score": round(float_score, 4),
            "direction_score": direction_score, "signals": signals,
            "price": snapshot.get("close_price") or snapshot.get("open_price"),
            "volume": snapshot.get("volume"), "volume_ratio": snapshot.get("volume_ratio"),
            "market_cap": snapshot.get("market_cap"),
        }

    def _calc_volume_score(self, snapshot: Dict) -> float:
        vol_ratio = snapshot.get("volume_ratio", 1.0)
        if vol_ratio <= 1.0:
            return 0.0
        return min((vol_ratio - 1.0) / 5.0, 1.0)

    def _calc_premarket_score(self, snapshot: Dict) -> float:
        pm = snapshot.get("premarket_change_pct")
        if pm is None:
            return 0.0
        return min(abs(pm) / 10.0, 1.0) * (1.0 if pm > 0 else 0.3)

    def _calc_sentiment_score(self, ticker: str, sentiment_results: Dict) -> float:
        scores = sentiment_results.get("sentiment_scores", {})
        ticker_data = scores.get(ticker)
        if not ticker_data:
            return 0.0
        combined = ticker_data.get("combined_sentiment", 0)
        return max(0, min((combined + 1.0) / 2.0, 1.0))

    def _calc_news_score(self, ticker: str, news_results: Dict) -> float:
        catalysts = news_results.get("catalyst_stocks", [])
        for c in catalysts:
            if c["ticker"] == ticker:
                return c["confidence"]
        news_for_ticker = news_results.get("all_news", {}).get(ticker, [])
        if not news_for_ticker:
            return 0.0
        catalyst_count = sum(1 for n in news_for_ticker if n.get("is_catalyst"))
        avg_sent = np.mean([n.get("sentiment_score", 0) for n in news_for_ticker]) if news_for_ticker else 0
        score = min(catalyst_count / 5.0, 0.5) + max(0, avg_sent) * 0.5
        return min(score, 1.0)

    def _calc_insider_score(self, ticker: str) -> float:
        try:
            from stock_intel.collectors.sec_collector import SECCollector
            sec = SECCollector()
            transactions = sec.fetch_insider_transactions(ticker, days_back=30)
            if not transactions:
                return 0.0
            buy_count = sum(1 for t in transactions if "buy" in t.get("transaction_type","").lower() or t.get("shares_traded",0) > 0)
            sell_count = sum(1 for t in transactions if "sell" in t.get("transaction_type","").lower() or t.get("shares_traded",0) < 0)
            total = buy_count + sell_count
            if total == 0:
                return 0.0
            ratio = buy_count / total
            return ratio if ratio > 0.5 else 1.0 - ratio
        except Exception:
            return 0.0

    def _calc_earnings_score(self, ticker: str) -> float:
        try:
            from stock_intel.collectors.sec_collector import SECCollector
            sec = SECCollector()
            earnings = sec.fetch_earnings_calendar(days_ahead=14)
            for e in earnings:
                if e.get("ticker", "").upper() == ticker.upper():
                    return 0.5
        except Exception:
            pass
        try:
            import yfinance as yf
            import pandas as pd
            stock = yf.Ticker(ticker)
            if stock.calendar:
                cal = stock.calendar
                if "Earnings Date" in cal:
                    ed = cal["Earnings Date"]
                    if isinstance(ed, (datetime.datetime, pd.Timestamp)):
                        days_until = (ed.date() - datetime.date.today()).days
                        if 0 <= days_until <= 14:
                            return max(0, 1.0 - days_until / 14.0)
        except Exception:
            pass
        return 0.0

    def _calc_technical_score(self, snapshot: Dict) -> float:
        score = 0.0
        rsi = snapshot.get("rsi_14")
        if rsi is not None:
            if 30 <= rsi <= 70:
                score += 0.3 * (1.0 - abs(rsi - 50) / 50.0)
            elif rsi < 30:
                score += 0.4
            elif rsi > 70:
                score += 0.1
        sma_20 = snapshot.get("sma_20")
        price = snapshot.get("close_price") or snapshot.get("open_price")
        if price and sma_20 and sma_20 > 0:
            score += 0.3 if price > sma_20 else 0.1
        vol_ratio = snapshot.get("volume_ratio", 1.0)
        if vol_ratio > 1.5:
            score += 0.3
        return min(score, 1.0)

    def _calc_momentum_score(self, snapshot: Dict) -> float:
        pm = snapshot.get("premarket_change_pct")
        if pm is None:
            return 0.0
        if pm > 0:
            return min(pm / 15.0, 1.0)
        return 0.0

    def _calc_float_score(self, snapshot: Dict) -> float:
        mcap = snapshot.get("market_cap")
        if mcap is not None and mcap > 0:
            if mcap < 300_000_000:
                return 1.0
            elif mcap < 1_000_000_000:
                return 0.8
            elif mcap < 2_000_000_000:
                return 0.6
            elif mcap < 10_000_000_000:
                return 0.4
            elif mcap < 50_000_000_000:
                return 0.2
            else:
                return 0.0
        float_shares = snapshot.get("float_shares")
        if float_shares is not None and float_shares > 0:
            if float_shares < 20_000_000:
                return 1.0
            elif float_shares < 50_000_000:
                return 0.7
            elif float_shares < 100_000_000:
                return 0.4
            else:
                return 0.1
        return 0.0

    def _collect_signals(self, snapshot: Dict, volume_score: float, premarket_score: float,
                          sentiment_score: float, news_score: float, insider_score: float,
                          earnings_score: float, technical_score: float, momentum_score: float,
                          float_score: float = 0.0) -> List[str]:
        signals = []
        if volume_score > 0.3:
            signals.append("unusual_volume")
        if snapshot.get("volume_ratio", 1.0) > 3.0:
            signals.append("extreme_volume")
        pm = snapshot.get("premarket_change_pct")
        if pm is not None:
            if pm > 3.0:
                signals.append("premarket_mover_up")
            elif pm < -3.0:
                signals.append("premarket_mover_down")
        if sentiment_score > 0.6:
            signals.append("positive_sentiment")
        if news_score > 0.5:
            signals.append("news_catalyst")
        if insider_score > 0.6:
            signals.append("insider_buying")
        if earnings_score > 0.3:
            signals.append("upcoming_earnings")
        rsi = snapshot.get("rsi_14")
        if rsi is not None:
            if rsi < 30:
                signals.append("oversold")
            elif rsi > 70:
                signals.append("overbought")
        if technical_score > 0.6:
            signals.append("technical_bullish")
        if float_score > 0.5:
            signals.append("low_float" if float_score > 0.7 else "small_cap")
        return signals

    @staticmethod
    def _generate_key_drivers(score_data: Dict) -> str:
        parts = []
        if score_data.get("volume_score", 0) > 0.3:
            parts.append(f"Volume surge ({score_data['volume_score']:.0%})")
        if score_data.get("premarket_score", 0) > 0.3:
            parts.append(f"Premarket momentum ({score_data.get('premarket_change', 0):.1f}%)")
        if score_data.get("sentiment_score", 0) > 0.5:
            parts.append("Strong sentiment")
        if score_data.get("news_catalyst_score", 0) > 0.3:
            parts.append("News catalyst")
        if score_data.get("insider_score", 0) > 0.5:
            parts.append("Insider activity")
        if score_data.get("technical_score", 0) > 0.5:
            parts.append("Technical setup")
        if score_data.get("momentum_score", 0) > 0.5:
            parts.append(f"Momentum ({score_data.get('premarket_change', 0):.1f}%)")
        if score_data.get("float_score", 0) > 0.5:
            parts.append("Low float / small cap")
        if not parts:
            parts.append("Balanced signals")
        return ", ".join(parts)

    @staticmethod
    def _generate_explanation(score_data: Dict) -> str:
        ticker = score_data["ticker"]
        total = score_data["total_score"]
        signals = score_data.get("signals", [])
        price = score_data.get("price", 0)
        signal_desc = ", ".join(signals[:5]) if signals else "baseline monitoring"
        return (f"{ticker} (${price:.2f}) scored {total:.3f}. "
                f"Key signals: {signal_desc}. "
                f"Driven by {score_data.get('key_drivers', 'mixed factors')}.")
