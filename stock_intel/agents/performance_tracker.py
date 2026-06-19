import datetime
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from stock_intel.agents.base_agent import BaseAgent
from stock_intel.db.database import DatabaseManager
from stock_intel.db.models import Recommendation

logger = logging.getLogger(__name__)


class PerformanceTracker(BaseAgent):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        super().__init__(db, config)

    def validate(self) -> bool:
        return True

    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.log_start()
        today = datetime.date.today()
        results = {"evaluated_count": 0, "correct_count": 0, "incorrect_count": 0,
                    "win_rate": 0.0, "avg_return": 0.0, "signal_performance": {},
                    "timestamp": datetime.datetime.now().isoformat()}

        unevaluated = self.db.get_un_evaluated_recommendations()
        logger.info(f"Evaluating {len(unevaluated)} recommendations")

        returns = []
        correct_count = 0
        signal_returns = {}

        for rec in unevaluated:
            eval_result = self._evaluate_recommendation(rec)
            if eval_result:
                correct = eval_result["accurate"]
                ret = eval_result["return_pct"]
                returns.append(ret)
                if correct:
                    correct_count += 1
                for sig in (rec.signals or []):
                    if isinstance(sig, str):
                        if sig not in signal_returns:
                            signal_returns[sig] = []
                        signal_returns[sig].append(ret)

        total = len(unevaluated)
        results["evaluated_count"] = total
        results["correct_count"] = correct_count
        results["incorrect_count"] = total - correct_count
        results["win_rate"] = round(correct_count / total, 4) if total > 0 else 0.0
        results["avg_return"] = round(np.mean(returns), 4) if returns else 0.0

        if returns:
            results["median_return"] = round(float(np.median(returns)), 4)
            results["total_return"] = round(float(np.sum(returns)), 4)
            results["max_gain"] = round(float(np.max(returns)), 4)
            results["max_loss"] = round(float(np.min(returns)), 4)
            results["std_return"] = round(float(np.std(returns)), 4)
            std = np.std(returns)
            results["sharpe_ratio"] = round(float(np.mean(returns) / std * np.sqrt(252)), 4) if std > 0 else 0.0

        previous_summaries = self.db.get_performance_history(days=365)
        if previous_summaries:
            total_correct = sum(s["correct"] for s in previous_summaries) + correct_count
            total_recs = sum(s["total_recs"] for s in previous_summaries) + total
            results["cumulative_win_rate"] = round(total_correct / total_recs, 4) if total_recs > 0 else 0.0
        else:
            results["cumulative_win_rate"] = results["win_rate"]

        if returns:
            gains = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = abs(np.mean(losses)) if losses else 1
            results["risk_reward_ratio"] = round(float(avg_gain / avg_loss), 4) if avg_loss > 0 else 0.0

        signal_perf = {}
        for sig, rets in signal_returns.items():
            sig_correct = sum(1 for r in rets if r > 0)
            signal_perf[sig] = {
                "total_occurrences": len(rets), "correct_predictions": sig_correct,
                "win_rate": round(sig_correct / len(rets), 4) if rets else 0,
                "avg_return_pct": round(float(np.mean(rets)), 4),
                "total_return_pct": round(float(np.sum(rets)), 4),
            }
        results["signal_performance"] = signal_perf

        perf_data = {
            "total_recommendations": total, "correct_predictions": correct_count,
            "incorrect_predictions": total - correct_count, "win_rate": results["win_rate"],
            "avg_return_pct": results.get("avg_return", 0),
            "median_return_pct": results.get("median_return", 0),
            "total_gain_pct": results.get("total_return", 0),
            "max_gain_pct": results.get("max_gain", 0),
            "max_loss_pct": results.get("max_loss", 0),
            "std_return_pct": results.get("std_return", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "cumulative_win_rate": results["cumulative_win_rate"],
            "risk_reward_ratio": results.get("risk_reward_ratio", 0),
        }

        recommended_recs = self.db.get_recommendations_by_date(today)
        rec_scores = [r for r in recommended_recs if r.evaluated]
        if rec_scores:
            perf_data["avg_volume_score"] = float(np.mean([r.volume_score or 0 for r in rec_scores]))
            perf_data["avg_premarket_score"] = float(np.mean([r.premarket_score or 0 for r in rec_scores]))
            perf_data["avg_sentiment_score"] = float(np.mean([r.sentiment_score or 0 for r in rec_scores]))
            perf_data["avg_news_score"] = float(np.mean([r.news_catalyst_score or 0 for r in rec_scores]))
            perf_data["avg_insider_score"] = float(np.mean([r.insider_score or 0 for r in rec_scores]))
            perf_data["avg_earnings_score"] = float(np.mean([r.earnings_score or 0 for r in rec_scores]))
            perf_data["avg_technical_score"] = float(np.mean([r.technical_score or 0 for r in rec_scores]))
            perf_data["avg_float_score"] = float(np.mean([r.float_score or 0 for r in rec_scores]))
            perf_data["avg_total_score"] = float(np.mean([r.total_score or 0 for r in rec_scores]))

        summary = self.db.save_performance_summary(today, perf_data)
        for sig, perf in signal_perf.items():
            self.db.save_signal_performance(summary.id, sig, perf)

        self.results = results
        self.log_end()
        return results

    def _evaluate_recommendation(self, rec: Recommendation) -> Optional[Dict]:
        try:
            import yfinance as yf
            ticker = rec.stock.ticker if rec.stock else None
            if not ticker:
                return None
            rec_date = rec.date
            if isinstance(rec_date, str):
                rec_date = datetime.datetime.strptime(rec_date, "%Y-%m-%d").date()

            lookahead_days = [1, 2, 5]
            rec_id = rec.id
            predicted_direction = rec.predicted_direction

            hist = yf.download(ticker, period="2wk", progress=False)
            if hist.empty:
                return None
            hist.index = pd.to_datetime(hist.index)

            rec_close = None
            for date_idx in hist.index:
                if date_idx.date() >= rec_date:
                    rec_close = hist.loc[date_idx, "Close"]
                    if isinstance(rec_close, pd.Series):
                        rec_close = rec_close.iloc[0]
                    break
            if rec_close is None:
                return None

            best_return = -999
            for days in lookahead_days:
                target_date = rec_date + datetime.timedelta(days=days)
                for date_idx in hist.index:
                    if date_idx.date() >= target_date:
                        target_close = hist.loc[date_idx, "Close"]
                        if isinstance(target_close, pd.Series):
                            target_close = target_close.iloc[0]
                        ret = ((target_close - rec_close) / rec_close) * 100
                        if ret > best_return:
                            best_return = ret
                        break
            if best_return == -999:
                return None

            predicted_up = predicted_direction == "up"
            accurate = (best_return > 0 and predicted_up) or (best_return < 0 and not predicted_up)

            with self.db.session() as session:
                db_rec = session.query(Recommendation).filter_by(id=rec_id).first()
                if db_rec:
                    db_rec.actual_close_pct = round(best_return, 2)
                    db_rec.prediction_accurate = accurate
                    db_rec.evaluated = True

            return {"accurate": accurate, "return_pct": best_return, "ticker": ticker,
                    "rec_date": rec_date, "predicted_direction": predicted_direction}
        except Exception as e:
            logger.warning(f"Eval error for rec #{rec.id}: {e}")
            return None
