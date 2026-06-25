import datetime
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import yfinance as yf
from stock_intel.agents.base_agent import BaseAgent
from stock_intel.db.database import DatabaseManager
from stock_intel.db.models import Recommendation
from stock_intel.utils.intraday import (
    TradeSpec, calc_entry_price, calc_exit_price, calc_current_price,
    next_market_date, planned_exit,
)
from stock_intel.utils.allocator import (
    compute_allocations, compute_portfolio_return, fetch_benchmark_return,
)

logger = logging.getLogger(__name__)


class PerformanceTracker(BaseAgent):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        super().__init__(db, config)

    def validate(self) -> bool:
        return True

    def _fetch_spy_return(self, rec_date: datetime.date) -> Optional[float]:
        try:
            hist = yf.download("SPY", period="2wk", progress=False)
            if hist.empty:
                return None
            hist.index = pd.to_datetime(hist.index)
            spy_entry = None
            for date_idx in hist.index:
                if date_idx.date() >= rec_date:
                    spy_entry = hist.loc[date_idx, "Close"]
                    if isinstance(spy_entry, pd.Series):
                        spy_entry = spy_entry.iloc[0]
                    break
            if spy_entry is None:
                return None
            target_date = rec_date + datetime.timedelta(days=5)
            spy_exit = None
            for date_idx in hist.index:
                if date_idx.date() >= target_date:
                    spy_exit = hist.loc[date_idx, "Close"]
                    if isinstance(spy_exit, pd.Series):
                        spy_exit = spy_exit.iloc[0]
                    break
            if spy_exit is None:
                return None
            return round(((spy_exit - spy_entry) / spy_entry) * 100, 2)
        except Exception:
            return None

    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.log_start()
        today = datetime.date.today()

        unevaluated = self.db.get_un_evaluated_recommendations()
        logger.info(f"Evaluating {len(unevaluated)} recommendations")

        returns = []
        returns_1d = []
        returns_2d = []
        returns_5d = []
        correct_count = 0
        correct_1d = 0
        correct_2d = 0
        correct_5d = 0
        signal_returns = {}
        all_drawdowns = []
        spy_returns = []

        for rec in unevaluated:
            eval_result = self._evaluate_recommendation(rec)
            if eval_result:
                correct = eval_result["accurate"]
                ret = eval_result["return_pct"]
                returns.append(ret)
                if correct:
                    correct_count += 1

                r1d = eval_result.get("return_1d")
                r2d = eval_result.get("return_2d")
                r5d = eval_result.get("return_5d")
                if r1d is not None:
                    returns_1d.append(r1d)
                    if eval_result.get("accurate_1d"):
                        correct_1d += 1
                if r2d is not None:
                    returns_2d.append(r2d)
                    if eval_result.get("accurate_2d"):
                        correct_2d += 1
                if r5d is not None:
                    returns_5d.append(r5d)
                    if eval_result.get("accurate_5d"):
                        correct_5d += 1

                for sig in (rec.signals or []):
                    if isinstance(sig, str):
                        if sig not in signal_returns:
                            signal_returns[sig] = []
                        signal_returns[sig].append(ret)

                if eval_result.get("max_drawdown") is not None:
                    all_drawdowns.append(eval_result["max_drawdown"])
                if eval_result.get("spy_return") is not None:
                    spy_returns.append(eval_result["spy_return"])

        total = len(unevaluated)
        results = {
            "evaluated_count": total,
            "correct_count": correct_count,
            "incorrect_count": total - correct_count,
            "win_rate": round(correct_count / total, 4) if total > 0 else 0.0,
            "win_rate_1d": round(correct_1d / len(returns_1d), 4) if returns_1d else 0.0,
            "win_rate_2d": round(correct_2d / len(returns_2d), 4) if returns_2d else 0.0,
            "win_rate_5d": round(correct_5d / len(returns_5d), 4) if returns_5d else 0.0,
            "avg_return": round(np.mean(returns), 4) if returns else 0.0,
            "avg_return_1d": round(float(np.mean(returns_1d)), 4) if returns_1d else 0.0,
            "avg_return_2d": round(float(np.mean(returns_2d)), 4) if returns_2d else 0.0,
            "avg_return_5d": round(float(np.mean(returns_5d)), 4) if returns_5d else 0.0,
        }

        if returns:
            results["median_return"] = round(float(np.median(returns)), 4)
            results["total_return"] = round(float(np.sum(returns)), 4)
            results["max_gain"] = round(float(np.max(returns)), 4)
            results["max_loss"] = round(float(np.min(returns)), 4)
            results["std_return"] = round(float(np.std(returns)), 4)
            std = np.std(returns)
            results["sharpe_ratio"] = round(float(np.mean(returns) / std * np.sqrt(252)), 4) if std > 0 else 0.0

        results["max_drawdown"] = round(float(np.min(all_drawdowns)), 4) if all_drawdowns else 0.0
        results["spy_return_pct"] = round(float(np.mean(spy_returns)), 4) if spy_returns else 0.0

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

        recommended_recs = self.db.get_recommendations_by_date(today)
        rec_scores = [r for r in recommended_recs if r.evaluated]
        if rec_scores:
            results["avg_volume_score"] = float(np.mean([r.volume_score or 0 for r in rec_scores]))
            results["avg_premarket_score"] = float(np.mean([r.premarket_score or 0 for r in rec_scores]))
            results["avg_sentiment_score"] = float(np.mean([r.sentiment_score or 0 for r in rec_scores]))
            results["avg_news_score"] = float(np.mean([r.news_catalyst_score or 0 for r in rec_scores]))
            results["avg_insider_score"] = float(np.mean([r.insider_score or 0 for r in rec_scores]))
            results["avg_earnings_score"] = float(np.mean([r.earnings_score or 0 for r in rec_scores]))
            results["avg_technical_score"] = float(np.mean([r.technical_score or 0 for r in rec_scores]))
            results["avg_float_score"] = float(np.mean([r.float_score or 0 for r in rec_scores]))
            results["avg_total_score"] = float(np.mean([r.total_score or 0 for r in rec_scores]))

        summary = self.db.save_performance_summary(today, {
            "total_recommendations": total, "correct_predictions": correct_count,
            "incorrect_predictions": total - correct_count, "win_rate": results["win_rate"],
            "win_rate_1d": results["win_rate_1d"],
            "win_rate_2d": results["win_rate_2d"],
            "win_rate_5d": results["win_rate_5d"],
            "avg_return_pct": results.get("avg_return", 0),
            "avg_return_1d": results.get("avg_return_1d", 0),
            "avg_return_2d": results.get("avg_return_2d", 0),
            "avg_return_5d": results.get("avg_return_5d", 0),
            "median_return_pct": results.get("median_return", 0),
            "total_gain_pct": results.get("total_return", 0),
            "max_gain_pct": results.get("max_gain", 0),
            "max_loss_pct": results.get("max_loss", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "std_return_pct": results.get("std_return", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "cumulative_win_rate": results["cumulative_win_rate"],
            "risk_reward_ratio": results.get("risk_reward_ratio", 0),
            "spy_return_pct": results.get("spy_return_pct", 0),
        })
        for sig_name, perf in signal_perf.items():
            self.db.save_signal_performance(summary.id, sig_name, perf)

        trade_results = self._process_trades(unevaluated)
        results.update(trade_results)

        self.results = results
        self.log_end()
        return results

    def _process_trades(self, unevaluated: List[Recommendation]) -> Dict[str, Any]:
        today = datetime.date.today()
        trade_log = {"positions_opened": 0, "positions_closed": 0, "trade_returns": []}

        for rec in unevaluated:
            ticker = rec.stock.ticker if rec.stock else None
            if not ticker:
                continue

            existing_pos = self.db.get_position_by_recommendation(rec.id)
            if existing_pos:
                continue

            trade_spec = TradeSpec.from_signals(rec.signals)
            rec_date = rec.date
            if isinstance(rec_date, str):
                rec_date = datetime.datetime.strptime(rec_date, "%Y-%m-%d").date()
            entry_date = next_market_date(rec_date)
            exit_dt = planned_exit(entry_date, trade_spec.holding_days)
            buy_price, fallback_price = calc_entry_price(ticker, entry_date, trade_spec)
            entry_price = buy_price if buy_price is not None else fallback_price
            if entry_price is None:
                logger.debug(f"Cannot determine entry price for {ticker} entry {entry_date}, skipping position")
                continue

            with self.db.session() as session:
                db_rec = session.query(Recommendation).filter_by(id=rec.id).first()
                if db_rec:
                    db_rec.buy_window_start = trade_spec.buy_start
                    db_rec.buy_window_end = trade_spec.buy_end
                    db_rec.sell_window_start = trade_spec.sell_start
                    db_rec.sell_window_end = trade_spec.sell_end
                    db_rec.holding_period_days = trade_spec.holding_days
                    db_rec.planned_exit_date = exit_dt

            if buy_price is not None:
                with self.db.session() as session:
                    db_rec = session.query(Recommendation).filter_by(id=rec.id).first()
                    if db_rec:
                        db_rec.trade_entry_price = entry_price

            self.db.save_position({
                "recommendation_id": rec.id,
                "ticker": ticker,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "holding_period_days": trade_spec.holding_days,
                "planned_exit_date": exit_dt,
                "status": "open",
            })
            trade_log["positions_opened"] += 1

        open_positions = self.db.get_open_positions()
        closed_count = 0

        allocations, cash_pct, _ = compute_allocations(
            [
                {
                    "ticker": (rec.stock.ticker if rec.stock else "?"),
                    "total_score": rec.total_score,
                    "signals": rec.signals,
                }
                for rec in unevaluated
                if rec.stock and rec.stock.ticker
            ],
            today,
        )
        for a in allocations:
            ticker = a["ticker"]
            for rec in unevaluated:
                r_ticker = rec.stock.ticker if rec.stock else None
                if r_ticker == ticker:
                    with self.db.session() as session:
                        db_r = session.query(Recommendation).filter_by(id=rec.id).first()
                        if db_r:
                            db_r.allocation_pct = a["allocation_pct"]
                            db_r.conviction_label = a["conviction"]
                    break

        trade_log["allocations"] = allocations
        trade_log["cash_pct"] = cash_pct
        for pos in open_positions:
            pticker = pos.ticker
            if today >= pos.planned_exit_date:
                rec = pos.recommendation
                trade_spec = TradeSpec.from_signals(rec.signals if rec else [])
                exit_price = calc_exit_price(pticker, today, trade_spec)
                if exit_price is None:
                    daily = yf.download(pticker, period="5d", progress=False)
                    if not daily.empty:
                        close_val = daily["Close"].iloc[-1]
                        if isinstance(close_val, pd.Series):
                            close_val = close_val.iloc[0]
                        exit_price = round(float(close_val), 4)
                if exit_price is not None:
                    self.db.close_position(pos.id, today, exit_price)
                    trade_log["trade_returns"].append({
                        "ticker": pticker,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "trade_return": round((exit_price - pos.entry_price) / pos.entry_price * 100, 2),
                        "holding_days": (today - pos.entry_date).days,
                    })
                    trade_log["positions_closed"] += 1
                    closed_count += 1
                else:
                    cur_price = calc_current_price(pticker)
                    if cur_price is not None:
                        self.db.update_position_price(pos.id, cur_price)
            else:
                cur_price = calc_current_price(pticker)
                if cur_price is not None:
                    self.db.update_position_price(pos.id, cur_price)

        trade_results = self._compute_portfolio_metrics(today, trade_log)
        return trade_results

    def _compute_portfolio_metrics(self, today: datetime.date,
                                    trade_log: Dict[str, Any]) -> Dict[str, Any]:
        open_positions = self.db.get_open_positions()
        closed_positions = self.db.get_closed_positions_since(today - datetime.timedelta(days=1))
        today_closed = self.db.get_closed_positions_since(today)

        total_invested = 0.0
        portfolio_value = 0.0
        all_trade_rets = []

        for pos in open_positions:
            if pos.status == "open":
                total_invested += pos.entry_price
                cur_val = pos.current_price if pos.current_price else pos.entry_price
                portfolio_value += cur_val

        for pos in closed_positions:
            total_invested += pos.entry_price
            portfolio_value += pos.exit_price if pos.exit_price else pos.entry_price
            if pos.trade_return is not None:
                all_trade_rets.append(pos.trade_return)

        for tr in trade_log.get("trade_returns", []):
            if tr["trade_return"] is not None:
                all_trade_rets.append(tr["trade_return"])

        cumulative_return = round((portfolio_value / total_invested - 1) * 100, 2) if total_invested > 0 else 0.0

        prev_snaps = self.db.get_portfolio_history(days=30)
        prev_value = prev_snaps[-1]["portfolio_value"] if prev_snaps else total_invested
        daily_return = round((portfolio_value / prev_value - 1) * 100, 2) if prev_value > 0 else 0.0

        allocations = trade_log.get("allocations", [])
        if allocations and open_positions:
            pos_list = [
                {"ticker": p.ticker, "current_return": p.current_return}
                for p in open_positions if p.status == "open"
            ]
            port_ret = compute_portfolio_return(allocations, pos_list)
        else:
            port_ret = cumulative_return

        bmark = fetch_benchmark_return("SPY", days=5)
        if bmark is not None and port_ret is not None:
            alpha_val = round(port_ret - bmark, 2)
        else:
            alpha_val = None

        port_data = {
            "total_invested": round(total_invested, 2),
            "portfolio_value": round(portfolio_value, 2),
            "daily_return": daily_return,
            "cumulative_return": cumulative_return,
            "open_positions": len(open_positions),
            "closed_positions_today": len(today_closed),
            "total_closed_positions": len(closed_positions),
            "benchmark_return": bmark,
            "alpha": alpha_val,
        }

        self.db.save_portfolio_snapshot(today, port_data)

        return {
            "portfolio": port_data,
            "open_positions_data": [
                {
                    "ticker": p.ticker,
                    "entry_date": p.entry_date.isoformat() if hasattr(p.entry_date, "isoformat") else str(p.entry_date),
                    "entry_price": p.entry_price,
                    "current_return": p.current_return,
                    "days_held": (today - p.entry_date).days,
                    "days_remaining": max(0, (p.planned_exit_date - today).days),
                    "planned_exit": p.planned_exit_date.isoformat() if hasattr(p.planned_exit_date, "isoformat") else str(p.planned_exit_date),
                }
                for p in open_positions if p.status == "open"
            ],
            "closed_positions_data": [
                {
                    "ticker": p.ticker,
                    "entry_date": p.entry_date.isoformat() if hasattr(p.entry_date, "isoformat") else str(p.entry_date),
                    "exit_date": p.exit_date.isoformat() if hasattr(p.exit_date, "isoformat") else str(p.exit_date),
                    "entry_price": p.entry_price,
                    "exit_price": p.exit_price,
                    "trade_return": p.trade_return,
                    "holding_days": (p.exit_date - p.entry_date).days if p.exit_date and p.entry_date else 0,
                }
                for p in today_closed
            ],
            "allocation_data": trade_log.get("allocations", []),
            "cash_pct": trade_log.get("cash_pct", 0),
            "portfolio_return": port_ret,
            "benchmark_return": bmark,
            "alpha": alpha_val,
        }

    def _evaluate_recommendation(self, rec: Recommendation) -> Optional[Dict]:
        try:
            ticker = rec.stock.ticker if rec.stock else None
            if not ticker:
                return None
            rec_date = rec.date
            if isinstance(rec_date, str):
                rec_date = datetime.datetime.strptime(rec_date, "%Y-%m-%d").date()

            lookahead_days = [1, 2, 5]
            rec_id = rec.id
            predicted_direction = rec.predicted_direction

            hist = yf.download(ticker, period="3wk", progress=False)
            if hist.empty:
                return None
            hist.index = pd.to_datetime(hist.index)
            hist = hist.sort_index()
            dates = hist.index.tolist()

            entry_idx = None
            for idx, date_idx in enumerate(dates):
                if date_idx.date() >= rec_date:
                    entry_idx = idx
                    break
            if entry_idx is None:
                return None
            if len(dates) - 1 - entry_idx < 1:
                logger.debug(f"Skipping directional eval for rec #{rec.id} ({ticker}): no data after {dates[entry_idx].date()}")
                return None
            entry_close = hist.loc[dates[entry_idx], "Close"]
            if isinstance(entry_close, pd.Series):
                entry_close = entry_close.iloc[0]

            horizon_returns = {}
            best_return = -999.0
            worst_return = 999.0

            for days in lookahead_days:
                exit_idx = min(entry_idx + days, len(dates) - 1)
                exit_close = hist.loc[dates[exit_idx], "Close"]
                if isinstance(exit_close, pd.Series):
                    exit_close = exit_close.iloc[0]
                ret = round(((exit_close - entry_close) / entry_close) * 100, 2)
                horizon_returns[days] = ret
                if ret > best_return:
                    best_return = ret
                if ret < worst_return:
                    worst_return = ret

            if not horizon_returns:
                return None
            if best_return == -999.0:
                best_return = max(horizon_returns.values())
            if worst_return == 999.0:
                worst_return = min(horizon_returns.values())

            return_1d = horizon_returns[1]
            return_2d = horizon_returns[2]
            return_5d = horizon_returns[5]

            predicted_up = predicted_direction == "up"
            accurate = (best_return > 0 and predicted_up) or (best_return < 0 and not predicted_up)
            accurate_1d = (return_1d > 0 and predicted_up) or (return_1d < 0 and not predicted_up)
            accurate_2d = (return_2d > 0 and predicted_up) or (return_2d < 0 and not predicted_up)
            accurate_5d = (return_5d > 0 and predicted_up) or (return_5d < 0 and not predicted_up)

            max_drawdown = 0.0
            running_peak = entry_close
            for date_idx in dates[entry_idx:]:
                close_val = hist.loc[date_idx, "Close"]
                if isinstance(close_val, pd.Series):
                    close_val = close_val.iloc[0]
                if close_val > running_peak:
                    running_peak = close_val
                dd = (close_val - running_peak) / running_peak
                if dd < max_drawdown:
                    max_drawdown = dd

            spy_ret = self._fetch_spy_return(rec_date)

            with self.db.session() as session:
                db_rec = session.query(Recommendation).filter_by(id=rec_id).first()
                if db_rec:
                    db_rec.actual_close_pct = round(best_return, 2)
                    db_rec.return_1d = round(return_1d, 2)
                    db_rec.return_2d = round(return_2d, 2)
                    db_rec.return_5d = round(return_5d, 2)
                    db_rec.best_return = round(best_return, 2)
                    db_rec.worst_return = round(worst_return, 2)
                    db_rec.max_drawdown = round(max_drawdown, 4)
                    db_rec.spy_return_pct = spy_ret
                    db_rec.prediction_accurate = accurate
                    db_rec.prediction_accurate_1d = accurate_1d
                    db_rec.prediction_accurate_2d = accurate_2d
                    db_rec.prediction_accurate_5d = accurate_5d
                    db_rec.evaluated = True

            return {
                "accurate": accurate, "accurate_1d": accurate_1d, "accurate_2d": accurate_2d, "accurate_5d": accurate_5d,
                "return_pct": best_return, "return_1d": return_1d, "return_2d": return_2d, "return_5d": return_5d,
                "best_return": best_return, "worst_return": worst_return, "max_drawdown": max_drawdown,
                "spy_return": spy_ret, "ticker": ticker, "rec_date": rec_date, "predicted_direction": predicted_direction,
            }
        except Exception as e:
            logger.warning(f"Eval error for rec #{rec.id}: {e}")
            return None
