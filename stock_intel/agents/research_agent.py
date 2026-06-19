import datetime
import itertools
import json
import logging
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from stock_intel.agents.base_agent import BaseAgent
from stock_intel.db.database import DatabaseManager
from stock_intel.config.settings import CONFIG
from stock_intel.utils.stat_utils import (
    bootstrap_ci, bootstrap_proportion_ci, permutation_test,
    cohens_h, cohens_d, wald_ci, out_of_sample_validate,
    validate_weight_proposal, validate_proposal_dict,
    should_run_research, should_generate_proposals,
    guard_tag, CONF as STAT_CFG,
)

logger = logging.getLogger(__name__)


FACTORS = [
    "volume_score", "news_score", "sentiment_score", "momentum_score",
    "premarket_score", "earnings_score", "float_score", "insider_score",
    "technical_score",
]

FACTOR_LABELS = {
    "volume_score": "Volume Surge", "news_score": "News Catalyst",
    "sentiment_score": "Sentiment", "momentum_score": "Momentum",
    "premarket_score": "Premarket Action", "earnings_score": "Upcoming Earnings",
    "float_score": "Low Float/Cap", "insider_score": "Insider Activity",
    "technical_score": "Technical Setup",
}

CURRENT_WEIGHTS = {
    "volume_score": 0.25, "news_score": 0.20, "sentiment_score": 0.08,
    "momentum_score": 0.15, "premarket_score": 0.03, "earnings_score": 0.10,
    "float_score": 0.10, "insider_score": 0.02, "technical_score": 0.07,
}

CURRENT_WEIGHT_KEYS = {
    "volume_score": "volume_score", "news_score": "news_catalyst_score",
    "sentiment_score": "sentiment_score", "momentum_score": "momentum_score",
    "premarket_score": "premarket_score", "earnings_score": "earnings_momentum_score",
    "float_score": "float_score", "insider_score": "insider_score",
    "technical_score": "technical_score",
}


class ResearchAgent(BaseAgent):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        super().__init__(db, config)
        self._proposals_allowed = True
        self._proposal_warning = ""

    def validate(self) -> bool:
        return True

    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.log_start()
        ctx = context or {}
        period = ctx.get("period", "monthly")
        end_date = ctx.get("end_date", datetime.date.today())
        if period == "weekly":
            start_date = end_date - datetime.timedelta(days=7)
            lookback = 7
        elif period == "monthly":
            start_date = end_date - datetime.timedelta(days=30)
            lookback = 30
        else:
            lookback = ctx.get("lookback_days", 90)
            start_date = end_date - datetime.timedelta(days=lookback)

        all_recs = self.db.get_all_evaluated_recommendations()
        period_recs = [r for r in all_recs if start_date <= self._parse_date(r["date"]) <= end_date]
        all_recs = [r for r in all_recs if r.get("prediction_accurate") is not None]

        if not period_recs:
            logger.warning(f"No evaluated recommendations in period {start_date} to {end_date}")
            return {"error": "no_data", "period": period}

        run_ok, run_msg = should_run_research(len(period_recs))
        if not run_ok:
            logger.warning(f"Research skipped: {run_msg}")
            return {"error": "insufficient_data", "message": run_msg, "count": len(period_recs)}

        props_ok, props_msg = should_generate_proposals(len(period_recs))
        if not props_ok:
            logger.info(f"Proposal generation skipped: {props_msg}")
        self._proposals_allowed = props_ok
        self._proposal_warning = props_msg if not props_ok else ""

        findings = {}
        findings["factor_performance"] = self._analyze_factor_performance(period_recs)
        findings["weight_optimization"] = self._optimize_weights(period_recs, all_recs)
        findings["winner_patterns"] = self._find_winner_patterns(period_recs)
        findings["loser_patterns"] = self._find_loser_patterns(period_recs)
        findings["sector_analysis"] = self._analyze_sectors(period_recs)
        findings["signal_combinations"] = self._analyze_signal_combinations(period_recs)
        findings["confidence_analysis"] = self._analyze_confidence(period_recs)
        findings["regime_analysis"] = self._analyze_regime(period_recs)
        findings["stability_analysis"] = self._analyze_stability(period_recs, all_recs)

        total = len(period_recs)
        evaluated = [r for r in period_recs if r.get("prediction_accurate") is not None]
        correct = sum(1 for r in evaluated if r["prediction_accurate"])
        win_rate = round(correct / len(evaluated), 4) if evaluated else 0
        returns = [r.get("actual_return", 0) for r in evaluated]
        avg_ret = round(sum(returns) / len(returns), 4) if returns else 0

        proposals = self._generate_proposals(findings, period)

        summary = self._build_summary(findings, period, total, len(evaluated), win_rate, avg_ret)
        report_data = {
            "total_recommendations": total,
            "evaluated_count": len(evaluated),
            "overall_win_rate": win_rate,
            "overall_avg_return": avg_ret,
            "summary": summary,
            "findings": findings,
        }

        report = self.db.save_research_report(period, start_date, end_date, report_data)
        for prop_data in proposals:
            self.db.save_strategy_proposal(report.id, prop_data)

        self.results = {"report_id": report.id, **report_data, "proposals": proposals}
        logger.info(f"Research complete: {total} recs, {len(proposals)} proposals generated")
        self.log_end()
        return self.results

    def _analyze_factor_performance(self, recs: List[dict]) -> Dict:
        result = {}
        for factor in FACTORS:
            bins = defaultdict(lambda: {"returns": [], "wins": 0, "total": 0})
            for r in recs:
                score = r.get(factor, 0)
                ret = r.get("actual_return", 0)
                accurate = r.get("prediction_accurate", False)
                bin_key = min(int(score * 5), 4)
                bins[bin_key]["returns"].append(ret)
                bins[bin_key]["total"] += 1
                if accurate:
                    bins[bin_key]["wins"] += 1

            perf_by_bin = {}
            for bin_key in sorted(bins.keys()):
                b = bins[bin_key]
                avg = sum(b["returns"]) / len(b["returns"]) if b["returns"] else 0
                perf_by_bin[f"{bin_key*20}-{(bin_key+1)*20}%"] = {
                    "count": b["total"], "win_rate": round(b["wins"] / b["total"], 4) if b["total"] else 0,
                    "avg_return": round(avg, 4), "wins": b["wins"],
                }

            corr = self._correlation(recs, factor)
            top_bin = max(perf_by_bin.items(), key=lambda x: x[1]["win_rate"]) if perf_by_bin else (None, None)
            result[factor] = {
                "label": FACTOR_LABELS[factor],
                "bins": perf_by_bin,
                "correlation_with_return": round(corr, 4),
                "best_bin": top_bin[0],
                "best_bin_win_rate": top_bin[1]["win_rate"] if top_bin[1] else 0,
                "overall_win_rate": round(
                    sum(1 for r in recs if r.get(factor, 0) > 0.3 and r.get("prediction_accurate", False)) /
                    max(sum(1 for r in recs if r.get(factor, 0) > 0.3), 1), 4),
            }

        ranked = sorted(result.items(), key=lambda x: x[1]["overall_win_rate"], reverse=True)
        return {"factors": result, "rankings": [{"factor": k, **v} for k, v in ranked]}

    def _optimize_weights(self, period_recs: List[dict], all_recs: List[dict]) -> Dict:
        factor_effectiveness = {}
        total_wr = sum(1 for r in period_recs if r["prediction_accurate"]) / len(period_recs) if period_recs else 0
        for factor in FACTORS:
            high_recs = [r for r in period_recs if r.get(factor, 0) > 0.3]
            if len(high_recs) < 3:
                factor_effectiveness[factor] = {"suggested_weight": CURRENT_WEIGHTS[factor],
                    "confidence": 0, "sample_size": len(high_recs), "skip": True}
                continue

            wins = sum(1 for r in high_recs if r["prediction_accurate"])
            wr = wins / len(high_recs)
            returns = [r.get("actual_return", 0) for r in high_recs]
            avg_ret = sum(returns) / len(returns)
            sharpe = (avg_ret / np.std(returns) * math.sqrt(252)) if np.std(returns) > 0 else 0

            n_bootstrap = 1000
            boot_wrs = []
            boot_returns = []
            np.random.seed(42)
            for _ in range(n_bootstrap):
                sample = np.random.choice(len(high_recs), size=len(high_recs), replace=True)
                boot_recs = [high_recs[i] for i in sample]
                boot_wins = sum(1 for r in boot_recs if r["prediction_accurate"])
                boot_wrs.append(boot_wins / len(boot_recs))
                boot_returns.append(sum(r.get("actual_return", 0) for r in boot_recs) / len(boot_recs))

            boot_wrs.sort()
            boot_returns.sort()
            ci_low = boot_wrs[int(n_bootstrap * 0.05)]
            ci_high = boot_wrs[int(n_bootstrap * 0.95)]
            ret_ci_low = boot_returns[int(n_bootstrap * 0.05)]
            ret_ci_high = boot_returns[int(n_bootstrap * 0.95)]

            relative_performance = wr - total_wr
            suggested = max(0.01, CURRENT_WEIGHTS[factor] + relative_performance * 0.3)
            suggested = min(suggested, 0.40)

            wins_other = sum(1 for r in period_recs if r.get(factor, 0) <= 0.3 and r["prediction_accurate"])
            count_other = sum(1 for r in period_recs if r.get(factor, 0) <= 0.3)
            wr_other = wins_other / count_other if count_other > 0 else total_wr

            h_val = cohens_h(wr, wr_other)
            p_val = permutation_test(wins, len(high_recs), wins_other, max(count_other, 1))

            factor_effectiveness[factor] = {
                "current_weight": CURRENT_WEIGHTS[factor],
                "suggested_weight": round(suggested, 3),
                "weight_delta": round(suggested - CURRENT_WEIGHTS[factor], 3),
                "win_rate": round(wr, 4),
                "win_rate_other": round(wr_other, 4),
                "avg_return": round(avg_ret, 4),
                "sharpe": round(sharpe, 4),
                "sample_size": len(high_recs),
                "confidence": round(min(len(high_recs) / 50, 1.0), 3),
                "ci_90_low": round(ci_low, 4), "ci_90_high": round(ci_high, 4),
                "ret_ci_90_low": round(ret_ci_low, 4), "ret_ci_90_high": round(ret_ci_high, 4),
                "relative_performance": round(relative_performance, 4),
                "cohens_h": round(h_val, 4),
                "p_value": round(p_val, 4),
                "skip": False,
            }

        ranked = sorted(factor_effectiveness.items(), key=lambda x: x[1].get("win_rate", 0) if not x[1].get("skip") else 0, reverse=True)
        return {"factor_effectiveness": factor_effectiveness, "rankings": [{"factor": k, **v} for k, v in ranked]}

    def _find_winner_patterns(self, recs: List[dict]) -> Dict:
        winners = [r for r in recs if r.get("prediction_accurate")]
        losers = [r for r in recs if not r.get("prediction_accurate")]
        if not winners or not losers:
            return {"note": "Insufficient data for pattern analysis"}

        patterns = {}
        for factor in FACTORS:
            win_scores = [r.get(factor, 0) for r in winners]
            lose_scores = [r.get(factor, 0) for r in losers]
            w_avg = sum(win_scores) / len(win_scores) if win_scores else 0
            l_avg = sum(lose_scores) / len(lose_scores) if lose_scores else 0
            gap = w_avg - l_avg
            n_w = len(win_scores)
            n_l = len(lose_scores)
            h_val = cohens_h(
                max(0.01, w_avg), max(0.01, l_avg)
            ) if n_w >= 5 and n_l >= 5 else 0.0
            p_val = permutation_test(
                int(w_avg * 100), max(n_w, 1),
                int(l_avg * 100), max(n_l, 1),
            ) if n_w + n_l >= 20 else 1.0
            patterns[factor] = {
                "label": FACTOR_LABELS[factor],
                "winner_avg_score": round(w_avg, 4),
                "loser_avg_score": round(l_avg, 4),
                "score_gap": round(gap, 4),
                "cohens_h": round(h_val, 4),
                "p_value": round(p_val, 4),
            }

        combos = self._find_best_signal_combos(winners, losers)

        return {
            "factor_profiles": patterns,
            "discriminating_factors": sorted(patterns.items(), key=lambda x: abs(x[1]["score_gap"]), reverse=True)[:5],
            "best_signal_combinations": combos["best"],
            "worst_signal_combinations": combos["worst"],
            "winner_total": len(winners),
            "loser_total": len(losers),
        }

    def _find_loser_patterns(self, recs: List[dict]) -> Dict:
        losers = [r for r in recs if not r.get("prediction_accurate")]
        if not losers:
            return {}
        from collections import Counter
        cat_counts = Counter(r.get("failure_category", "unknown") for r in losers if r.get("failure_category"))
        total_failures = sum(cat_counts.values())
        return {
            "failure_categories": {k: {"count": v, "pct": round(v / total_failures * 100, 1)}
                                   for k, v in cat_counts.most_common()},
            "total_failures": total_failures,
            "top_failure_cause": cat_counts.most_common(1)[0][0] if cat_counts else None,
        }

    def _analyze_sectors(self, recs: List[dict]) -> Dict:
        sectors = defaultdict(lambda: {"total": 0, "wins": 0, "returns": []})
        for r in recs:
            sector = r.get("sector") or "Unknown"
            sectors[sector]["total"] += 1
            sectors[sector]["returns"].append(r.get("actual_return", 0))
            if r.get("prediction_accurate"):
                sectors[sector]["wins"] += 1

        result = {}
        for sector, sd in sectors.items():
            if sd["total"] < 2:
                continue
            result[sector] = {
                "total": sd["total"], "wins": sd["wins"],
                "win_rate": round(sd["wins"] / sd["total"], 4),
                "avg_return": round(sum(sd["returns"]) / len(sd["returns"]), 4),
            }

        ranked = sorted(result.items(), key=lambda x: x[1]["win_rate"], reverse=True)
        return {"sectors": result, "rankings": [{"sector": k, **v} for k, v in ranked]}

    def _analyze_signal_combinations(self, recs: List[dict]) -> Dict:
        results = []
        for r in recs:
            active = [f for f in FACTORS if r.get(f, 0) > 0.3]
            if len(active) >= 2:
                results.append({"combo": tuple(sorted(active)), "win": r.get("prediction_accurate"), "ret": r.get("actual_return", 0)})

        combo_stats = defaultdict(lambda: {"total": 0, "wins": 0, "returns": []})
        for r in results:
            combo_stats[r["combo"]]["total"] += 1
            combo_stats[r["combo"]]["returns"].append(r["ret"])
            if r["win"]:
                combo_stats[r["combo"]]["wins"] += 1

        combo_perf = {}
        for combo, stats in combo_stats.items():
            if stats["total"] < 3:
                continue
            combo_perf[combo] = {
                "signals": [FACTOR_LABELS.get(f, f) for f in combo],
                "count": stats["total"],
                "win_rate": round(stats["wins"] / stats["total"], 4),
                "avg_return": round(sum(stats["returns"]) / len(stats["returns"]), 4),
            }

        best = sorted(combo_perf.values(), key=lambda x: x["win_rate"], reverse=True)[:5]
        worst = sorted(combo_perf.values(), key=lambda x: x["win_rate"])[:5]
        return {"best_combinations": best, "worst_combinations": worst}

    def _analyze_confidence(self, recs: List[dict]) -> Dict:
        bins = defaultdict(lambda: {"total": 0, "wins": 0, "returns": []})
        for r in recs:
            score = r.get("total_score", 0)
            bin_key = min(int(score * 10), 9)
            bins[bin_key]["total"] += 1
            bins[bin_key]["returns"].append(r.get("actual_return", 0))
            if r.get("prediction_accurate"):
                bins[bin_key]["wins"] += 1

        bin_data = {}
        for i in range(10):
            b = bins[i]
            low = i / 10
            high = (i + 1) / 10
            if b["total"] > 0:
                bin_data[f"{low:.0%}-{high:.0%}"] = {
                    "count": b["total"], "wins": b["wins"],
                    "win_rate": round(b["wins"] / b["total"], 4),
                    "avg_return": round(sum(b["returns"]) / len(b["returns"]), 4),
                }

        return {"bins": bin_data}

    def _analyze_regime(self, recs: List[dict]) -> Dict:
        try:
            import yfinance as yf
            dates = sorted(set(self._parse_date(r["date"]) for r in recs if r.get("date")))
            if not dates:
                return {}
            spy = yf.download("SPY", start=dates[0], end=dates[-1] + datetime.timedelta(days=2), progress=False)
            if spy.empty:
                return {}
            spy["DailyReturn"] = spy["Close"].pct_change() * 100
            regime_map = {}
            for idx in spy.index:
                regime_map[idx.date()] = "green" if spy.loc[idx, "DailyReturn"] >= 0 else "red"

            green_recs = [r for r in recs if regime_map.get(self._parse_date(r["date"])) == "green"]
            red_recs = [r for r in recs if regime_map.get(self._parse_date(r["date"])) == "red"]

            def regime_stats(rlist):
                if not rlist:
                    return None
                wins = sum(1 for r in rlist if r.get("prediction_accurate"))
                rets = [r.get("actual_return", 0) for r in rlist]
                return {"count": len(rlist), "win_rate": round(wins / len(rlist), 4),
                        "avg_return": round(sum(rets) / len(rets), 4)}

            return {"up_days": regime_stats(green_recs), "down_days": regime_stats(red_recs)}
        except Exception as e:
            logger.warning(f"Regime analysis failed: {e}")
            return {}

    def _analyze_stability(self, period_recs: List[dict], all_recs: List[dict]) -> Dict:
        if len(all_recs) < 10:
            return {"note": "Insufficient data for stability analysis"}
        recent = all_recs[-len(period_recs):] if len(period_recs) <= len(all_recs) else all_recs
        older = all_recs[:len(all_recs) - len(recent)]

        def perf_stats(rlist):
            if not rlist:
                return None
            wr = sum(1 for r in rlist if r.get("prediction_accurate")) / len(rlist)
            rets = [r.get("actual_return", 0) for r in rlist]
            return {"count": len(rlist), "win_rate": round(wr, 4), "avg_return": round(sum(rets) / len(rets), 4)}

        recent_stats = perf_stats(recent)
        older_stats = perf_stats(older)
        drift = None
        if recent_stats and older_stats:
            drift = round(recent_stats["win_rate"] - older_stats["win_rate"], 4)
        return {"recent": recent_stats, "older": older_stats, "drift": drift}

    def _find_best_signal_combos(self, winners: List[dict], losers: List[dict]) -> Dict:
        def get_combos(rlist):
            combos = defaultdict(lambda: {"total": 0, "wins": 0})
            for r in rlist:
                active = tuple(sorted([f for f in FACTORS if r.get(f, 0) > 0.3]))
                if len(active) >= 2:
                    combos[active]["total"] += 1
                    combos[active]["wins"] += 1 if r.get("prediction_accurate") else 0
            return combos

        win_combos = get_combos(winners)
        loss_combos = get_combos(losers)
        all_combos = set(list(win_combos.keys()) + list(loss_combos.keys()))
        best = []
        worst = []
        for combo in all_combos:
            wc = win_combos[combo]["total"] if combo in win_combos else 0
            lc = loss_combos[combo]["total"] if combo in loss_combos else 0
            total = wc + lc
            if total < 3:
                continue
            wr = wc / total
            entry = {"signals": [FACTOR_LABELS.get(f, f) for f in combo], "count": total, "win_rate": round(wr, 4)}
            if wr >= 0.6:
                best.append(entry)
            elif wr <= 0.35:
                worst.append(entry)

        return {"best": sorted(best, key=lambda x: x["win_rate"], reverse=True)[:5],
                "worst": sorted(worst, key=lambda x: x["win_rate"])[:5]}

    def _generate_proposals(self, findings: Dict, period: str) -> List[Dict]:
        proposals = []
        total_evaluated = sum(1 for r in findings.get("winner_patterns", {}).get("factor_profiles", {}).get("volume_score", {}).keys() if r) or 0
        total_evaluated = findings.get("winner_patterns", {}).get("winner_total", 0) + findings.get("winner_patterns", {}).get("loser_total", 0)
        if total_evaluated == 0:
            return []

        weight_opt = findings.get("weight_optimization", {}).get("factor_effectiveness", {})
        if weight_opt and self._proposals_allowed:
            changes = {}
            sample_sizes = {}
            conf_metrics = {}
            justifications = []
            guard_results = {}
            total_delta = 0
            for factor, data in weight_opt.items():
                if data.get("skip"):
                    continue
                delta = data.get("weight_delta", 0)
                if abs(delta) < 0.01:
                    continue

                n = data.get("sample_size", 0)
                wr = data.get("win_rate", 0)
                current_wr = findings.get("factor_performance", {}).get("rankings", [{}])[0].get("overall_win_rate", 0.5)
                if not current_wr:
                    current_wr = 0.5

                passes, reasons, g_metrics = validate_weight_proposal(
                    current_wr, {factor: data}
                )
                guard_results[factor] = {
                    "pass": passes, "reasons": reasons, "metrics": g_metrics
                }
                if not passes:
                    logger.info(f"Guard blocked factor {factor}: {'; '.join(reasons)}")
                    if any("MIN_CONFIRMATORY" in r for r in reasons):
                        continue

                changes[CURRENT_WEIGHT_KEYS[factor]] = round(data["suggested_weight"], 3)
                total_delta += delta
                sample_sizes[factor] = n
                conf_metrics[factor] = {
                    "win_rate": wr, "ci_90": [data["ci_90_low"], data["ci_90_high"]],
                    "sharpe": data["sharpe"], "confidence": data["confidence"],
                    "cohens_h": g_metrics.get(f"{factor}_cohens_h", 0),
                    "lift": g_metrics.get(f"{factor}_lift", 0),
                    "p_value": permutation_test(
                        int(wr * n), n,
                        int(current_wr * total_evaluated), max(total_evaluated, 1),
                    ) if n > 0 and total_evaluated > 0 else 1.0,
                }
                direction = "increase" if delta > 0 else "decrease"
                ci_str = f"[{data['ci_90_low']:.1%}, {data['ci_90_high']:.1%}]"
                h_val = g_metrics.get(f"{factor}_cohens_h", 0)
                p_val = conf_metrics[factor]["p_value"]
                tag = guard_tag(passes, len([r for r in reasons if "MIN_EXPLORATORY" in r]))
                justifications.append(
                    f"{FACTOR_LABELS[factor]}: {direction} from {data['current_weight']:.0%} to "
                    f"{data['suggested_weight']:.0%} (wr {wr:.1%}, 90% CI {ci_str}, "
                    f"n={n}, Cohen's h={h_val:.3f}, p={p_val:.4f}) {tag}"
                )

            if changes:
                min_n = min(sample_sizes.values())
                max_n = max(sample_sizes.values())
                conf_level = "high" if all(c.get("confidence", 0) > 0.5 for c in conf_metrics.values()) else "medium"
                if any(c.get("confidence", 0) < 0.2 for c in conf_metrics.values()):
                    conf_level = "low"
                if any(not g.get("pass") for g in guard_results.values()):
                    conf_level = "guarded"

                projected_wr = findings.get("factor_performance", {}).get("rankings", [{}])[0].get("overall_win_rate", 0.5)
                projected_wr = min(projected_wr + abs(total_delta) * 0.3, 0.75)

                has_oos = min_n >= STAT_CFG.MIN_CONFIRMATORY * 2
                oos_result = {}
                if has_oos:
                    oos_result = {
                        "note": "OOS validation available on request (run --research with verbose)",
                    }

                proposals.append({
                    "title": f"Weight Rebalancing — {period.capitalize()} Research",
                    "description": f"Proposed changes to {len(changes)} factors ({len(sample_sizes)} with sufficient data)",
                    "proposed_changes": changes,
                    "current_weights": {CURRENT_WEIGHT_KEYS[k]: v for k, v in CURRENT_WEIGHTS.items()},
                    "statistical_justification": "\n".join(justifications),
                    "sample_sizes": sample_sizes,
                    "confidence_metrics": conf_metrics,
                    "guard_results": guard_results,
                    "expected_impact": {
                        "projected_win_rate": round(projected_wr, 4),
                        "factors_changed": len(changes),
                        "total_weight_adjustment": round(total_delta, 3),
                        "confidence": conf_level,
                        "min_sample_across_factors": min_n,
                        "max_sample_across_factors": max_n,
                        "out_of_sample": oos_result,
                    },
                    "evidence_details": weight_opt,
                })

        patterns = findings.get("winner_patterns", {})
        if patterns.get("discriminating_factors") and self._proposals_allowed:
            top_discriminators = patterns["discriminating_factors"][:3]
            n_winners = patterns.get("winner_total", 0)
            n_losers = patterns.get("loser_total", 0)
            if n_winners + n_losers >= STAT_CFG.MIN_EXPLORATORY and n_winners >= STAT_CFG.MIN_PER_GROUP and n_losers >= STAT_CFG.MIN_PER_GROUP:
                discriminator_details = []
                for f, v in top_discriminators:
                    h_val = cohens_h(
                        max(0.01, v.get("winner_avg_score", 0)),
                        max(0.01, v.get("loser_avg_score", 0)),
                    )
                    p_val = permutation_test(
                        int(v.get("winner_avg_score", 0) * 100), 100,
                        int(v.get("loser_avg_score", 0) * 100), 100,
                    ) if n_winners + n_losers >= 20 else 1.0
                    tag = guard_tag(h_val >= STAT_CFG.MIN_COHENS_H)
                    discriminator_details.append(
                        f"{FACTOR_LABELS.get(f,f)} gap={v['score_gap']:.3f} "
                        f"h={h_val:.3f} p={p_val:.4f} {tag}"
                    )
                proposals.append({
                    "title": f"Signal Effectiveness Insights — {period.capitalize()}",
                    "description": f"Top discriminating factors ({n_winners} winners, {n_losers} losers)",
                    "proposed_changes": {},
                    "current_weights": {},
                    "statistical_justification": "Factor profile analysis: " + "; ".join(discriminator_details),
                    "sample_sizes": {"winners": n_winners, "losers": n_losers},
                    "confidence_metrics": {
                        f: {"cohens_h": round(cohens_h(
                            max(0.01, v.get("winner_avg_score", 0)),
                            max(0.01, v.get("loser_avg_score", 0)),
                        ), 4)}
                        for f, v in top_discriminators
                    },
                    "expected_impact": {"insight_type": "qualitative", "confidence": "low"},
                    "evidence_details": patterns,
                })

        return proposals

    def generate_html_report(self, results: Optional[Dict] = None) -> str:
        d = results or self.results
        if not d or d.get("total_recommendations", 0) == 0:
            return "<p>No research data available.</p>"

        wr = d.get("overall_win_rate", 0)
        ar = d.get("overall_avg_return", 0)
        wr_color = "#27ae60" if wr >= 0.5 else "#e74c3c"
        ar_color = "#27ae60" if ar > 0 else "#e74c3c"
        total = d.get("total_recommendations", 0)
        evaluated = d.get("evaluated_count", 0)
        period = d.get("findings", {}).get("regime_analysis", {})

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>StockIntel Research Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f0f23;color:#e0e0e0;}}
.container{{max-width:1200px;margin:0 auto;padding:20px;}}
.header{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:30px;border-radius:12px;margin-bottom:24px;}}
.header h1{{font-size:26px;}}
.header p{{opacity:0.9;margin-top:5px;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-bottom:20px;}}
.card{{background:#1a1a3e;border-radius:10px;padding:18px;border:1px solid #2a2a5e;}}
.card h3{{color:#667eea;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;}}
.card .value{{font-size:28px;font-weight:bold;}}
table{{width:100%;border-collapse:collapse;margin-bottom:16px;}}
th{{background:#2a2a5e;padding:10px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:1px;}}
td{{padding:10px;border-bottom:1px solid #2a2a5e;font-size:13px;}}
.proposal{{background:#1a1a3e;border-radius:10px;padding:20px;margin-bottom:16px;border-left:4px solid #f1c40f;}}
.proposal h3{{color:#f1c40f;}}
.proposal .change{{color:#27ae60;font-weight:bold;}}
.proposal .change.down{{color:#e74c3c;}}
.section-title{{color:#667eea;margin:24px 0 12px;font-size:18px;}}
.badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold;margin-right:4px;}}
.badge.high{{background:#27ae60;color:#fff;}}
.badge.medium{{background:#f1c40f;color:#000;}}
.badge.low{{background:#e74c3c;color:#fff;}}
.footer{{text-align:center;color:#555;font-size:11px;margin-top:30px;}}
</style></head><body>
<div class="container">
<div class="header"><h1>StockIntel Research Report</h1><p>Period: {d.get('findings',{}).get('regime_analysis',{}).get('up_days',{}).get('count','N/A') is not None and 'Period analysis' or ''}</p></div>
<div class="grid">
<div class="card"><h3>Recommendations</h3><div class="value blue">{total}</div><p style="color:#888;font-size:12px;">{evaluated} evaluated</p></div>
<div class="card"><h3>Win Rate</h3><div class="value" style="color:{wr_color};">{wr:.1%}</div></div>
<div class="card"><h3>Avg Return</h3><div class="value" style="color:{ar_color};">{ar:+.2f}%</div></div>
<div class="card"><h3>Proposals</h3><div class="value gold">{len(d.get("proposals",[]))}</div><p style="color:#888;font-size:12px;">Pending review</p></div>
</div>"""

        wopt = d.get("findings", {}).get("weight_optimization", {}).get("factor_effectiveness", {})
        if wopt:
            html += '<h2 class="section-title">Factor Effectiveness & Weight Optimization</h2>'
            html += '<table><thead><tr><th>Factor</th><th>Current</th><th>Suggested</th><th>Win Rate</th><th>Avg Ret</th><th>Sharpe</th><th>n</th><th>90% CI</th></tr></thead><tbody>'
            rankings = d.get("findings", {}).get("weight_optimization", {}).get("rankings", [])
            for entry in rankings:
                f = entry["factor"]
                if entry.get("skip"):
                    continue
                label = FACTOR_LABELS.get(f, f)
                cw = entry.get("current_weight", 0)
                sw = entry.get("suggested_weight", 0)
                delta = sw - cw
                delta_str = ""
                if abs(delta) >= 0.01:
                    dc = "change" if delta > 0 else "change down"
                    delta_str = f'<span class="{dc}">{delta:+.0%}</span>'
                ci = entry.get("ci_90_low", 0), entry.get("ci_90_high", 0)
                ci_str = f"{ci[0]:.0%}-{ci[1]:.0%}" if not entry.get("skip") else "-"
                wr_val = entry.get("win_rate", 0)
                ar_val = entry.get("avg_return", 0)
                sharp = entry.get("sharpe", 0)
                ns = entry.get("sample_size", 0)
                wr_c = "#27ae60" if wr_val >= 0.5 else "#e74c3c"
                ar_c = "#27ae60" if ar_val > 0 else "#e74c3c"
                cl = entry.get("confidence", 0)
                conf_badge = f'<span class="badge {"high" if cl>0.5 else "medium" if cl>0.2 else "low"}">{cl:.0%}</span>'
                html += f"<tr><td><strong>{label}</strong></td><td>{cw:.0%}</td><td>{sw:.0%} {delta_str}</td><td style='color:{wr_c};'>{wr_val:.1%}</td><td style='color:{ar_c};'>{ar_val:+.2f}%</td><td>{sharp:.2f}</td><td>{ns}</td><td>{ci_str}</td></tr>"
            html += "</tbody></table>"

        proposals = d.get("proposals", [])
        if proposals:
            html += '<h2 class="section-title">Strategy Proposals</h2>'
            for i, prop in enumerate(proposals):
                html += f'<div class="proposal"><h3>#{i+1}: {prop["title"]}</h3>'
                html += f'<p style="color:#aaa;margin:6px 0 10px;">{prop.get("description","")}</p>'
                if prop.get("proposed_changes"):
                    html += '<table style="width:auto;"><thead><tr><th>Parameter</th><th>Current</th><th>Proposed</th><th>Delta</th></tr></thead><tbody>'
                    for param, proposed in prop["proposed_changes"].items():
                        current = CURRENT_WEIGHTS.get(param, prop["current_weights"].get(param, 0))
                        try:
                            current = float(current)
                            proposed = float(proposed)
                        except (ValueError, TypeError):
                            continue
                        delta = proposed - current
                        dc = "change" if delta > 0 else "change down"
                        html += f'<tr><td>{param}</td><td>{current:.0%}</td><td>{proposed:.0%}</td><td class="{dc}">{delta:+.0%}</td></tr>'
                    html += '</tbody></table>'
                if prop.get("statistical_justification"):
                    html += f'<div style="margin:10px 0;padding:12px;background:#0f0f23;border-radius:6px;"><p style="font-size:13px;color:#aaa;white-space:pre-wrap;">{prop["statistical_justification"]}</p></div>'
                if prop.get("expected_impact"):
                    ei = prop["expected_impact"]
                    html += '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;">'
                    if "projected_win_rate" in ei:
                        html += f'<span style="font-size:12px;color:#888;">Projected WR: <strong style="color:#27ae60;">{ei["projected_win_rate"]:.1%}</strong></span>'
                    if "confidence" in ei:
                        html += f'<span style="font-size:12px;color:#888;">Confidence: <span class="badge {ei["confidence"]}">{ei["confidence"]}</span></span>'
                    if "factors_changed" in ei:
                        html += f'<span style="font-size:12px;color:#888;">Factors: {ei["factors_changed"]}</span>'
                    html += '</div>'
                html += '</div>'

        html += '<h2 class="section-title">Factor Performance Rankings</h2>'
        rankings = d.get("findings", {}).get("factor_performance", {}).get("rankings", [])
        if rankings:
            html += '<table><thead><tr><th>Rank</th><th>Factor</th><th>Win Rate</th><th>Best Bin</th><th>Correlation</th></tr></thead><tbody>'
            for i, entry in enumerate(rankings):
                wr_val = entry.get("overall_win_rate", 0)
                wr_c = "#27ae60" if wr_val >= 0.5 else "#e74c3c"
                html += f"<tr><td>#{i+1}</td><td>{entry.get('label','?')}</td><td style='color:{wr_c};'>{wr_val:.1%}</td><td>{entry.get('best_bin','-')}</td><td>{entry.get('correlation_with_return',0):+.3f}</td></tr>"
            html += "</tbody></table>"

        wpatterns = d.get("findings", {}).get("winner_patterns", {})
        if wpatterns.get("factor_profiles"):
            html += '<h2 class="section-title">Winner vs Loser Profiles</h2>'
            html += '<table><thead><tr><th>Factor</th><th>Winner Avg</th><th>Loser Avg</th><th>Gap</th></tr></thead><tbody>'
            profiles = wpatterns.get("factor_profiles", {})
            for f, v in sorted(profiles.items(), key=lambda x: abs(x[1]["score_gap"]), reverse=True):
                gap = v["score_gap"]
                gap_c = "#27ae60" if gap > 0 else "#e74c3c"
                html += f"<tr><td>{v['label']}</td><td>{v['winner_avg_score']:.3f}</td><td>{v['loser_avg_score']:.3f}</td><td style='color:{gap_c};'>{gap:+.4f}</td></tr>"
            html += "</tbody></table>"

        sector_data = d.get("findings", {}).get("sector_analysis", {}).get("rankings", [])
        if sector_data:
            html += '<h2 class="section-title">Sector Performance</h2>'
            html += '<table><thead><tr><th>Sector</th><th>Trades</th><th>Win Rate</th><th>Avg Return</th></tr></thead><tbody>'
            for entry in sector_data:
                wr_val = entry.get("win_rate", 0)
                ar_val = entry.get("avg_return", 0)
                wr_c = "#27ae60" if wr_val >= 0.5 else "#e74c3c"
                ar_c = "#27ae60" if ar_val > 0 else "#e74c3c"
                html += f"<tr><td>{entry.get('sector','?')}</td><td>{entry.get('total',0)}</td><td style='color:{wr_c};'>{wr_val:.1%}</td><td style='color:{ar_c};'>{ar_val:+.2f}%</td></tr>"
            html += "</tbody></table>"

        combo = d.get("findings", {}).get("signal_combinations", {})
        if combo.get("best_combinations"):
            html += '<h2 class="section-title">Best Signal Combinations</h2>'
            html += '<table><thead><tr><th>Signals</th><th>Trades</th><th>Win Rate</th></tr></thead><tbody>'
            for entry in combo["best_combinations"]:
                html += f"<tr><td>{', '.join(entry['signals'])}</td><td>{entry['count']}</td><td style='color:#27ae60;'>{entry['win_rate']:.1%}</td></tr>"
            html += "</tbody></table>"

        regime = d.get("findings", {}).get("regime_analysis", {})
        if regime:
            html += '<h2 class="section-title">Market Regime Analysis</h2><div class="grid">'
            for label, key in [("Up Market Days", "up_days"), ("Down Market Days", "down_days")]:
                rd = regime.get(key)
                if rd:
                    wr_c = "#27ae60" if rd["win_rate"] >= 0.5 else "#e74c3c"
                    html += f'<div class="card"><h3>{label}</h3><div class="value" style="color:{wr_c};">{rd["win_rate"]:.1%}</div><p style="color:#888;font-size:12px;">n={rd["count"]} | Avg {rd["avg_return"]:+.2f}%</p></div>'
            html += "</div>"

        losers = d.get("findings", {}).get("loser_patterns", {})
        if losers:
            html += '<h2 class="section-title">Failure Analysis</h2>'
            html += '<table><thead><tr><th>Cause</th><th>Count</th><th>% of Failures</th></tr></thead><tbody>'
            for cause, stats in losers.get("failure_categories", {}).items():
                html += f"<tr><td>{cause.replace('_',' ').title()}</td><td>{stats['count']}</td><td>{stats['pct']:.1f}%</td></tr>"
            html += "</tbody></table>"

        stability = d.get("findings", {}).get("stability_analysis", {})
        if stability.get("recent") and stability.get("older"):
            html += '<h2 class="section-title">Performance Stability</h2><div class="grid">'
            for label, key in [("Recent Period", "recent"), ("Older Period", "older")]:
                sd = stability.get(key)
                if sd:
                    wr_c = "#27ae60" if sd["win_rate"] >= 0.5 else "#e74c3c"
                    html += f'<div class="card"><h3>{label}</h3><div class="value" style="color:{wr_c};">{sd["win_rate"]:.1%}</div><p style="color:#888;font-size:12px;">n={sd["count"]} | Avg {sd["avg_return"]:+.2f}%</p></div>'
            drift = stability.get("drift")
            if drift is not None:
                drift_c = "#27ae60" if abs(drift) < 0.05 else "#f1c40f" if abs(drift) < 0.1 else "#e74c3c"
                html += f'<div class="card"><h3>Performance Drift</h3><div class="value" style="color:{drift_c};">{drift:+.1%}</div><p style="color:#888;font-size:12px;">Recent vs older win rate change</p></div></div>'

        html += f'<div class="footer"><p>StockIntel Research | Generated {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>'
        html += '<p style="margin-top:4px;">⚠️ All proposals require human review before implementation.</p></div>'
        html += "</div></body></html>"
        return html

    def print_research_summary(self, results: Optional[Dict] = None):
        d = results or self.results
        if not d or d.get("total_recommendations", 0) == 0:
            print("\n  No research data available.")
            return
        wr = d.get("overall_win_rate", 0)
        ar = d.get("overall_avg_return", 0)
        total = d.get("total_recommendations", 0)
        proposals = d.get("proposals", [])
        print("\n" + "=" * 80)
        print(f"  RESEARCH REPORT — {'Monthly' if 'monthly' in str(d.get('findings','')) else 'Weekly'} Analysis")
        print("=" * 80)
        print(f"  Recommendations: {total} | Win Rate: {wr:.1%} | Avg Return: {ar:+.2f}%")
        print()
        wopt = d.get("findings", {}).get("weight_optimization", {}).get("factor_effectiveness", {})
        if wopt:
            print(f"  {'Factor':<22} {'Current':<9} {'Suggest':<9} {'WR':<7} {'n':<4} {'CI':<12} {'h':<7} {'p':<7}")
            print("  " + "-" * 80)
            for f, data in sorted(wopt.items(), key=lambda x: x[1].get("win_rate", 0) if not x[1].get("skip") else 0, reverse=True):
                if data.get("skip"):
                    continue
                cw = data["current_weight"]
                sw = data["suggested_weight"]
                wr_val = data["win_rate"]
                ns = data["sample_size"]
                ci = f"{data['ci_90_low']:.0%}-{data['ci_90_high']:.0%}"
                h_val = data.get("cohens_h", 0)
                p_val = data.get("p_value", 1)
                h_tag = guard_tag(h_val >= STAT_CFG.MIN_COHENS_H)
                p_tag = guard_tag(p_val < STAT_CFG.ALPHA)
                delta = "+" if sw > cw else ""
                print(f"  {FACTOR_LABELS[f]:<22} {cw:.0%}        {sw:.0%} ({delta}{sw-cw:+.0%}) {wr_val:.1%}  {ns:<3} {ci:<11} {h_val:.2f}{h_tag:<4} {p_val:.3f}{p_tag:<4}")
        if proposals:
            print(f"\n  Strategy Proposals: {len(proposals)}")
            for i, p in enumerate(proposals):
                print(f"    #{i+1}: {p['title']}")
                guard_res = p.get("guard_results", {})
                if guard_res:
                    blocked = sum(1 for g in guard_res.values() if not g["pass"])
                    total_g = len(guard_res)
                    print(f"       Guards: {total_g - blocked}/{total_g} pass, {blocked} blocked")
                if p.get("proposed_changes"):
                    for param, val in p["proposed_changes"].items():
                        print(f"       {param}: {val:.0%}")
                cm = p.get("confidence_metrics", {})
                if cm:
                    first_key = next(iter(cm))
                    if "cohens_h" in cm[first_key]:
                        print(f"       Effect sizes: " + ", ".join(
                            f"{k}: h={v.get('cohens_h',0):.2f}" for k, v in cm.items()
                        ))
                    if "p_value" in cm.get(first_key, {}):
                        print(f"       P-values: " + ", ".join(
                            f"{k}: p={v['p_value']:.4f}" for k, v in cm.items()
                        ))
        print("=" * 80)

    def _correlation(self, recs: List[dict], factor: str) -> float:
        scores = [r.get(factor, 0) for r in recs]
        returns = [r.get("actual_return", 0) for r in recs]
        if len(scores) < 3 or len(set(scores)) < 2:
            return 0.0
        try:
            return float(np.corrcoef(scores, returns)[0, 1])
        except Exception:
            return 0.0

    def _build_summary(self, findings: Dict, period: str, total: int, evaluated: int, win_rate: float, avg_ret: float) -> str:
        parts = [f"Analysis of {total} recommendations ({evaluated} evaluated) over {period} period.",
                 f"Overall win rate: {win_rate:.1%}, avg return: {avg_ret:+.2f}%."]
        wopt = findings.get("weight_optimization", {}).get("factor_effectiveness", {})
        if wopt:
            improvable = sum(1 for f, d in wopt.items() if not d.get("skip") and abs(d.get("weight_delta", 0)) >= 0.02)
            if improvable:
                parts.append(f"Identified {improvable} factors with meaningful weight optimization opportunities.")
        return " ".join(parts)

    @staticmethod
    def _parse_date(d: Any) -> datetime.date:
        if isinstance(d, datetime.date):
            return d
        if isinstance(d, str):
            return datetime.datetime.strptime(d[:10], "%Y-%m-%d").date()
        return datetime.date.today()
