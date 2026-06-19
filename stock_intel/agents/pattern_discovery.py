import datetime
import itertools
import logging
import math
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from stock_intel.agents.base_agent import BaseAgent
from stock_intel.db.database import DatabaseManager
from stock_intel.utils.stat_utils import (
    permutation_test, cohens_h, cohens_d,
    validate_pattern_discovery,
    guard_tag, CONF as STAT_CFG,
)

logger = logging.getLogger(__name__)

SIGNAL_FIELDS = [
    ("volume_ratio", "Volume Surge", 3.0),
    ("rsi_14", "RSI", 70),
    ("premarket_change_pct", "Premarket Change", 3.0),
    ("short_term_momentum", "Momentum", 5.0),
    ("atr_14", "ATR Ratio", 2.0),
]

TEMPORAL_LABELS = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday",
}

GUARD_HEADER = "  [STAT GUARDS]"


class PatternDiscoveryAgent(BaseAgent):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        super().__init__(db, config)

    def validate(self) -> bool:
        return True

    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        ctx = context or {}
        lookback = ctx.get("lookback_days", 90)
        end_date = ctx.get("end_date", datetime.date.today())
        start_date = end_date - datetime.timedelta(days=lookback)

        recs = self.db.get_all_evaluated_recommendations()
        period_recs = [r for r in recs if start_date <= self._parse_date(r["date"]) <= end_date]
        evaluated = [r for r in period_recs if r.get("prediction_accurate") is not None]
        if len(evaluated) < STAT_CFG.MIN_ABSOLUTE:
            return {"error": "insufficient_data", "count": len(evaluated),
                    "min_required": STAT_CFG.MIN_ABSOLUTE}

        discoveries = {}
        discoveries["threshold_optimization"] = self._optimize_thresholds(evaluated)
        discoveries["multi_factor_rules"] = self._discover_rules(evaluated)
        discoveries["temporal_patterns"] = self._analyze_temporal(evaluated)
        discoveries["factor_interactions"] = self._analyze_interactions(evaluated)
        discoveries["snapshot_patterns"] = self._analyze_snapshot_patterns(evaluated, end_date)

        guard_summary = self._summarize_guards(discoveries)

        self.results = {
            "period_recs": len(evaluated),
            "discoveries": discoveries,
            "guards": guard_summary,
        }
        logger.info(f"Pattern discovery: {len(evaluated)} recs, "
                     f"{len(guard_summary.get('passed', []))} patterns passed guards, "
                     f"{len(guard_summary.get('blocked', []))} blocked")
        return self.results

    def _optimize_thresholds(self, recs: List[dict]) -> Dict:
        results = {}
        for field, label, default_thresh in SIGNAL_FIELDS:
            scores = []
            for r in recs:
                val = r.get(field)
                if val is not None:
                    scores.append({
                        "value": val, "win": r.get("prediction_accurate", False),
                        "ret": r.get("actual_return", 0),
                    })
            if len(scores) < STAT_CFG.MIN_EXPLORATORY:
                continue

            thresholds = np.linspace(
                np.percentile([s["value"] for s in scores], 10),
                np.percentile([s["value"] for s in scores], 90),
                20,
            )
            best = {"threshold": default_thresh, "win_rate": 0, "avg_return": 0, "count": 0}
            for t in thresholds:
                above = [s for s in scores if s["value"] > t]
                if len(above) < STAT_CFG.MIN_OCCURRENCES_PATTERN:
                    continue
                wr = sum(1 for s in above if s["win"]) / len(above)
                ar = sum(s["ret"] for s in above) / len(above)
                if wr > best["win_rate"]:
                    best = {"threshold": round(float(t), 3), "win_rate": round(wr, 4),
                            "avg_return": round(ar, 4), "count": len(above)}

            below = [s for s in scores if s["value"] < best["threshold"]]
            below_wr = sum(1 for s in below if s["win"]) / len(below) if below else 0
            lift = best["win_rate"] - below_wr

            passes, reasons, g_metrics = validate_pattern_discovery(
                field, best["count"], best["win_rate"],
                len(below), below_wr, len(scores),
            )

            h_val = g_metrics.get("cohens_h", 0)
            p_val = g_metrics.get("p_value", 1.0)
            lift = g_metrics.get("lift", 0)

            results[field] = {
                "label": label,
                "optimal_threshold": best["threshold"],
                "above_win_rate": best["win_rate"],
                "above_count": best["count"],
                "below_win_rate": round(below_wr, 4),
                "below_count": len(below),
                "lift": round(lift, 4),
                "cohens_h": round(h_val, 4),
                "p_value": round(p_val, 4),
                "guard_pass": passes,
                "guard_reasons": reasons,
            }
        return {"findings": list(results.values())}

    def _discover_rules(self, recs: List[dict]) -> Dict:
        rules = []
        fields = [f for f in SIGNAL_FIELDS]
        for r in range(1, 4):
            for combo in itertools.combinations(fields, r):
                self._evaluate_combination(combo, recs, rules)

        rules.sort(key=lambda x: x["win_rate"], reverse=True)
        top_rules = [
            r for r in rules
            if r["count"] >= STAT_CFG.MIN_OCCURRENCES_RULE
            and r.get("cohens_h", 0) >= STAT_CFG.MIN_COHENS_H
            and r.get("p_value", 1.0) < STAT_CFG.ALPHA
        ][:10]
        worst_rules = sorted(
            [r for r in rules if r["count"] >= STAT_CFG.MIN_OCCURRENCES_RULE],
            key=lambda x: x["win_rate"],
        )[:5]

        return {
            "best_rules": top_rules,
            "worst_rules": [
                {k: v for k, v in r.items() if k in ("signals", "fields", "count", "win_rate", "avg_return")}
                for r in worst_rules
            ],
        }

    def _evaluate_combination(self, combo: Tuple, recs: List[dict], rules: List[dict]):
        field_names = [c[0] for c in combo]
        labels = [c[1] for c in combo]
        base_thresholds = {c[0]: c[2] for c in combo}

        matched = []
        unmatched = []
        for r in recs:
            match = True
            for fn in field_names:
                val = r.get(fn)
                thresh = base_thresholds[fn]
                if val is None:
                    match = False
                    break
                if "rsi" in fn and fn == "rsi_14":
                    if not (val < 35 or val > 65):
                        match = False
                        break
                elif val < thresh:
                    match = False
                    break
            if match:
                matched.append(r)
            else:
                unmatched.append(r)

        if len(matched) < STAT_CFG.MIN_OCCURRENCES_RULE:
            return
        if len(unmatched) < STAT_CFG.MIN_PER_GROUP:
            return

        wr = sum(1 for m in matched if m.get("prediction_accurate")) / len(matched)
        ar = sum(m.get("actual_return", 0) for m in matched) / len(matched)

        other_wr = sum(1 for m in unmatched if m.get("prediction_accurate")) / len(unmatched)
        h_val = cohens_h(wr, other_wr)
        matched_wins = int(wr * len(matched))
        other_wins = int(other_wr * len(unmatched))
        p_val = permutation_test(
            matched_wins, len(matched),
            other_wins, len(unmatched),
        )

        rules.append({
            "signals": labels,
            "fields": field_names,
            "count": len(matched),
            "win_rate": round(wr, 4),
            "avg_return": round(ar, 4),
            "cohens_h": round(h_val, 4),
            "p_value": round(p_val, 4),
        })

    def _analyze_temporal(self, recs: List[dict]) -> Dict:
        dow = defaultdict(lambda: {"total": 0, "wins": 0, "returns": []})
        for r in recs:
            d = self._parse_date(r["date"])
            dow[d.weekday()]["total"] += 1
            dow[d.weekday()]["returns"].append(r.get("actual_return", 0))
            if r.get("prediction_accurate"):
                dow[d.weekday()]["wins"] += 1

        total_recs = len(recs)
        total_wr = sum(1 for r in recs if r.get("prediction_accurate")) / total_recs if total_recs > 0 else 0

        day_results = {}
        for day_num, data in sorted(dow.items()):
            if data["total"] < STAT_CFG.MIN_OCCURRENCES_TEMPORAL:
                continue
            day_wr = data["wins"] / data["total"]
            h_val = cohens_h(day_wr, total_wr)
            p_val = permutation_test(
                data["wins"], data["total"],
                int(total_wr * total_recs), total_recs,
            ) if total_recs >= 20 else 1.0
            passes = day_wr >= STAT_CFG.MIN_PER_GROUP and h_val >= STAT_CFG.MIN_COHENS_H and p_val < STAT_CFG.ALPHA
            day_results[TEMPORAL_LABELS[day_num]] = {
                "count": data["total"],
                "win_rate": round(day_wr, 4),
                "avg_return": round(sum(data["returns"]) / len(data["returns"]), 4),
                "cohens_h": round(h_val, 4),
                "p_value": round(p_val, 4),
                "guard_pass": passes,
            }

        return {"day_of_week": day_results}

    def _analyze_interactions(self, recs: List[dict]) -> Dict:
        pairs = list(itertools.combinations([
            ("volume_ratio", "Volume"), ("rsi_14", "RSI"),
            ("premarket_change_pct", "Premarket"), ("short_term_momentum", "Momentum"),
        ], 2))

        interactions = []
        for (f1, l1), (f2, l2) in pairs:
            for r in recs:
                r["_interaction"] = abs(r.get(f1, 0) or 0) * abs(r.get(f2, 0) or 0)

            p75_1 = np.percentile([r.get(f1, 0) or 0 for r in recs], 75)
            p75_2 = np.percentile([r.get(f2, 0) or 0 for r in recs], 75)
            threshold = p75_1 * p75_2
            high = [r for r in recs if r["_interaction"] > threshold]

            if len(high) < STAT_CFG.MIN_OCCURRENCES_PATTERN:
                continue

            wr = sum(1 for r in high if r.get("prediction_accurate")) / len(high)
            ar = sum(r.get("actual_return", 0) for r in high) / len(high)

            rest = [r for r in recs if r not in high]
            rest_wr = sum(1 for r in rest if r.get("prediction_accurate")) / len(rest) if rest else 0
            h_val = cohens_h(wr, rest_wr)
            p_val = permutation_test(
                int(wr * len(high)), len(high),
                int(rest_wr * len(rest)), max(len(rest), 1),
            ) if len(rest) >= STAT_CFG.MIN_PER_GROUP else 1.0

            passes = h_val >= STAT_CFG.MIN_COHENS_H and p_val < STAT_CFG.ALPHA
            interactions.append({
                "factor1": l1, "factor2": l2,
                "count": len(high), "win_rate": round(wr, 4),
                "avg_return": round(ar, 4),
                "cohens_h": round(h_val, 4),
                "p_value": round(p_val, 4),
                "guard_pass": passes,
            })

        return {"interactions": sorted(interactions, key=lambda x: x["win_rate"], reverse=True)}

    def _analyze_snapshot_patterns(self, recs: List[dict], end_date: datetime.date) -> Dict:
        try:
            window = min(len(recs) * 2, 60)
            start = end_date - datetime.timedelta(days=window)
            snapshots = self.db.get_stock_snapshots_by_date_range(start, end_date)
            if not snapshots:
                return {}

            results = {}
            fields_to_check = ["volume_ratio", "rsi_14", "premarket_change_pct", "short_term_momentum"]
            for field in fields_to_check:
                vals = [s.get(field) for s in snapshots if s.get(field) is not None]
                if len(vals) < STAT_CFG.MIN_EXPLORATORY:
                    continue
                p95 = float(np.percentile(vals, 95))
                p05 = float(np.percentile(vals, 5))
                mean = float(np.mean(vals))
                results[field] = {
                    "mean": round(mean, 3), "p5": round(p05, 3),
                    "p95": round(p95, 3), "count": len(vals),
                }
            return {"snapshot_stats": results}
        except Exception as e:
            logger.warning(f"Snapshot pattern analysis failed: {e}")
            return {}

    def _summarize_guards(self, discoveries: Dict) -> Dict:
        passed = []
        blocked = []
        for t in discoveries.get("threshold_optimization", {}).get("findings", []):
            label = t.get("label", "?")
            if t.get("guard_pass"):
                passed.append(f"threshold:{label}")
            else:
                blocked.append(f"threshold:{label} ({'; '.join(t.get('guard_reasons', ['no_reason']))})")
        for day, data in discoveries.get("temporal_patterns", {}).get("day_of_week", {}).items():
            if data.get("guard_pass"):
                passed.append(f"temporal:{day}")
            else:
                h = data.get("cohens_h", 0)
                p = data.get("p_value", 1)
                blocked.append(f"temporal:{day} (h={h:.2f}, p={p:.3f})")
        for ix in discoveries.get("factor_interactions", {}).get("interactions", []):
            if ix.get("guard_pass"):
                passed.append(f"interaction:{ix['factor1']}x{ix['factor2']}")
            else:
                blocked.append(f"interaction:{ix['factor1']}x{ix['factor2']}")
        rules = discoveries.get("multi_factor_rules", {}).get("best_rules", [])
        for r in rules:
            passed.append(f"rule:{','.join(r['signals'])}")
        return {"passed": passed, "blocked": blocked,
                "n_passed": len(passed), "n_blocked": len(blocked)}

    @staticmethod
    def _parse_date(d: Any) -> datetime.date:
        if isinstance(d, (datetime.date,)):
            return d
        if isinstance(d, str):
            return datetime.datetime.strptime(d[:10], "%Y-%m-%d").date()
        return datetime.date.today()
