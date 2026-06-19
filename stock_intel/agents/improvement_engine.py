import datetime
import logging
from typing import Dict, List, Any, Optional
from stock_intel.agents.base_agent import BaseAgent
from stock_intel.agents.research_agent import FACTORS, FACTOR_LABELS, CURRENT_WEIGHTS, CURRENT_WEIGHT_KEYS
from stock_intel.db.database import DatabaseManager
from stock_intel.utils.stat_utils import (
    permutation_test, cohens_h, cohens_d, bootstrap_proportion_ci,
    validate_weight_proposal, validate_proposal_dict,
    should_generate_proposals, guard_tag, CONF as STAT_CFG,
)

logger = logging.getLogger(__name__)


class ImprovementEngine(BaseAgent):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        super().__init__(db, config)

    def validate(self) -> bool:
        return True

    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        ctx = context or {}
        research = ctx.get("research_results")
        patterns = ctx.get("pattern_results")
        calibration = ctx.get("calibration_results", {})
        post_mortem = ctx.get("post_mortem_results", {})
        existing_reports = self.db.get_research_reports(limit=5)

        total_eval = 0
        if research:
            total_eval = research.get("evaluated_count", 0) or research.get("total_recommendations", 0)

        props_ok, props_msg = should_generate_proposals(total_eval)
        if not total_eval or not props_ok:
            logger.info(f"Improvement engine skipped: {props_msg}")
            return {"proposals": [], "note": props_msg, "total": 0, "has_approvable": False}

        proposals = []
        has_approvable = False
        guard_report = []

        if research:
            wopt = research.get("findings", {}).get("weight_optimization", {})
            weight_props = self._generate_weight_proposals(wopt, research, total_eval)
            for p in weight_props:
                passes, reasons = validate_proposal_dict(p)
                if not passes:
                    logger.info(f"Guard blocked improvement proposal: {'; '.join(reasons)}")
                    guard_report.append({"title": p.get("title", "?"), "blocked": True, "reasons": reasons})
                    continue
                proposals.append(p)
                guard_report.append({"title": p.get("title", "?"), "blocked": False})
                if p.get("proposal_type") == "weight_rebalance":
                    has_approvable = True

            if patterns:
                pattern_props = self._generate_pattern_proposals(patterns, total_eval)
                for p in pattern_props:
                    passes, reasons = validate_proposal_dict(p)
                    if not passes:
                        guard_report.append({"title": p.get("title", "?"), "blocked": True, "reasons": reasons})
                        continue
                    proposals.append(p)
                    guard_report.append({"title": p.get("title", "?"), "blocked": False})

            if calibration:
                cal_props = self._generate_calibration_proposals(calibration, total_eval)
                for p in cal_props:
                    passes, reasons = validate_proposal_dict(p)
                    if not passes:
                        guard_report.append({"title": p.get("title", "?"), "blocked": True, "reasons": reasons})
                        continue
                    proposals.append(p)
                    guard_report.append({"title": p.get("title", "?"), "blocked": False})

            if post_mortem:
                pm_props = self._generate_post_mortem_proposals(post_mortem, total_eval)
                for p in pm_props:
                    passes, reasons = validate_proposal_dict(p)
                    if not passes:
                        guard_report.append({"title": p.get("title", "?"), "blocked": True, "reasons": reasons})
                        continue
                    proposals.append(p)
                    guard_report.append({"title": p.get("title", "?"), "blocked": False})

        if not proposals:
            return {"proposals": [], "note": "no_proposals_passed_guards",
                    "total": 0, "has_approvable": False, "guard_report": guard_report}

        result = {
            "proposals": proposals,
            "total": len(proposals),
            "has_approvable": has_approvable,
            "guard_report": guard_report,
        }
        logger.info(f"Improvement engine: {len(proposals)}/{len(guard_report)} proposals passed guards")
        return result

    def _generate_weight_proposals(self, wopt: Dict, research: Dict, total_eval: int) -> List[Dict]:
        factor_data = wopt.get("factor_effectiveness", {})
        if not factor_data:
            return []

        findings = research.get("findings", {})
        current_wr = research.get("overall_win_rate", 0)

        changes = {}
        justifications = []
        sample_sizes = {}
        conf_metrics = {}
        guard_results = {}
        best_n = 0
        total_pos_delta = 0
        total_neg_delta = 0

        for factor, data in factor_data.items():
            if data.get("skip"):
                continue
            delta = data.get("weight_delta", 0)
            if abs(delta) < 0.01:
                continue

            n = data.get("sample_size", 0)
            wr = data.get("win_rate", 0)
            best_n = max(best_n, n)

            passes, reasons, g_metrics = validate_weight_proposal(
                current_wr, {factor: data}
            )
            guard_results[factor] = {
                "pass": passes, "reasons": reasons, "metrics": g_metrics
            }
            if not passes and any("MIN_CONFIRMATORY" in r for r in reasons):
                logger.info(f"Guard blocked factor {factor}: {'; '.join(reasons)}")
                continue

            key = CURRENT_WEIGHT_KEYS.get(factor, factor)
            changes[key] = round(data["suggested_weight"], 3)
            sample_sizes[factor] = n

            h_val = g_metrics.get(f"{factor}_cohens_h", 0)
            p_val = permutation_test(
                int(wr * n), n,
                int(current_wr * total_eval), max(total_eval, 1),
            ) if n > 0 and total_eval > 0 else 1.0

            conf_metrics[factor] = {
                "win_rate": wr,
                "ci_90": [data["ci_90_low"], data["ci_90_high"]],
                "sharpe": data["sharpe"],
                "confidence": data["confidence"],
                "cohens_h": round(h_val, 4),
                "p_value": round(p_val, 4),
            }
            if delta > 0:
                total_pos_delta += delta
            else:
                total_neg_delta += abs(delta)
            direction = "increase" if delta > 0 else "decrease"
            tag = guard_tag(passes, len([r for r in reasons if "EXPLORATORY" in r]))
            justifications.append(
                f"{FACTOR_LABELS[factor]}: {direction} from {data['current_weight']:.0%} to "
                f"{data['suggested_weight']:.0%} (wr {wr:.1%}, "
                f"90% CI [{data['ci_90_low']:.1%}, {data['ci_90_high']:.1%}], "
                f"n={n}, h={h_val:.3f}, p={p_val:.4f}) {tag}"
            )

        if not changes:
            return []

        net_delta = total_pos_delta - total_neg_delta
        projected_wr = min(current_wr + abs(net_delta) * 0.2 + 0.01, 0.75)
        conf_level = "high" if all(c.get("confidence", 0) > 0.5 for c in conf_metrics.values()) else "medium"
        if any(c.get("confidence", 0) < 0.2 for c in conf_metrics.values()):
            conf_level = "low"
        if any(not g.get("pass") for g in guard_results.values()):
            conf_level = "guarded"

        return [{
            "title": f"Weight Rebalancing ({len(changes)} factors)",
            "description": f"Adjust {len(changes)} factor weights based on empirical performance (n={best_n} max)",
            "proposed_changes": changes,
            "current_weights": {CURRENT_WEIGHT_KEYS[k]: v for k, v in CURRENT_WEIGHTS.items()},
            "statistical_justification": "\n".join(justifications),
            "sample_sizes": sample_sizes,
            "confidence_metrics": conf_metrics,
            "guard_results": guard_results,
            "expected_impact": {
                "projected_win_rate": round(projected_wr, 4),
                "current_win_rate": round(current_wr, 4),
                "projected_improvement": round(projected_wr - current_wr, 4),
                "factors_changed": len(changes),
                "net_weight_delta": round(net_delta, 3),
                "confidence": conf_level,
                "min_sample": min(sample_sizes.values()) if sample_sizes else 0,
                "max_sample": best_n,
            },
            "evidence_details": {"factor_analysis": factor_data, "type": "weight_rebalance"},
            "proposal_type": "weight_rebalance",
        }]

    def _generate_pattern_proposals(self, patterns: Dict, total_eval: int) -> List[Dict]:
        discoveries = patterns.get("discoveries", {})
        guard_summary = patterns.get("guards", {})
        proposals = []

        thresholds = discoveries.get("threshold_optimization", {}).get("findings", [])
        for t in thresholds:
            if not t.get("guard_pass"):
                continue
            if total_eval < STAT_CFG.MIN_EXPLORATORY:
                continue
            lift = t.get("lift", 0)
            if abs(lift) < STAT_CFG.MIN_LIFT:
                continue
            proposals.append({
                "title": f"Signal Threshold: {t['label']}",
                "description": (f"Optimal {t['label']} threshold is {t['optimal_threshold']} "
                               f"(wr {t['above_win_rate']:.1%} above vs {t['below_win_rate']:.1%} below, "
                               f"lift {lift:.1%}, n={t['above_count']}, "
                               f"h={t.get('cohens_h',0):.2f}, p={t.get('p_value',1):.3f})"),
                "proposed_changes": {},
                "statistical_justification": (
                    f"Cross-validated threshold optimization found optimal {t['label']} threshold "
                    f"at {t['optimal_threshold']}. Above-threshold: {t['above_win_rate']:.1%} "
                    f"vs below-threshold: {t['below_win_rate']:.1%}. "
                    f"Cohen's h={t.get('cohens_h',0):.3f}, p={t.get('p_value',1):.4f}."
                ),
                "sample_sizes": {"above_threshold": t["above_count"], "below_threshold": t.get("below_count", 0)},
                "confidence_metrics": {
                    "cohens_h": t.get("cohens_h", 0),
                    "p_value": t.get("p_value", 1),
                    "lift": lift,
                },
                "expected_impact": {
                    "insight_type": "threshold_discovery",
                    "confidence": "low" if t.get("p_value", 1) > STAT_CFG.ALPHA else "medium",
                },
                "evidence_details": t,
                "proposal_type": "threshold_insight",
            })

        best_rules = discoveries.get("multi_factor_rules", {}).get("best_rules", [])
        if best_rules and total_eval >= STAT_CFG.MIN_CONFIRMATORY:
            proposal = {
                "title": f"Top Signal Rules ({len(best_rules)} found)",
                "description": "Multi-factor signal combinations with highest win rates (all passed statistical guards)",
                "proposed_changes": {},
                "statistical_justification": "\n".join(
                    f"Rule {i+1}: {', '.join(r['signals'])} — "
                    f"wr {r['win_rate']:.1%}, avg ret {r['avg_return']:+.2f}%, "
                    f"n={r['count']}, h={r.get('cohens_h',0):.2f}, p={r.get('p_value',1):.4f}"
                    for i, r in enumerate(best_rules)
                ),
                "sample_sizes": {f"rule_{i}": r["count"] for i, r in enumerate(best_rules)},
                "confidence_metrics": {
                    f"rule_{i}": {
                        "cohens_h": r.get("cohens_h", 0),
                        "p_value": r.get("p_value", 1),
                    }
                    for i, r in enumerate(best_rules)
                },
                "expected_impact": {"insight_type": "signal_rules", "confidence": "medium"},
                "evidence_details": {"rules": best_rules, "type": "multi_factor_rules"},
                "proposal_type": "signal_rule_insight",
            }
            proposals.append(proposal)

        temporal = discoveries.get("temporal_patterns", {}).get("day_of_week", {})
        if temporal and total_eval >= STAT_CFG.MIN_CONFIRMATORY:
            passing_days = {d: s for d, s in temporal.items() if s.get("guard_pass")}
            if passing_days:
                best_day = max(passing_days.items(), key=lambda x: x[1]["win_rate"])
                worst_day = min(passing_days.items(), key=lambda x: x[1]["win_rate"])
                detail_lines = []
                for day, stats in sorted(temporal.items()):
                    tag = guard_tag(stats.get("guard_pass", False))
                    detail_lines.append(
                        f"{day}: wr {stats['win_rate']:.1%}, avg {stats['avg_return']:+.2f}%, "
                        f"n={stats['count']}, h={stats.get('cohens_h',0):.2f}, p={stats.get('p_value',1):.3f} {tag}"
                    )
                proposals.append({
                    "title": "Day-of-Week Performance Patterns",
                    "description": f"Best: {best_day[0]} ({best_day[1]['win_rate']:.1%}), "
                                   f"Worst: {worst_day[0]} ({worst_day[1]['win_rate']:.1%})",
                    "proposed_changes": {},
                    "statistical_justification": "\n".join(detail_lines),
                    "sample_sizes": {day: s["count"] for day, s in temporal.items()},
                    "confidence_metrics": {
                        day: {"cohens_h": s.get("cohens_h", 0), "p_value": s.get("p_value", 1)}
                        for day, s in temporal.items()
                    },
                    "expected_impact": {"insight_type": "temporal_pattern", "confidence": "low"},
                    "evidence_details": {"temporal": temporal, "type": "day_of_week"},
                    "proposal_type": "temporal_insight",
                })

        return proposals

    def _generate_calibration_proposals(self, calibration: Dict, total_eval: int) -> List[Dict]:
        if total_eval < STAT_CFG.MIN_CONFIRMATORY:
            return []
        if calibration.get("total_recommendations", 0) < STAT_CFG.MIN_EXPLORATORY:
            return []

        ece = calibration.get("ece", 1)
        if ece < 0.05:
            return []

        mce = calibration.get("mce", 0)
        n_recs = calibration.get("total_recommendations", 0)
        avg_conf = calibration.get("average_confidence", 0)
        acc = calibration.get("overall_accuracy", 0)
        conf_bias = avg_conf - acc if avg_conf and acc else 0

        proposals = [{
            "title": "Confidence Calibration Adjustment",
            "description": f"ECE={ece:.1%}, MCE={mce:.1%}, confidence bias={conf_bias:+.1%} (n={n_recs})",
            "proposed_changes": {},
            "statistical_justification": (
                f"Expected Calibration Error: {ece:.1%} (threshold: <5% good). "
                f"Max Calibration Error: {mce:.1%}. "
                f"Overall accuracy: {acc:.1%}, Average confidence: {avg_conf:.1%}, "
                f"Confidence bias: {conf_bias:+.1%}. "
                f"Recommendation: {'recalibrate via Platt scaling or isotonic regression' if ece > 0.1 else 'monitor'}"
            ),
            "sample_sizes": {"total_recs": n_recs},
            "confidence_metrics": {
                "ece": round(ece, 4), "mce": round(mce, 4),
                "confidence_bias": round(conf_bias, 4),
                "accuracy": round(acc, 4),
                "avg_confidence": round(avg_conf, 4),
            },
            "expected_impact": {
                "insight_type": "calibration_fix", "confidence": "medium",
                "ece": ece,
                "severity": "high" if ece > 0.10 else "medium" if ece > 0.05 else "low",
            },
            "evidence_details": {"calibration": calibration, "type": "calibration"},
            "proposal_type": "calibration_insight",
        }]
        return proposals

    def _generate_post_mortem_proposals(self, post_mortem: Dict, total_eval: int) -> List[Dict]:
        if total_eval < STAT_CFG.MIN_CONFIRMATORY:
            return []
        cats = post_mortem.get("failure_categories", {})
        total_failures = post_mortem.get("total_failures", 0)
        if not cats or total_failures < STAT_CFG.MIN_FAILURES_ANALYSIS:
            return []

        top_cause = max(cats.items(), key=lambda x: x[1]["count"]) if cats else None
        if not top_cause or top_cause[1]["pct"] < STAT_CFG.MIN_FAILURE_PCT * 100:
            return []

        n_dominated = sum(1 for c in cats.values() if c["pct"] > 30)
        severity = "high" if n_dominated > 0 else "medium"

        proposals = [{
            "title": f"Failure Pattern: {top_cause[0].replace('_', ' ').title()}",
            "description": (
                f"{top_cause[1]['pct']:.1f}% of {total_failures} failures "
                f"attributed to {top_cause[0].replace('_', ' ')} "
                f"(n={top_cause[1]['count']})"
            ),
            "proposed_changes": {},
            "statistical_justification": (
                f"Failure breakdown ({total_failures} total):\n"
                + "\n".join(
                    f"  {k}: {v['pct']:.1f}% ({v['count']})"
                    for k, v in sorted(cats.items(), key=lambda x: x[1]['count'], reverse=True)
                )
                + f"\n\nTop cause '{top_cause[0]}' dominates at {top_cause[1]['pct']:.1f}% "
                f"({top_cause[1]['count']} occurrences)."
            ),
            "sample_sizes": {"total_failures": total_failures},
            "confidence_metrics": {
                "top_cause_pct": round(top_cause[1]["pct"] / 100, 4),
                "top_cause_count": top_cause[1]["count"],
                "total_causes": len(cats),
            },
            "expected_impact": {
                "insight_type": "failure_pattern",
                "confidence": "low",
                "severity": severity,
            },
            "evidence_details": {"failure_categories": cats, "type": "post_mortem"},
            "proposal_type": "failure_insight",
        }]
        return proposals

    def _analyze_proposal_continuity(self, existing_reports: List) -> List[Dict]:
        return []

    def print_report(self, results: Optional[Dict] = None):
        d = results or self.results
        if not d or not d.get("proposals"):
            n_blocked = len(d.get("guard_report", [])) if d else 0
            note = d.get("note", "no_improvements_found") if d else "no_data"
            print(f"\n  No improvement proposals generated. ({note})")
            if n_blocked:
                print(f"  {n_blocked} proposals blocked by statistical guards.")
            return

        proposals = d["proposals"]
        guard_report = d.get("guard_report", [])
        approvable = sum(1 for p in proposals if p.get("proposal_type") == "weight_rebalance")
        blocked = sum(1 for g in guard_report if g.get("blocked"))
        print("\n" + "=" * 80)
        print(f"  IMPROVEMENT ENGINE — {len(proposals)} proposals, {blocked} blocked by guards")
        print("=" * 80)
        for i, p in enumerate(proposals):
            ptype = p.get("proposal_type", "unknown").replace("_", " ").title()
            gr = next((g for g in guard_report if g.get("title") == p.get("title")), None)
            blocked_tag = " [BLOCKED]" if gr and gr.get("blocked") else ""
            print(f"\n  #{i+1} [{ptype}]{blocked_tag} {p['title']}")
            print(f"      {p.get('description', '')[:120]}")
            if p.get("proposed_changes"):
                for k, v in p["proposed_changes"].items():
                    print(f"      {k}: {v:.0%}")
            cm = p.get("confidence_metrics", {})
            if cm and any("cohens_h" in v for v in cm.values() if isinstance(v, dict)):
                vals = [(k, v.get("cohens_h", 0), v.get("p_value", 1))
                        for k, v in cm.items() if isinstance(v, dict)]
                h_str = ", ".join(f"{k}: h={h:.2f} p={p:.4f}" for k, h, p in vals[:3])
                if h_str:
                    print(f"      Effect sizes: {h_str}")
            if "p_value" in cm if isinstance(cm, dict) and not any(isinstance(v, dict) for v in cm.values()) else False:
                print(f"      p-value: {cm.get('p_value', 1):.4f}")
            ei = p.get("expected_impact", {})
            if ei.get("projected_win_rate"):
                print(f"      Current WR: {ei.get('current_win_rate', 0):.1%} -> "
                      f"Projected: {ei['projected_win_rate']:.1%}")
            sample_sizes = p.get("sample_sizes", {})
            if sample_sizes:
                n_vals = list(sample_sizes.values()) if isinstance(sample_sizes, dict) else [sample_sizes]
                n_max = max(n_vals) if n_vals else 0
                n_min = min(n_vals) if n_vals else 0
                print(f"      Sample sizes: min={n_min}, max={n_max}")
            if gr and gr.get("reasons"):
                for r in gr["reasons"]:
                    print(f"      [GUARD FAIL] {r}")
        print("=" * 80)
