import datetime
import logging
from typing import Dict, List, Optional, Any
from collections import OrderedDict
from stock_intel.agents.base_agent import BaseAgent
from stock_intel.db.database import DatabaseManager

logger = logging.getLogger(__name__)


FACTOR_META = OrderedDict([
    ("volume_score",      {"label": "High Volume",       "color": "#3498db", "icon": "Vol"}),
    ("news_score",        {"label": "News Catalyst",     "color": "#f39c12", "icon": "News"}),
    ("sentiment_score",   {"label": "Sentiment",         "color": "#9b59b6", "icon": "Sent"}),
    ("momentum_score",    {"label": "Momentum",          "color": "#e74c3c", "icon": "Mom"}),
    ("earnings_score",    {"label": "Upcoming Earnings", "color": "#2ecc71", "icon": "Earn"}),
    ("float_score",       {"label": "Low Float/Cap",     "color": "#1abc9c", "icon": "Float"}),
    ("insider_score",     {"label": "Insider Activity",  "color": "#e67e22", "icon": "Ins"}),
    ("technical_score",   {"label": "Technical Setup",   "color": "#f1c40f", "icon": "Tech"}),
    ("premarket_score",   {"label": "Premarket Action",  "color": "#e91e63", "icon": "PM"}),
])

FACTOR_KEYS = list(FACTOR_META.keys())


class AttributionEngine(BaseAgent):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        super().__init__(db, config)

    def validate(self) -> bool:
        return True

    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.log_start()
        days = (context or {}).get("lookback_days", 90)
        recs = self.db.get_evaluated_recommendations(days=days)
        if not recs:
            logger.warning("No evaluated recommendations found for attribution")
            return {"total_recommendations": 0, "factor_attributions": {}}

        factor_returns: Dict[str, list] = {k: [] for k in FACTOR_KEYS}
        multi_lists: Dict[str, list] = {k: [] for k in FACTOR_KEYS}

        for rec in recs:
            scores = {k: rec.get(k, 0) for k in FACTOR_KEYS}
            actual_return = rec.get("actual_return", 0)
            accurate = rec.get("prediction_accurate", False)

            max_score = max(scores.values()) if scores else 0
            if max_score <= 0:
                continue

            dominant_factors = [k for k, v in scores.items() if v == max_score]

            for k in dominant_factors:
                factor_returns[k].append(actual_return)

            top_n = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            for k, _ in top_n:
                if scores[k] > 0:
                    multi_lists[k].append(actual_return)

        attributions = {}
        for k in FACTOR_KEYS:
            returns = factor_returns[k]
            if not returns:
                continue
            win_count = sum(1 for r in returns if r > 0)
            total = len(returns)
            avg_ret = sum(returns) / total
            best_3d = max(returns) if returns else 0
            worst_3d = min(returns) if returns else 0
            meta = FACTOR_META[k]
            attributions[k] = {
                "factor_label": meta["label"],
                "color": meta["color"],
                "icon": meta["icon"],
                "win_rate": round(win_count / total, 4) if total > 0 else 0.0,
                "avg_return": round(avg_ret, 4),
                "total_trades": total,
                "win_count": win_count,
                "loss_count": total - win_count,
                "best_return": round(best_3d, 2),
                "worst_return": round(worst_3d, 2),
            }

        multi_attributions = {}
        for k in FACTOR_KEYS:
            returns = multi_lists[k]
            if not returns:
                continue
            win_count = sum(1 for r in returns if r > 0)
            total = len(returns)
            avg_ret = sum(returns) / total
            multi_attributions[k] = {
                "factor_label": FACTOR_META[k]["label"],
                "win_rate": round(win_count / total, 4) if total > 0 else 0.0,
                "avg_return": round(avg_ret, 4),
                "total_trades": total,
                "win_count": win_count,
            }

        self.results = {
            "total_recommendations": len(recs),
            "lookback_days": days,
            "factor_attributions": attributions,
            "multi_factor_attributions": multi_attributions,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        logger.info(f"Attribution: {sum(a['total_trades'] for a in attributions.values())} factor-trades across {len(attributions)} factors")
        self.log_end()
        return self.results

    def print_report(self, results: Optional[Dict] = None):
        data = results or self.results
        if not data or not data.get("factor_attributions"):
            print("\n  No attribution data available yet.")
            return
        print("\n" + "=" * 70)
        print(f"  SIGNAL ATTRIBUTION REPORT — Last {data['lookback_days']} Days")
        print("=" * 70)
        print(f"  Total Recommendations Evaluated: {data['total_recommendations']}")
        print()
        print(f"  {'Factor':<25} {'Trades':<8} {'Win Rate':<10} {'Avg Return':<12} {'Best':<8} {'Worst':<8}")
        print("  " + "-" * 71)
        sorted_factors = sorted(data["factor_attributions"].items(), key=lambda x: x[1]["win_rate"], reverse=True)
        for k, attr in sorted_factors:
            wr = attr["win_rate"]
            ar = attr["avg_return"]
            wr_color = "GREEN" if wr >= 0.5 else "RED"
            ar_color = "GREEN" if ar > 0 else "RED"
            print(f"  {attr['factor_label']:<25} {attr['total_trades']:<8} {wr:.1%}      {ar:>+.2f}%     {attr['best_return']:>+6.2f} {attr['worst_return']:>+6.2f}")
        print("=" * 70)
