import datetime
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Tuple
from stock_intel.agents.base_agent import BaseAgent
from stock_intel.db.database import DatabaseManager

logger = logging.getLogger(__name__)


FACTOR_LABELS = {
    "volume_score": "Volume",
    "news_catalyst_score": "News",
    "sentiment_score": "Sentiment",
    "momentum_score": "Momentum",
    "earnings_score": "Earnings",
    "float_score": "Float/Cap",
    "insider_score": "Insider",
    "technical_score": "Technical",
    "premarket_score": "Premarket",
}


class PeriodicReporter(BaseAgent):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        super().__init__(db, config)

    def validate(self) -> bool:
        return True

    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.log_start()
        ctx = context or {}
        period = ctx.get("period", "monthly")
        end_date = ctx.get("end_date", datetime.date.today())
        if period == "weekly":
            start_date = end_date - datetime.timedelta(days=7)
        elif period == "monthly":
            start_date = end_date - datetime.timedelta(days=30)
        else:
            days = ctx.get("lookback_days", 30)
            start_date = end_date - datetime.timedelta(days=days)

        label = self._period_label(start_date, end_date)
        recs = self.db.get_recommendations_by_date_range(start_date, end_date)

        result = self._compute_stats(recs, start_date, end_date, label)
        self.results = result
        logger.info(f"Periodic report ({label}): {result['total_predictions']} predictions, {result['win_rate']:.1%} win rate")
        self.log_end()
        return result

    def _compute_stats(self, recs: List[dict], start_date, end_date, label: str) -> Dict:
        total = len(recs)
        evaluated = [r for r in recs if r.get("prediction_accurate") is not None]
        accurate = sum(1 for r in evaluated if r["prediction_accurate"])
        total_eval = len(evaluated)
        win_rate = round(accurate / total_eval, 4) if total_eval > 0 else 0.0
        returns = [r.get("actual_close_pct") or 0 for r in evaluated]
        avg_return = round(sum(returns) / len(returns), 4) if returns else 0.0
        total_return = round(sum(returns), 4) if returns else 0.0
        max_gain = round(max(returns), 4) if returns else 0.0
        max_loss = round(min(returns), 4) if returns else 0.0

        dominant_counts = Counter()
        dominant_wins = Counter()
        for r in evaluated:
            scores = {k: r.get(k, 0) for k in FACTOR_LABELS}
            max_score = max(scores.values()) if scores else 0
            if max_score <= 0:
                continue
            top = [k for k, v in scores.items() if v == max_score]
            for k in top:
                dominant_counts[k] += 1
                if r["prediction_accurate"]:
                    dominant_wins[k] += 1

        signal_stats = {}
        for k in FACTOR_LABELS:
            if dominant_counts[k] > 0:
                signal_stats[k] = {
                    "label": FACTOR_LABELS[k],
                    "count": dominant_counts[k],
                    "win_rate": round(dominant_wins[k] / dominant_counts[k], 4),
                    "wins": dominant_wins[k],
                }

        best_signal = max(signal_stats.items(), key=lambda x: x[1]["win_rate"]) if signal_stats else (None, None)
        worst_signal = min(signal_stats.items(), key=lambda x: x[1]["win_rate"]) if signal_stats else (None, None)

        sector_data = defaultdict(lambda: {"count": 0, "wins": 0, "returns": []})
        for r in evaluated:
            sector = r.get("sector") or "Unknown"
            sector_data[sector]["count"] += 1
            if r["prediction_accurate"]:
                sector_data[sector]["wins"] += 1
            sector_data[sector]["returns"].append(r.get("actual_close_pct") or 0)

        sector_stats = {}
        for sector, sd in sector_data.items():
            if sd["count"] >= 2:
                sector_stats[sector] = {
                    "count": sd["count"],
                    "win_rate": round(sd["wins"] / sd["count"], 4),
                    "wins": sd["wins"],
                    "avg_return": round(sum(sd["returns"]) / len(sd["returns"]), 4),
                }

        top_sector = max(sector_stats.items(), key=lambda x: x[1]["win_rate"]) if sector_stats else (None, None)
        worst_sector = min(sector_stats.items(), key=lambda x: x[1]["win_rate"]) if sector_stats else (None, None)

        return {
            "report_label": label,
            "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_predictions": total,
            "evaluated_predictions": total_eval,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "total_return": total_return,
            "max_gain": max_gain,
            "max_loss": max_loss,
            "best_signal": best_signal[1],
            "best_signal_key": best_signal[0],
            "worst_signal": worst_signal[1],
            "worst_signal_key": worst_signal[0],
            "signal_stats": signal_stats,
            "top_sector": top_sector[1],
            "top_sector_key": top_sector[0],
            "worst_sector": worst_sector[1],
            "worst_sector_key": worst_sector[0],
            "sector_stats": sector_stats,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def generate_html(self, result: Optional[Dict] = None) -> str:
        d = result or self.results
        if not d or d.get("total_predictions", 0) == 0:
            return "<p>No data available for this period.</p>"

        wr = d.get("win_rate", 0)
        ar = d.get("avg_return", 0)
        wr_color = "#27ae60" if wr >= 0.5 else "#e74c3c"
        ar_color = "#27ae60" if ar > 0 else "#e74c3c"

        bs = d.get("best_signal", {}) or {}
        ws = d.get("worst_signal", {}) or {}
        ts = d.get("top_sector", {}) or {}
        wos = d.get("worst_sector", {}) or {}

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#1a1a3e;color:#e0e0e0;margin:0;padding:0;}}
.container{{max-width:700px;margin:0 auto;padding:30px;}}
.header{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:25px;border-radius:12px;margin-bottom:20px;text-align:center;}}
.header h1{{margin:0;font-size:22px;}}
.header p{{opacity:0.9;margin:5px 0 0;font-size:14px;}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px;}}
.card{{background:#0f0f23;border-radius:8px;padding:16px;border:1px solid #2a2a5e;}}
.card h3{{color:#667eea;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 6px;}}
.card .value{{font-size:26px;font-weight:bold;}}
table{{width:100%;border-collapse:collapse;margin-bottom:16px;}}
th{{background:#2a2a5e;padding:8px;text-align:left;font-size:11px;text-transform:uppercase;}}
td{{padding:8px;border-bottom:1px solid #2a2a5e;font-size:13px;}}
.footer{{text-align:center;color:#555;font-size:11px;margin-top:20px;}}
</style></head><body>
<div class="container">
<div class="header"><h1>StockIntel Weekly Report</h1><p>{d.get("report_label", "")}</p></div>
<div class="grid">
<div class="card"><h3>Predictions</h3><div class="value blue">{d['total_predictions']}</div></div>
<div class="card"><h3>Win Rate</h3><div class="value" style="color:{wr_color};">{wr:.1%}</div></div>
<div class="card"><h3>Avg Return</h3><div class="value" style="color:{ar_color};">{ar:+.2f}%</div></div>
<div class="card"><h3>Best Signal</h3><div class="value" style="color:#27ae60;">{bs.get("label","N/A")}</div><p style="font-size:11px;color:#888;">{bs.get("win_rate",0):.1%} win rate ({bs.get("count",0)} trades)</p></div>
</div>"""

        if d.get("signal_stats"):
            html += '<h3 style="color:#667eea;">Signal Performance</h3><table><thead><tr><th>Signal</th><th>Trades</th><th>Wins</th><th>Win Rate</th></tr></thead><tbody>'
            for k, ss in sorted(d["signal_stats"].items(), key=lambda x: x[1]["win_rate"], reverse=True):
                c = "#27ae60" if ss["win_rate"] >= 0.5 else "#e74c3c"
                html += f"<tr><td>{ss['label']}</td><td>{ss['count']}</td><td>{ss['wins']}</td><td style='color:{c};'>{ss['win_rate']:.1%}</td></tr>"
            html += "</tbody></table>"

        if d.get("sector_stats"):
            html += '<h3 style="color:#667eea;">Sector Breakdown</h3><table><thead><tr><th>Sector</th><th>Trades</th><th>Win Rate</th><th>Avg Return</th></tr></thead><tbody>'
            for sector, ss in sorted(d["sector_stats"].items(), key=lambda x: x[1]["win_rate"], reverse=True):
                c = "#27ae60" if ss["win_rate"] >= 0.5 else "#e74c3c"
                rc = "#27ae60" if ss["avg_return"] > 0 else "#e74c3c"
                html += f"<tr><td>{sector}</td><td>{ss['count']}</td><td style='color:{c};'>{ss['win_rate']:.1%}</td><td style='color:{rc};'>{ss['avg_return']:+.2f}%</td></tr>"
            html += "</tbody></table>"

        html += f"""<div style="margin-top:16px;padding:16px;background:#0f0f23;border-radius:8px;border:1px solid #2a2a5e;">
<h3 style="color:#667eea;margin:0 0 8px;">Summary</h3>
<p style="font-size:13px;color:#aaa;">{d['total_predictions']} predictions evaluated over this period.</p>
<p style="font-size:13px;color:#aaa;">Win rate: <strong style="color:{wr_color};">{wr:.1%}</strong> | Avg return: <strong style="color:{ar_color};">{ar:+.2f}%</strong></p>
<p style="font-size:13px;color:#aaa;">Best signal: <strong style="color:#27ae60;">{bs.get("label","N/A")}</strong> ({bs.get("win_rate",0):.1%}) | Worst: <strong style="color:#e74c3c;">{ws.get("label","N/A")}</strong> ({ws.get("win_rate",0):.1%})</p>
<p style="font-size:13px;color:#aaa;">Top sector: <strong style="color:#27ae60;">{ts.get("label","N/A") or d.get("top_sector_key","N/A")}</strong> | Worst sector: <strong style="color:#e74c3c;">{wos.get("label","N/A") or d.get("worst_sector_key","N/A")}</strong></p>
</div>
<div class="footer"><p>StockIntel Platform | Auto-generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></div>
</div></body></html>"""
        return html

    def print_report(self, result: Optional[Dict] = None):
        d = result or self.results
        if not d or d.get("total_predictions", 0) == 0:
            print("\n  No data available for this period.")
            return
        wr = d.get("win_rate", 0)
        ar = d.get("avg_return", 0)
        bs = d.get("best_signal", {}) or {}
        ws = d.get("worst_signal", {}) or {}
        ts = d.get("top_sector_key", "N/A")
        wos = d.get("worst_sector_key", "N/A")
        print("\n" + "=" * 70)
        print(f"  {d.get('report_label', 'Periodic Report')}")
        print("=" * 70)
        print(f"  Predictions: {d['total_predictions']}")
        print(f"  Win Rate:    {wr:.1%}")
        print(f"  Avg Return:  {ar:+.2f}%")
        print(f"  Total Return: {d.get('total_return',0):+.2f}%")
        print(f"  Best Signal: {bs.get('label','N/A')} ({bs.get('win_rate',0):.1%})")
        print(f"  Worst Signal: {ws.get('label','N/A')} ({ws.get('win_rate',0):.1%})")
        print(f"  Top Sector:  {ts}")
        print(f"  Worst Sector: {wos}")
        print("=" * 70)

    def _period_label(self, start: datetime.date, end: datetime.date) -> str:
        if (end - start).days <= 8:
            week_num = (end.day - 1) // 7 + 1
            month_name = end.strftime("%B")
            return f"{week_num}{['st','nd','rd','th'][min(week_num-1,3)]} week of {month_name} {end.year}"
        return f"{start.strftime('%b %d')} - {end.strftime('%b %d, %Y')} Report"
