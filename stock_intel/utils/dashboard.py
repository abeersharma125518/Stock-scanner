import datetime
import json
import logging
import os
from typing import Dict, List, Optional
from stock_intel.db.database import DatabaseManager
from stock_intel.config.settings import CONFIG
from stock_intel.agents.attribution_engine import AttributionEngine, FACTOR_META
from stock_intel.agents.calibration import CalibrationAnalyzer

logger = logging.getLogger(__name__)


class Dashboard:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.data_dir = CONFIG.data_dir

    def generate_html_report(self) -> str:
        today = datetime.date.today()
        recs = self.db.get_latest_recommendations(limit=10)
        perf = self.db.get_performance_history(days=90)
        latest_perf = perf[-1] if perf else {}
        signal_perf = self.db.get_signal_performance_history(days=90)

        engine = AttributionEngine(self.db)
        attribution = engine.execute({"lookback_days": 90})
        cal = CalibrationAnalyzer(self.db)
        calibration = cal.execute({"lookback_days": 90})
        reliability_svg = cal.build_reliability_svg(calibration)

        wr_val = latest_perf.get("win_rate", 0) if latest_perf else None
        ar_val = latest_perf.get("avg_return", 0) if latest_perf else None
        tr_val = latest_perf.get("total_recs", 0) if latest_perf else 0
        sr_val = latest_perf.get("sharpe_ratio", 0) if latest_perf else None

        wr_cls = "green" if wr_val is not None and wr_val >= 0.5 else "red"
        ar_cls = "green" if ar_val is not None and ar_val > 0 else "red"
        wr_display = f"{wr_val:.1%}" if wr_val is not None else "N/A"
        ar_display = f"{ar_val:.2f}%" if ar_val is not None else "N/A"
        sr_display = f"{sr_val:.2f}" if sr_val is not None else "N/A"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>StockIntel Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f0f23;color:#e0e0e0;}}
.container{{max-width:1400px;margin:0 auto;padding:20px;}}
.header{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:30px;border-radius:12px;margin-bottom:24px;}}
.header h1{{font-size:28px;}}
.header p{{opacity:0.9;margin-top:5px;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:24px;}}
.card{{background:#1a1a3e;border-radius:10px;padding:20px;border:1px solid #2a2a5e;}}
.card h3{{color:#667eea;font-size:14px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}}
.card .value{{font-size:32px;font-weight:bold;}}
.card .value.green{{color:#27ae60;}}
.card .value.red{{color:#e74c3c;}}
.card .value.blue{{color:#3498db;}}
.card .value.gold{{color:#f1c40f;}}
table{{width:100%;border-collapse:collapse;}}
th{{background:#2a2a5e;padding:12px;text-align:left;font-size:12px;text-transform:uppercase;letter-spacing:1px;}}
td{{padding:12px;border-bottom:1px solid #2a2a5e;}}
tr:hover{{background:#1f1f45;}}
.badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold;}}
.badge.up{{background:#27ae60;color:#fff;}}
.badge.down{{background:#e74c3c;color:#fff;}}
.badge.volume{{background:#3498db;color:#fff;}}
.badge.news{{background:#f39c12;color:#fff;}}
.badge.sentiment{{background:#9b59b6;color:#fff;}}
.badge.insider{{background:#1abc9c;color:#fff;}}
.badge.technical{{background:#e67e22;color:#fff;}}
.badge.earnings{{background:#2ecc71;color:#fff;}}
.badge.momentum{{background:#e74c3c;color:#fff;}}
.rec-rank{{font-size:24px;font-weight:bold;color:#667eea;}}
@media(max-width:768px){{.grid{{grid-template-columns:1fr;}}}}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>StockIntel Platform</h1>
<p>Daily Stock Intelligence Report | {today.isoformat()}</p>
</div>
<div class="grid">
<div class="card"><h3>Win Rate</h3><div class="value {wr_cls}">{wr_display}</div><p style="color:#888;font-size:12px;">Latest period</p></div>
<div class="card"><h3>Avg Return</h3><div class="value {ar_cls}">{ar_display}</div><p style="color:#888;font-size:12px;">Per recommendation</p></div>
<div class="card"><h3>Total Scored</h3><div class="value blue">{tr_val}</div><p style="color:#888;font-size:12px;">Latest recommendations</p></div>
<div class="card"><h3>Sharpe Ratio</h3><div class="value gold">{sr_display}</div><p style="color:#888;font-size:12px;">Risk-adjusted return</p></div>
</div>
<h2 style="margin:24px 0 12px;color:#667eea;">Top 10 Recommendations</h2>
<table><thead><tr><th>#</th><th>Ticker</th><th>Score</th><th>Direction</th><th>Signals</th><th>Key Drivers</th></tr></thead><tbody>"""

        sig_map = {"unusual_volume":("volume","Vol"),"premarket_mover_up":("momentum","PM"),"positive_sentiment":("sentiment","Sent"),"news_catalyst":("news","News"),"insider_buying":("insider","Insider"),"upcoming_earnings":("earnings","Earn"),"oversold":("technical","OS"),"overbought":("technical","OB"),"technical_bullish":("technical","Tech"),"low_float":("momentum","Float"),"small_cap":("momentum","SC")}
        for r in recs:
            signal_badges = ""
            for s in (r.get("signals") or []):
                if isinstance(s, str):
                    cls, label = sig_map.get(s, ("neutral", s[:4]))
                    signal_badges += f'<span class="badge {cls}">{label}</span> '
            direction = f'<span class="badge {"up" if r.get("predicted_direction","up")=="up" else "down"}">{r.get("predicted_direction","?")}</span>'
            html += f"<tr><td class='rec-rank'>{r.get('rank','?')}</td><td><strong>{r.get('ticker','?')}</strong></td><td>{r.get('score',0):.3f}</td><td>{direction}</td><td>{signal_badges}</td><td style='font-size:12px;color:#aaa;'>{(r.get('explanation','') or '')[:120]}</td></tr>"

        html += "</tbody></table>"
        html += '<h2 style="margin:24px 0 12px;color:#667eea;">Signal Performance</h2><div class="grid">'
        for sig, sp_list in signal_perf.items():
            if sp_list:
                last = sp_list[-1]
                wr_val = last.get("win_rate",0)
                avg_ret = last.get("avg_return",0)
                cls = "volume" if "volume" in sig else "news" if "news" in sig else "sentiment" if "sentiment" in sig else "insider" if "insider" in sig else "earnings" if "earnings" in sig else "technical"
                html += f'<div class="card"><h3>{sig.replace("_"," ").title()}</h3><div class="value {"green" if wr_val>=0.5 else "red"}">{wr_val:.1%}</div><p style="font-size:12px;color:#888;">Win Rate | Avg Ret: <span class="{"#27ae60" if avg_ret>0 else "#e74c3c"}">{avg_ret:.2f}%</span></p></div>'
        html += "</div>"

        cal_total = calibration.get("total_recommendations", 0)
        if cal_total > 0:
            ece = calibration.get("ece", 0)
            ece_color = "#27ae60" if ece < 0.05 else "#f1c40f" if ece < 0.1 else "#e74c3c"
            ovr = calibration.get("overall_accuracy", 0)
            avg_conf = calibration.get("average_confidence", 0)
            html += '<h2 style="margin:24px 0 12px;color:#667eea;">Confidence Calibration (90d)</h2>'
            html += '<div class="grid">'
            html += f'<div class="card"><h3>Overall Accuracy</h3><div class="value {"green" if ovr >= 0.5 else "red"}">{ovr:.1%}</div><p style="color:#888;font-size:12px;">Across {cal_total} recommendations</p></div>'
            html += f'<div class="card"><h3>Avg Confidence</h3><div class="value blue">{avg_conf:.1%}</div><p style="color:#888;font-size:12px;">Mean predicted score</p></div>'
            html += f'<div class="card"><h3>ECE</h3><div class="value" style="color:{ece_color};">{ece:.1%}</div><p style="color:#888;font-size:12px;">Expected Calibration Error</p></div>'
            html += f'<div class="card"><h3>MCE</h3><div class="value gold">{calibration.get("mce",0):.1%}</div><p style="color:#888;font-size:12px;">Max Calibration Error</p></div>'
            html += '</div>'
            html += '<div style="display:flex;justify-content:center;margin-bottom:24px;">'
            html += reliability_svg
            html += '</div>'

        html += '<h2 style="margin:24px 0 12px;color:#667eea;">Signal Attribution — Dominant Factor Win Rates (90d)</h2>'
        html += '<div class="grid">'
        for k, meta in FACTOR_META.items():
            attr = attribution.get("factor_attributions", {}).get(k)
            if attr and attr["total_trades"] > 0:
                wr = attr["win_rate"]
                ar = attr["avg_return"]
                tc = attr["total_trades"]
                wr_color = "green" if wr >= 0.5 else "red"
                ar_color = "green" if ar > 0 else "red"
                html += f'<div class="card" style="border-left:4px solid {meta["color"]};"><h3>{meta["icon"]} {meta["label"]}</h3><div class="value {wr_color}">{wr:.1%}</div><p style="font-size:12px;color:#888;">Win Rate | <span style="color:{ar_color};">{ar:+.2f}%</span> avg | {tc} trades</p></div>'
        html += '</div>'

        html += '<h2 style="margin:24px 0 12px;color:#667eea;">Performance History (90 Days)</h2>'
        html += "<table><thead><tr><th>Date</th><th>Win Rate</th><th>Avg Return</th><th>Correct</th><th>Total</th><th>Sharpe</th></tr></thead><tbody>"
        for p in perf[-30:]:
            wr_val = p.get("win_rate",0)
            html += f"<tr><td>{p.get('date','?')}</td><td style='color:{'#27ae60' if wr_val and wr_val>=0.5 else '#e74c3c'};'>{wr_val:.1%}</td><td>{p.get('avg_return',0):.2f}%</td><td>{p.get('correct',0)}</td><td>{p.get('total_recs',0)}</td><td>{p.get('sharpe_ratio',0):.2f}</td></tr>"
        html += "</tbody></table>"

        proposals = self.db.get_proposals(status="proposed")
        if proposals:
            html += '<h2 style="margin:24px 0 12px;color:#667eea;">Open Strategy Proposals</h2>'
            for p in proposals:
                html += f'<div class="card" style="margin-bottom:12px;border-left:4px solid #f1c40f;">'
                html += f'<h3 style="color:#f1c40f;">#{p.id}: {p.title}</h3>'
                html += f'<p style="color:#aaa;font-size:13px;margin:6px 0;">{p.description[:200]}</p>'
                html += f'<p style="color:#888;font-size:11px;">{p.created_at.strftime("%Y-%m-%d %H:%M")} | {p.status}</p>'
                if p.proposed_changes:
                    html += '<table style="width:auto;margin-top:8px;"><thead><tr><th>Parameter</th><th>Current</th><th>Proposed</th><th>Delta</th></tr></thead><tbody>'
                    for k, v in p.proposed_changes.items():
                        try:
                            vf = float(v)
                            current = CONFIG.weights.as_dict().get(k, 0)
                            delta = vf - current
                            dc = "change" if delta > 0 else "change down"
                            html += f'<tr><td>{k}</td><td>{current:.0%}</td><td>{vf:.0%}</td><td class="{dc}">{delta:+.0%}</td></tr>'
                        except (ValueError, TypeError):
                            pass
                    html += '</tbody></table>'
                if p.arguments:
                    html += '<div style="margin-top:6px;font-size:12px;">'
                    for a in p.arguments:
                        icon = "🟢" if a.stance == "for" else "🔴"
                        html += f'<p style="color:#888;">{icon} <strong>{a.agent_name}</strong>: {a.argument[:150]}</p>'
                    html += '</div>'
                html += '</div>'

        recent_reports = self.db.get_research_reports(limit=1)
        if recent_reports:
            r = recent_reports[0]
            findings = r.findings_json or {}
            html += '<h2 style="margin:24px 0 12px;color:#667eea;">Latest Research Insights</h2>'
            html += f'<div class="grid">'
            html += f'<div class="card"><h3>Report Period</h3><div class="value blue">{r.period_start} to {r.period_end}</div></div>'
            html += f'<div class="card"><h3>Win Rate</h3><div class="value {"green" if (r.overall_win_rate or 0) >= 0.5 else "red"}">{r.overall_win_rate:.1%}</div><p style="color:#888;font-size:12px;">n={r.evaluated_count or 0}</p></div>'
            html += f'<div class="card"><h3>Avg Return</h3><div class="value {"green" if (r.overall_avg_return or 0) > 0 else "red"}">{r.overall_avg_return:+.2f}%</div></div>'
            html += f'<div class="card"><h3>Proposals</h3><div class="value gold">{len(r.proposals or [])}</div></div>'
            html += '</div>'

            wopt = findings.get("weight_optimization", {}).get("rankings", [])
            if wopt:
                html += '<h3 style="color:#667eea;margin:12px 0;">Factor Rankings (by win rate)</h3>'
                html += '<table><thead><tr><th>Factor</th><th>Win Rate</th><th>Current</th><th>Suggested</th></tr></thead><tbody>'
                for entry in wopt[:9]:
                    if entry.get("skip"): continue
                    f_label = entry.get("factor", "?")
                    wr_val = entry.get("win_rate", 0)
                    cw = entry.get("current_weight", 0)
                    sw = entry.get("suggested_weight", 0)
                    html += f"<tr><td>{f_label}</td><td style='color:{'#27ae60' if wr_val>=0.5 else '#e74c3c'}'>{wr_val:.1%}</td><td>{cw:.0%}</td><td>{sw:.0%}</td></tr>"
                html += '</tbody></table>'

        html += f"""<div style="margin-top:30px;padding:20px;background:#1a1a3e;border-radius:10px;border:1px solid #2a2a5e;"><h3 style="color:#667eea;">About StockIntel</h3><p style="color:#888;font-size:13px;">Scans 1000+ US stocks daily using Yahoo Finance, Reddit, SEC EDGAR, and FinViz. Detects unusual volume, premarket movers, earnings events, insider activity, news catalysts, and sentiment shifts. Each stock scored via configurable weighted system.</p></div>
<div style="margin-top:20px;text-align:center;color:#555;font-size:11px;"><p>StockIntel Platform | Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></div>
</div></body></html>"""

        output_path = os.path.join(self.data_dir, "dashboard.html")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Dashboard HTML written to {output_path}")
        return output_path

    def print_cli_dashboard(self):
        recs = self.db.get_latest_recommendations(limit=10)
        perf = self.db.get_performance_history(days=90)
        latest_perf = perf[-1] if perf else {}
        engine = AttributionEngine(self.db)
        attribution = engine.execute({"lookback_days": 90})
        cal = CalibrationAnalyzer(self.db)
        calibration = cal.execute({"lookback_days": 90})
        print("\n" + "=" * 80)
        print(f"  STOCKINTEL DASHBOARD - {datetime.date.today().isoformat()}")
        print("=" * 80)
        if latest_perf:
            print(f"\n  Performance: Win Rate {latest_perf.get('win_rate',0):.1%} | Avg Return {latest_perf.get('avg_return',0):.2f}% | Sharpe {latest_perf.get('sharpe_ratio',0):.2f}")
            print(f"  Correct: {latest_perf.get('correct',0)}/{latest_perf.get('total_recs',0)}")
        cal_total = calibration.get("total_recommendations", 0)
        if cal_total > 0:
            ece = calibration.get("ece", 0)
            ovr = calibration.get("overall_accuracy", 0)
            ece_tag = "GOOD" if ece < 0.05 else "WARN" if ece < 0.1 else "POOR"
            print(f"  Calibration: Accuracy {ovr:.1%} | ECE {ece:.1%} [{ece_tag}] | N={cal_total}")
        print(f"\n  {'Rank':<6} {'Ticker':<8} {'Score':<8} {'Dir':<6} {'Signals':<40}")
        print("  " + "-" * 68)
        for r in recs:
            signals = ", ".join(r.get("signals",[])[:3]) if r.get("signals") else ""
            direction = r.get("predicted_direction","?")
            print(f"  #{r.get('rank','?'):<4} {r.get('ticker','?'):<8} {r.get('score',0):.3f}  {direction:<6} {signals:<40}")
        fa = attribution.get("factor_attributions", {})
        if fa:
            print(f"\n  {'Factor':<22} {'Win Rate':<10} {'Avg Ret':<10} {'Trades':<8}")
            print("  " + "-" * 50)
            for k, meta in FACTOR_META.items():
                attr = fa.get(k)
                if attr and attr["total_trades"] > 0:
                    print(f"  {meta['label']:<22} {attr['win_rate']:.1%}      {attr['avg_return']:>+6.2f}%  {attr['total_trades']:<8}")
        print("\n" + "=" * 80)

    def save_json(self):
        recs = self.db.get_latest_recommendations(limit=10)
        perf = self.db.get_performance_history(days=90)
        signal_perf = self.db.get_signal_performance_history(days=90)
        engine = AttributionEngine(self.db)
        attribution = engine.execute({"lookback_days": 90})
        cal = CalibrationAnalyzer(self.db)
        calibration = cal.execute({"lookback_days": 90})
        data = {"date": datetime.date.today().isoformat(), "timestamp": datetime.datetime.now().isoformat(),
                "recommendations": recs, "performance": perf, "signal_performance": signal_perf,
                "attribution": attribution, "calibration": calibration}
        path = os.path.join(self.data_dir, "dashboard_data.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Dashboard JSON written to {path}")
        return path
