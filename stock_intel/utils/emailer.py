import datetime
import logging
import os
import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from typing import Dict, List, Optional
from stock_intel.config.settings import CONFIG
from stock_intel.utils.intraday import TradeSpec, next_market_date

logger = logging.getLogger(__name__)


class EmailReporter:
    def __init__(self):
        self.config = CONFIG.email
        self.enabled = self.config.enabled
        self._password: Optional[str] = None

    def _get_password(self) -> str:
        if self._password is not None:
            return self._password
        pw = self.config.sender_password
        if not pw:
            pw = os.environ.get(self.config.password_env_var, "")
        self._password = pw or ""
        return self._password

    def send_daily_report(self, context: dict) -> bool:
        if not self.enabled:
            logger.info("Email reporting disabled — set email.enabled=true in config.json")
            return False

        html = self._build_daily_html(context)
        date_str = context.get("date", datetime.date.today().isoformat())
        subject = f"StockIntel Daily Report - {date_str}"

        attachments = []
        data_dir = CONFIG.data_dir
        for fname in ("dashboard.html", "dashboard_data.json"):
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                attachments.append(fpath)

        return self._send_email(subject, html, attachments)

    def send_alert(self, alert_type: str, ticker: str, message: str, priority: str = "normal") -> bool:
        if not self.enabled:
            return False
        color = {"high": "#e74c3c", "medium": "#f39c12", "normal": "#3498db"}.get(priority, "#3498db")
        html = f"""<html><body style="font-family:Arial,sans-serif;">
            <h2 style="color:{color};">{alert_type}</h2>
            <p><strong>Stock:</strong> {ticker}</p>
            <p><strong>Message:</strong> {message}</p>
            <p><small>Generated at {datetime.datetime.now().isoformat()}</small></p>
        </body></html>"""
        subject = f"StockIntel Alert: {alert_type} - {ticker}"
        return self._send_email(subject, html)

    @staticmethod
    def _buy_sell_advice(signals: Optional[List[str]], buy_date: str) -> (str, str):
        spec = TradeSpec.from_signals(signals)
        return spec.buy_tip_text(buy_date), spec.sell_tip_text(buy_date)

    def _build_daily_html(self, ctx: dict) -> str:
        report_date = ctx.get("date", datetime.date.today().isoformat())
        run_ts = ctx.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        scanned = ctx.get("stocks_scanned", 0)
        rec_count = ctx.get("recommendations_count", 0)
        elapsed = ctx.get("elapsed_seconds", 0)
        elapsed_str = f"{elapsed:.0f}s" if elapsed < 120 else f"{elapsed / 60:.1f}min"
        next_market = next_market_date().strftime("%a %b %d")

        recs = ctx.get("recommendations", [])
        rec_rows = ""
        for r in recs[:10]:
            rank = r.get("rank", "?")
            ticker = r.get("ticker", "?")
            score = r.get("total_score", r.get("score", 0))
            direction = r.get("predicted_direction", "?")
            signals_list = r.get("signals", []) or []
            signals = ", ".join(signals_list[:4]) if signals_list else "-"
            buy_tip, sell_tip = self._buy_sell_advice(signals_list, next_market)
            dir_color = "#27ae60" if direction == "up" else "#e74c3c"
            rec_rows += f"""<tr>
                <td style="padding:8px;border-bottom:1px solid #eee;">#{rank}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;"><strong>{ticker}</strong></td>
                <td style="padding:8px;border-bottom:1px solid #eee;">{score:.3f}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;color:{dir_color};">{direction}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;font-size:12px;">{signals}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;font-size:11px;color:#27ae60;">{buy_tip}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;font-size:11px;color:#e74c3c;">{sell_tip}</td>
            </tr>"""

        open_positions = ctx.get("open_positions", [])
        open_rows = ""
        for p in open_positions:
            ret = p.get("current_return", 0)
            days_left = p.get("days_remaining", 0)
            ret_color = "#27ae60" if ret >= 0 else "#e74c3c"
            open_rows += f"""<tr>
                <td style="padding:8px;border-bottom:1px solid #eee;"><strong>{p.get("ticker", "?")}</strong></td>
                <td style="padding:8px;border-bottom:1px solid #eee;">{p.get("entry_date", "")}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;">${p.get("entry_price", 0):.2f}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;color:{ret_color};">{ret:+.2f}%</td>
                <td style="padding:8px;border-bottom:1px solid #eee;">{p.get("days_held", 0)}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;">{days_left}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;">{p.get("planned_exit", "")}</td>
            </tr>"""

        closed_positions = ctx.get("closed_positions", [])
        closed_rows = ""
        for p in closed_positions:
            ret = p.get("trade_return", 0)
            ret_color = "#27ae60" if ret >= 0 else "#e74c3c"
            closed_rows += f"""<tr>
                <td style="padding:8px;border-bottom:1px solid #eee;"><strong>{p.get("ticker", "?")}</strong></td>
                <td style="padding:8px;border-bottom:1px solid #eee;">{p.get("entry_date", "")}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;">{p.get("exit_date", "")}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;">{p.get("holding_days", 0)}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;">${p.get("entry_price", 0):.2f}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;">${p.get("exit_price", 0):.2f}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;color:{ret_color};font-weight:bold;">{ret:+.2f}%</td>
            </tr>"""

        portfolio = ctx.get("portfolio", {})
        port_html = ""
        if portfolio:
            cum_ret = portfolio.get("cumulative_return", 0)
            bmark = portfolio.get("benchmark_return")
            alpha = portfolio.get("alpha")
            bmark_str = f"{bmark:+.2f}%" if bmark is not None else "N/A"
            alpha_str = f"{alpha:+.2f}%" if alpha is not None else "N/A"
            alpha_color = "#27ae60" if (alpha or 0) >= 0 else "#e74c3c"
            port_html = f"""<table style="width:100%;border-collapse:collapse;">
                <tr><td style="padding:6px;"><strong>Portfolio Value</strong></td><td style="padding:6px;">${portfolio.get('portfolio_value', 0):,.2f}</td></tr>
                <tr><td style="padding:6px;"><strong>Total Invested</strong></td><td style="padding:6px;">${portfolio.get('total_invested', 0):,.2f}</td></tr>
                <tr><td style="padding:6px;"><strong>Cumulative Return</strong></td><td style="padding:6px;color:{'#27ae60' if cum_ret>=0 else '#e74c3c'};font-weight:bold;">{cum_ret:+.2f}%</td></tr>
                <tr><td style="padding:6px;"><strong>Benchmark (SPY)</strong></td><td style="padding:6px;">{bmark_str}</td></tr>
                <tr><td style="padding:6px;"><strong>Alpha</strong></td><td style="padding:6px;color:{alpha_color};font-weight:bold;">{alpha_str}</td></tr>
                <tr><td style="padding:6px;"><strong>Open Positions</strong></td><td style="padding:6px;">{portfolio.get('open_positions', 0)}</td></tr>
                <tr><td style="padding:6px;"><strong>Total Closed</strong></td><td style="padding:6px;">{portfolio.get('total_closed_positions', 0)}</td></tr>
                <tr><td style="padding:6px;"><strong>Today&apos;s Closed</strong></td><td style="padding:6px;">{portfolio.get('closed_positions_today', 0)}</td></tr>
            </table>"""

        allocation_data = ctx.get("allocation_data", [])
        cash_pct = ctx.get("cash_pct", 0)
        alloc_rows = ""
        for a in allocation_data:
            lab = a.get("conviction", "N/A")
            lab_color = {"Very High": "#8e44ad", "High": "#27ae60", "Medium": "#f39c12", "Speculative": "#e74c3c"}.get(lab, "#888")
            alloc_rows += f"""<tr>
                <td style="padding:8px;border-bottom:1px solid #eee;"><strong>{a.get("ticker", "?")}</strong></td>
                <td style="padding:8px;border-bottom:1px solid #eee;">{a.get("total_score", 0):.3f}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;color:{lab_color};font-weight:bold;">{lab}</td>
                <td style="padding:8px;border-bottom:1px solid #eee;">{a.get("allocation_pct", 0):.1f}%</td>
            </tr>"""
        if cash_pct > 0:
            alloc_rows += f"""<tr>
                <td style="padding:8px;border-bottom:1px solid #eee;"><strong>CASH</strong></td>
                <td style="padding:8px;border-bottom:1px solid #eee;">—</td>
                <td style="padding:8px;border-bottom:1px solid #eee;color:#888;">Reserve</td>
                <td style="padding:8px;border-bottom:1px solid #eee;">{cash_pct:.1f}%</td>
            </tr>"""

        best3 = recs[:3] if recs else []
        best_html = ""
        for r in best3:
            ticker = r.get("ticker", "?")
            score = r.get("total_score", r.get("score", 0))
            direction = r.get("predicted_direction", "?")
            explanation = r.get("explanation", "") or ""
            signals_list = r.get("signals", []) or []
            buy_tip, sell_tip = self._buy_sell_advice(signals_list, next_market)
            best_html += f"""<div style="margin-bottom:12px;padding:12px;background:#f8f9fa;border-radius:8px;border-left:4px solid #667eea;">
                <div style="font-size:18px;font-weight:bold;color:#333;">{ticker} <span style="font-size:14px;color:#888;">Score {score:.3f} | predicted {direction}</span></div>
                <p style="margin:6px 0 0;font-size:13px;color:#666;">{explanation[:200]}</p>
                <div style="margin-top:8px;font-size:12px;">
                    <span style="color:#27ae60;">&#9650; {buy_tip}</span>
                    <span style="margin-left:16px;color:#e74c3c;">&#9660; {sell_tip}</span>
                </div>
            </div>"""

        perf = ctx.get("performance", {})
        win_rate = perf.get("win_rate", 0)
        wr_1d = perf.get("win_rate_1d", 0)
        wr_2d = perf.get("win_rate_2d", 0)
        wr_5d = perf.get("win_rate_5d", 0)
        avg_ret = perf.get("avg_return", 0)
        avg_ret_1d = perf.get("avg_return_1d", 0)
        avg_ret_2d = perf.get("avg_return_2d", 0)
        avg_ret_5d = perf.get("avg_return_5d", 0)
        max_dd = perf.get("max_drawdown", 0)
        spy_ret = perf.get("spy_return_pct", 0)
        sharpe = perf.get("sharpe_ratio", 0)
        eval_count = perf.get("evaluated_count", perf.get("total_recs", 0))
        wr_color = "#27ae60" if win_rate >= 0.5 else "#e74c3c"
        perf_html = f"""<table style="width:100%;border-collapse:collapse;">
            <tr><td style="padding:6px;"><strong>Overall Win Rate</strong></td><td style="padding:6px;color:{wr_color};font-weight:bold;">{win_rate:.1%}</td></tr>
            <tr><td style="padding:6px 6px 6px 24px;font-size:12px;">At +1 day</td><td style="padding:6px;color:{'#27ae60' if wr_1d>=0.5 else '#e74c3c'};font-size:12px;">{wr_1d:.1%} (avg {avg_ret_1d:+.2f}%)</td></tr>
            <tr><td style="padding:6px 6px 6px 24px;font-size:12px;">At +2 days</td><td style="padding:6px;color:{'#27ae60' if wr_2d>=0.5 else '#e74c3c'};font-size:12px;">{wr_2d:.1%} (avg {avg_ret_2d:+.2f}%)</td></tr>
            <tr><td style="padding:6px 6px 6px 24px;font-size:12px;">At +5 days</td><td style="padding:6px;color:{'#27ae60' if wr_5d>=0.5 else '#e74c3c'};font-size:12px;">{wr_5d:.1%} (avg {avg_ret_5d:+.2f}%)</td></tr>
            <tr><td style="padding:6px;"><strong>Avg Return (best horizon)</strong></td><td style="padding:6px;color:{'#27ae60' if avg_ret > 0 else '#e74c3c'};">{avg_ret:+.2f}%</td></tr>
            <tr><td style="padding:6px;"><strong>Max Drawdown</strong></td><td style="padding:6px;color:#e74c3c;">{max_dd:.2%}</td></tr>
            <tr><td style="padding:6px;"><strong>Sharpe Ratio</strong></td><td style="padding:6px;">{sharpe:.2f}</td></tr>
            <tr><td style="padding:6px;"><strong>SPY (same 5d period)</strong></td><td style="padding:6px;color:{'#27ae60' if spy_ret > 0 else '#e74c3c'};">{spy_ret:+.2f}%</td></tr>
            <tr><td style="padding:6px;"><strong>Evaluated Trades</strong></td><td style="padding:6px;">{eval_count}</td></tr>
        </table>"""

        factor_perf = ctx.get("factor_performance", {})
        factor_rows = ""
        for k, v in factor_perf.items():
            wr = v.get("win_rate", 0)
            ar = v.get("avg_return", 0)
            tc = v.get("total_trades", 0)
            if tc > 0:
                factor_rows += f"<tr><td style='padding:4px;'>{k}</td><td style='padding:4px;color:{'#27ae60' if wr>=0.5 else '#e74c3c'};'>{wr:.1%}</td><td style='padding:4px;'>{ar:+.2f}%</td><td style='padding:4px;'>{tc}</td></tr>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#333;margin:0;padding:0;background:#f4f4f6;}}
.container{{max-width:700px;margin:0 auto;padding:20px;}}
.header{{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:24px;border-radius:10px;margin-bottom:20px;}}
.header h1{{margin:0 0 4px;font-size:24px;}}
.header p{{margin:0;opacity:0.9;font-size:14px;}}
.section{{background:white;border-radius:8px;padding:16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.08);}}
.section h2{{margin:0 0 12px;font-size:16px;color:#667eea;text-transform:uppercase;letter-spacing:1px;}}
table{{width:100%;border-collapse:collapse;font-size:13px;}}
th{{background:#f8f9fa;padding:8px 6px;text-align:left;font-size:11px;text-transform:uppercase;color:#888;letter-spacing:0.5px;border-bottom:2px solid #eee;}}
td{{padding:8px 6px;border-bottom:1px solid #eee;}}
tr:hover{{background:#fafafa;}}
.tag{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:bold;margin-right:4px;}}
.tag-open{{background:#e8f5e9;color:#27ae60;}}
.tag-closed{{background:#e3f2fd;color:#1976d2;}}
.footer{{text-align:center;font-size:11px;color:#aaa;margin-top:20px;}}
</style></head>
<body>
<div class="container">
<div class="header">
<h1>StockIntel Daily Report</h1>
<p>{report_date} | Pipeline completed</p>
</div>

<div class="section">
<h2>Executive Summary</h2>
<table>
<tr><td style="width:180px;"><strong>Prediction Timestamp</strong></td><td>{run_ts}</td></tr>
<tr><td><strong>Prediction For</strong></td><td>{next_market} — hold 1–5 trading days</td></tr>
<tr><td><strong>Stocks Scanned</strong></td><td>{scanned:,}</td></tr>
<tr><td><strong>Recommendations</strong></td><td>{rec_count}</td></tr>
<tr><td><strong>Pipeline Runtime</strong></td><td>{elapsed_str}</td></tr>
</table>
</div>

<div class="section">
<h2>NEW PICKS</h2>
<table>
<thead><tr><th>Rank</th><th>Ticker</th><th>Score</th><th>Dir</th><th>Signals</th><th style="color:#27ae60;">Buy Timing</th><th style="color:#e74c3c;">Sell Timing</th></tr></thead>
<tbody>{rec_rows}</tbody>
</table>
</div>

<div class="section">
<h2>PORTFOLIO ALLOCATION</h2>
{'<table><thead><tr><th>Ticker</th><th>Score</th><th>Conviction</th><th>Allocation %</th></tr></thead><tbody>' + alloc_rows + '</tbody></table>' if alloc_rows else '<p style="color:#888;">No allocation data — recommendations pending.</p>'}
</div>

<div class="section">
<h2>OPEN POSITIONS</h2>
{'<table><thead><tr><th>Ticker</th><th>Entry Date</th><th>Entry Price</th><th>Unrealized P/L</th><th>Days Held</th><th>Days Left</th><th>Planned Exit</th></tr></thead><tbody>' + open_rows + '</tbody></table>' if open_rows else '<p style="color:#888;">No open positions.</p>'}
</div>

<div class="section">
<h2>CLOSED POSITIONS</h2>
{'<table><thead><tr><th>Ticker</th><th>Entry Date</th><th>Exit Date</th><th>Held</th><th>Entry $</th><th>Exit $</th><th>Realized P/L</th></tr></thead><tbody>' + closed_rows + '</tbody></table>' if closed_rows else '<p style="color:#888;">No positions closed since last report.</p>'}
</div>

<div class="section">
<h2>Portfolio Summary</h2>
{port_html if port_html else '<p style="color:#888;">Portfolio tracking initializing — data will accumulate after trade execution begins.</p>'}
</div>

<div class="section">
<h2>Performance Summary</h2>
{perf_html}
</div>

<div class="section">
<h2>Factor Performance</h2>
{'<table><thead><tr><th>Factor</th><th>Win Rate</th><th>Avg Ret</th><th>Trades</th></tr></thead><tbody>' + factor_rows + '</tbody></table>' if factor_rows else '<p style="color:#888;">No evaluated trades yet — data accumulates as predictions are verified.</p>'}
</div>

<div class="footer">
<p>Generated automatically by StockIntel Platform</p>
</div>
</div>
</body>
</html>"""

    def _send_email(self, subject: str, html_body: str, attachments: Optional[List[str]] = None) -> bool:
        password = self._get_password()
        if not self.config.sender_email or not password or not self.config.recipient_email:
            logger.warning("Email not configured: missing sender, password, or recipient")
            return False
        try:
            if attachments:
                msg = MIMEMultipart("mixed")
                alt = MIMEMultipart("alternative")
                alt.attach(MIMEText(html_body, "html"))
                msg.attach(alt)
            else:
                msg = MIMEMultipart("alternative")
                msg.attach(MIMEText(html_body, "html"))

            msg["Subject"] = subject
            msg["From"] = self.config.sender_email
            msg["To"] = self.config.recipient_email

            if attachments:
                for fpath in attachments:
                    fname = os.path.basename(fpath)
                    with open(fpath, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f"attachment; filename={fname}")
                    msg.attach(part)

            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.sender_email, password)
                server.sendmail(self.config.sender_email, self.config.recipient_email, msg.as_string())
            logger.info(f"Email sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False
