import datetime
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from stock_intel.config.settings import CONFIG

logger = logging.getLogger(__name__)


class EmailReporter:
    def __init__(self):
        self.config = CONFIG.email
        self.enabled = self.config.enabled

    def send_daily_report(self, recommendations: List[Dict], performance: Optional[Dict] = None, dashboard_url: str = "") -> bool:
        if not self.enabled:
            logger.info("Email reporting is disabled")
            return False
        html = self._build_report_html(recommendations, performance, dashboard_url)
        subject = f"StockIntel Daily Report - {datetime.date.today().isoformat()}"
        return self._send_email(subject, html)

    def send_alert(self, alert_type: str, ticker: str, message: str, priority: str = "normal") -> bool:
        if not self.enabled:
            return False
        html = f"""
        <html><body style="font-family:Arial,sans-serif;">
            <h2 style="color:{'#e74c3c' if priority == 'high' else '#f39c12' if priority == 'medium' else '#3498bc'};">{alert_type}</h2>
            <p><strong>Stock:</strong> {ticker}</p>
            <p><strong>Message:</strong> {message}</p>
            <p><small>Generated at {datetime.datetime.now().isoformat()}</small></p>
        </body></html>"""
        subject = f"StockIntel Alert: {alert_type} - {ticker}"
        return self._send_email(subject, html)

    def _build_report_html(self, recommendations: List[Dict], performance: Optional[Dict], dashboard_url: str) -> str:
        today = datetime.date.today().isoformat()
        rec_rows = ""
        for r in recommendations:
            direction_icon = "\U0001f7e2" if r.get("predicted_direction","up") == "up" else "\U0001f534"
            signals = ", ".join(r.get("signals",[])[:4]) if r.get("signals") else "N/A"
            price = r.get("component_scores",{}).get("price", "$?")
            rec_rows += f"<tr><td style='padding:8px;border-bottom:1px solid #ddd;'>#{r['rank']}</td><td style='padding:8px;border-bottom:1px solid #ddd;'><strong>{r['ticker']}</strong></td><td style='padding:8px;border-bottom:1px solid #ddd;'>{r['total_score']:.3f}</td><td style='padding:8px;border-bottom:1px solid #ddd;'>{direction_icon} {r.get('predicted_direction','?')}</td><td style='padding:8px;border-bottom:1px solid #ddd;'>{signals}</td></tr>"

        perf_section = ""
        if performance:
            perf_color = "#27ae60" if performance.get("win_rate",0) >= 0.5 else "#e74c3c"
            perf_section = f"""
            <h3>Performance Summary</h3>
            <table style="width:100%;border-collapse:collapse;margin-top:10px;">
                <tr><td style="padding:6px;"><strong>Win Rate:</strong></td><td style="padding:6px;color:{perf_color};"><strong>{performance.get("win_rate",0):.1%}</strong></td></tr>
                <tr><td style="padding:6px;"><strong>Avg Return:</strong></td><td style="padding:6px;">{performance.get("avg_return",0):.2f}%</td></tr>
                <tr><td style="padding:6px;"><strong>Correct/Total:</strong></td><td style="padding:6px;">{performance.get("correct_count",0)}/{performance.get("evaluated_count",0)}</td></tr>
                <tr><td style="padding:6px;"><strong>Sharpe:</strong></td><td style="padding:6px;">{performance.get("sharpe_ratio",0):.2f}</td></tr>
            </table>"""

        return f"""<html><head><style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#333;}}.container{{max-width:800px;margin:0 auto;padding:20px;}}.header{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:20px;border-radius:10px;}}table{{width:100%;border-collapse:collapse;}}th{{background:#f5f5f5;padding:10px;text-align:left;}}.footer{{margin-top:20px;font-size:12px;color:#999;}}</style></head><body><div class='container'><div class='header'><h1>StockIntel Daily Report</h1><p>{today} | Top {len(recommendations)} Opportunities</p></div>{perf_section}<h3>Top Recommendations</h3><table><thead><tr><th>Rank</th><th>Ticker</th><th>Score</th><th>Dir</th><th>Signals</th></tr></thead><tbody>{rec_rows}</tbody></table><div class='footer'><p>Generated automatically by StockIntel Platform</p>{f'<p><a href="{dashboard_url}">View Dashboard</a></p>' if dashboard_url else ''}</div></div></body></html>"""

    def _send_email(self, subject: str, html_body: str) -> bool:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.config.sender_email
            msg["To"] = self.config.recipient_email
            msg.attach(MIMEText(html_body, "html"))
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.sender_email, self.config.sender_password)
                server.sendmail(self.config.sender_email, self.config.recipient_email, msg.as_string())
            logger.info(f"Email sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False
