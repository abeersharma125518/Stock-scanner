import datetime
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from stock_intel.agents.base_agent import BaseAgent
from stock_intel.db.database import DatabaseManager

logger = logging.getLogger(__name__)

NUM_BINS = 10


class CalibrationAnalyzer(BaseAgent):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        super().__init__(db, config)

    def validate(self) -> bool:
        return True

    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.log_start()
        days = (context or {}).get("lookback_days", 90)
        recs = self.db.get_evaluated_recommendations(days=days)
        if not recs:
            logger.warning("No evaluated recommendations for calibration")
            return {"total": 0, "bins": [], "ece": 0.0, "mce": 0.0}

        bins = [[] for _ in range(NUM_BINS)]
        for r in recs:
            score = r.get("total_score") or 0
            accurate = r.get("prediction_accurate", False)
            idx = min(int(score * NUM_BINS), NUM_BINS - 1)
            bins[idx].append(accurate)

        bin_data = []
        total_accurate = 0
        total_count = 0
        ece_numerator = 0.0

        for i in range(NUM_BINS):
            low = i / NUM_BINS
            high = (i + 1) / NUM_BINS
            count = len(bins[i])
            if count == 0:
                bin_data.append({
                    "bin_label": f"{low:.0%}-{high:.0%}",
                    "bin_low": low, "bin_high": high,
                    "count": 0, "win_rate": None, "confidence": (low + high) / 2,
                    "gap": None,
                })
                continue
            win_count = sum(bins[i])
            win_rate = win_count / count
            confidence = (low + high) / 2
            gap = abs(win_rate - confidence)
            ece_numerator += gap * count
            total_accurate += win_count
            total_count += count
            bin_data.append({
                "bin_label": f"{low:.0%}-{high:.0%}",
                "bin_low": low, "bin_high": high,
                "count": count, "win_rate": round(win_rate, 4),
                "confidence": round(confidence, 4),
                "gap": round(gap, 4),
                "win_count": win_count,
            })

        ece = round(ece_numerator / total_count, 4) if total_count > 0 else 0.0
        mce = round(max((b.get("gap") or 0) for b in bin_data), 4) if bin_data else 0.0
        overall_accuracy = round(total_accurate / total_count, 4) if total_count > 0 else 0.0

        self.results = {
            "total_recommendations": total_count,
            "lookback_days": days,
            "overall_accuracy": overall_accuracy,
            "average_confidence": round(sum(b["confidence"] * b["count"] for b in bin_data if b["count"] > 0) / total_count, 4) if total_count > 0 else 0.0,
            "ece": ece,
            "mce": mce,
            "bins": bin_data,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        logger.info(f"Calibration: {total_count} recs, ECE={ece:.4f}, MCE={mce:.4f}")
        self.log_end()
        return self.results

    def build_reliability_svg(self, results: Optional[Dict] = None, width: int = 500, height: int = 400) -> str:
        data = results or self.results
        bins = data.get("bins", [])
        if not bins or not any(b["count"] > 0 for b in bins):
            return "<p>No calibration data available yet.</p>"

        margins = {"top": 40, "right": 30, "bottom": 50, "left": 60}
        plot_w = width - margins["left"] - margins["right"]
        plot_h = height - margins["top"] - margins["bottom"]

        def scale_x(val):
            return margins["left"] + val * plot_w

        def scale_y(val):
            return margins["top"] + (1.0 - val) * plot_h

        svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1a1a3e;border-radius:8px;">'

        svg += f'<text x="{width//2}" y="22" fill="#667eea" font-size="14" font-weight="bold" text-anchor="middle">Reliability Diagram</text>'

        svg += f'<text x="{margins["left"]-8}" y="{margins["top"]-5}" fill="#888" font-size="10" text-anchor="end">1.0</text>'
        svg += f'<text x="{margins["left"]-8}" y="{margins["top"]+plot_h//2}" fill="#888" font-size="10" text-anchor="end">0.5</text>'
        svg += f'<text x="{margins["left"]-8}" y="{margins["top"]+plot_h+4}" fill="#888" font-size="10" text-anchor="end">0.0</text>'

        for i in range(11):
            x = scale_x(i / 10)
            svg += f'<text x="{x}" y="{margins["top"]+plot_h+14}" fill="#888" font-size="9" text-anchor="middle">{i/10:.0%}</text>'

        svg += '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#333" stroke-width="1"/>'.format(
            scale_x(0), scale_y(0), scale_x(1), scale_y(0))
        svg += '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#333" stroke-width="1"/>'.format(
            scale_x(0), scale_y(1), scale_x(0), scale_y(0))

        svg += '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>'.format(
            scale_x(0), scale_y(0), scale_x(1), scale_y(1))

        svg += '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#2a2a5e" stroke-width="1"/>'.format(
            scale_x(0), scale_y(1), scale_x(1), scale_y(1))
        svg += '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#2a2a5e" stroke-width="1"/>'.format(
            scale_x(0), scale_y(0), scale_x(0), scale_y(0))

        svg += f'<text x="{scale_x(0.5)}" y="{margins["top"]+plot_h+34}" fill="#888" font-size="10" text-anchor="middle">Confidence</text>'
        svg += f'<text x="12" y="{margins["top"]+plot_h//2}" fill="#888" font-size="10" text-anchor="middle" transform="rotate(-90,12,{margins["top"]+plot_h//2})">Accuracy</text>'

        for b in bins:
            if b["count"] == 0:
                continue
            c = b["confidence"]
            wr = b["win_rate"]
            bw = plot_w / NUM_BINS * 0.7
            bar_x = scale_x(c) - bw / 2
            bar_h = wr * plot_h
            bar_y = scale_y(wr)
            gap = b.get("gap") or 0
            color = "#27ae60" if gap < 0.1 else "#f1c40f" if gap < 0.2 else "#e74c3c"
            svg += f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.7" rx="2"/>'
            if b["count"] >= 5:
                svg += f'<text x="{scale_x(c):.1f}" y="{bar_y - 4:.1f}" fill="#ccc" font-size="8" text-anchor="middle">{wr:.0%}</text>'
            svg += f'<circle cx="{scale_x(c):.1f}" cy="{scale_y(wr):.1f}" r="3" fill="{color}" stroke="#fff" stroke-width="1"/>'

        ece = data.get("ece", 0)
        mce = data.get("mce", 0)
        total = data.get("total_recommendations", 0)
        ece_color = "#27ae60" if ece < 0.05 else "#f1c40f" if ece < 0.1 else "#e74c3c"
        svg += f'<text x="{width-margins["right"]}" y="{margins["top"]+plot_h-30}" fill="#888" font-size="10" text-anchor="end">ECE: <tspan fill="{ece_color}">{ece:.1%}</tspan></text>'
        svg += f'<text x="{width-margins["right"]}" y="{margins["top"]+plot_h-16}" fill="#888" font-size="10" text-anchor="end">MCE: {mce:.1%}</text>'
        svg += f'<text x="{width-margins["right"]}" y="{margins["top"]+plot_h-2}" fill="#888" font-size="10" text-anchor="end">N={total}</text>'

        svg += '<rect x="{}" y="{}" width="12" height="12" fill="#27ae60" opacity="0.7" rx="2"/>'.format(
            scale_x(0.72), margins["top"] + 4)
        svg += f'<text x="{scale_x(0.72)+16}" y="{margins["top"]+13}" fill="#888" font-size="9">Gap &lt; 10%</text>'
        svg += '<rect x="{}" y="{}" width="12" height="12" fill="#f1c40f" opacity="0.7" rx="2"/>'.format(
            scale_x(0.72), margins["top"] + 20)
        svg += f'<text x="{scale_x(0.72)+16}" y="{margins["top"]+29}" fill="#888" font-size="9">Gap 10-20%</text>'
        svg += '<rect x="{}" y="{}" width="12" height="12" fill="#e74c3c" opacity="0.7" rx="2"/>'.format(
            scale_x(0.72), margins["top"] + 36)
        svg += f'<text x="{scale_x(0.72)+16}" y="{margins["top"]+45}" fill="#888" font-size="9">Gap &gt; 20%</text>'

        svg += '</svg>'
        return svg

    def print_report(self, results: Optional[Dict] = None):
        data = results or self.results
        if not data or data.get("total_recommendations", 0) == 0:
            print("\n  No calibration data available yet.")
            return
        bins = data.get("bins", [])
        print("\n" + "=" * 70)
        print(f"  CONFIDENCE CALIBRATION — Last {data['lookback_days']} Days")
        print("=" * 70)
        print(f"  Total: {data['total_recommendations']} | Overall Accuracy: {data.get('overall_accuracy',0):.1%}")
        print(f"  Avg Confidence: {data.get('average_confidence',0):.1%} | ECE: {data.get('ece',0):.1%} | MCE: {data.get('mce',0):.1%}")
        print()
        print(f"  {'Bin':<14} {'Count':<8} {'Win Rate':<10} {'Confidence':<12} {'Gap':<8}")
        print("  " + "-" * 52)
        for b in bins:
            if b["count"] == 0:
                print(f"  {b['bin_label']:<14} {'-':<8} {'-':<10} {'-':<12} {'-':<8}")
            else:
                gap = b.get("gap") or 0
                wr_color = "GREEN" if b["win_rate"] >= 0.5 else "RED"
                gap_char = "+" if b["win_rate"] > b["confidence"] else "-" if b["win_rate"] < b["confidence"] else "="
                print(f"  {b['bin_label']:<14} {b['count']:<8} {b['win_rate']:.1%}     {b['confidence']:.1%}       {gap:.1%} {gap_char}")
        print("=" * 70)
        if data.get("ece", 0) > 0.1:
            print("  WARNING: Calibration error > 10% — confidence scores need recalibration")
        elif data.get("ece", 0) > 0.05:
            print("  NOTE: Calibration error > 5% — consider recalibrating after more data")
        else:
            print("  Confidence scores are well-calibrated!")
