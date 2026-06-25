#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stock_intel.main import StockIntelPipeline
from stock_intel.db.database import DatabaseManager
from stock_intel.config.settings import CONFIG, AppConfig, ScoringWeights


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="StockIntel - Daily Stock Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  python -m stock_intel.run --full
  python -m stock_intel.run --research
  python -m stock_intel.run --proposals
  python -m stock_intel.run --approve 1
  python -m stock_intel.run --reject 1 --reason "Not enough data"
  python -m stock_intel.run --argue 1 --stance for --argument "..."
  python -m stock_intel.run --default-weights
  python -m stock_intel.run --scan
  python -m stock_intel.run --full --verbose
        """,
    )
    parser.add_argument("--full", action="store_true", help="Run full pipeline (scan -> score -> evaluate -> research -> improve)")
    parser.add_argument("--scan", action="store_true", help="Run scan phase only (Yahoo + FinViz)")
    parser.add_argument("--score", action="store_true", help="Run scan + news + sentiment + score phases")
    parser.add_argument("--evaluate", action="store_true", help="Run post-market evaluation only")
    parser.add_argument("--dashboard", action="store_true", help="Generate dashboard from existing DB data")
    parser.add_argument("--attribution", action="store_true", help="Run signal attribution report only")
    parser.add_argument("--calibration", action="store_true", help="Run confidence calibration report only")
    parser.add_argument("--postmortem", action="store_true", help="Run post-mortem analysis on failed predictions")
    parser.add_argument("--weekly-report", action="store_true", help="Generate weekly performance report")
    parser.add_argument("--monthly-report", action="store_true", help="Generate monthly performance report")
    parser.add_argument("--schedule", action="store_true", help="Run full pipeline in schedule mode (for cron/task scheduler)")
    parser.add_argument("--research", action="store_true", help="Run research + pattern discovery + improvement engine")
    parser.add_argument("--proposals", action="store_true", help="List all open strategy proposals")
    parser.add_argument("--proposals-all", action="store_true", help="List ALL proposals regardless of status")
    parser.add_argument("--approve", type=int, metavar="ID", help="Approve proposal by ID and apply changes to config")
    parser.add_argument("--reject", type=int, metavar="ID", help="Reject proposal by ID")
    parser.add_argument("--reason", type=str, default="", help="Reason for rejection (used with --reject)")
    parser.add_argument("--argue", type=int, metavar="ID", help="Add argument to proposal by ID")
    parser.add_argument("--stance", type=str, choices=["for", "against"], default="for", help="Stance for argument")
    parser.add_argument("--argument", type=str, default="", help="Argument text (used with --argue)")
    parser.add_argument("--default-weights", action="store_true", help="Reset weights to defaults")
    parser.add_argument("--test-email", action="store_true", help="Send a test email to verify SMTP config")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    db = DatabaseManager()
    pipeline = StockIntelPipeline(db)

    if args.default_weights:
        cfg = CONFIG
        cfg.weights = ScoringWeights()
        cfg.save()
        print("Weights reset to defaults.")
        return

    if args.approve:
        _approve_proposal(db, args.approve)
        return

    if args.reject:
        _reject_proposal(db, args.reject, args.reason)
        return

    if args.argue:
        _add_argument(db, args.argue, args.stance, args.argument)
        return

    if args.proposals or args.proposals_all:
        _list_proposals(db, all_status=args.proposals_all)
        return

    if args.test_email:
        _send_test_email(db)
        return

    if args.research:
        pipeline.run_evaluate_only()
        pipeline.run_research_phase()
        pipeline.run_pattern_phase()
        pipeline.run_improvement_phase()
        pipeline.run_dashboard_only()
        return

    if args.full or args.schedule:
        pipeline.run_full()
    elif args.scan:
        pipeline.run_scan_only()
    elif args.score:
        pipeline.run_score_only()
    elif args.evaluate:
        pipeline.run_evaluate_only()
    elif args.dashboard:
        pipeline.run_dashboard_only()
    elif args.attribution:
        pipeline.run_attribution_only()
    elif args.calibration:
        pipeline.run_calibration_only()
    elif args.postmortem:
        pipeline.run_post_mortem_only()
    elif args.weekly_report:
        pipeline.run_periodic_report("weekly")
    elif args.monthly_report:
        pipeline.run_periodic_report("monthly")
    else:
        parser.print_help()
        sys.exit(1)


def _list_proposals(db: DatabaseManager, all_status: bool = False):
    proposals = db.get_proposals(status=None if all_status else "proposed")
    if not proposals:
        print("\n  No proposals found.")
        return
    print(f"\n{'='*80}")
    print(f"  STRATEGY PROPOSALS ({len(proposals)} total)")
    print(f"{'='*80}")
    for p in proposals:
        status_tag = f"[{p.status.upper()}]" if p.status != "proposed" else ""
        print(f"\n  #{p.id} {status_tag} {p.title}")
        print(f"      {p.description[:150]}")
        print(f"      Created: {p.created_at.strftime('%Y-%m-%d %H:%M')}")
        if p.status == "proposed":
            print(f"      > Approve: python -m stock_intel.run --approve {p.id}")
            print(f"      > Reject:  python -m stock_intel.run --reject {p.id} --reason \"...\"")
        if p.arguments:
            for a in p.arguments:
                marker = "+" if a.stance == "for" else "-"
                print(f"      {marker} [{a.agent_name}] {a.argument[:100]}")
        if p.proposed_changes:
            print(f"      Changes:")
            for k, v in p.proposed_changes.items():
                current = CONFIG.weights.as_dict().get(k, 0)
                delta = float(v) - current
                direction = "+" if delta > 0 else ""
                print(f"        {k}: {current:.0%} -> {v:.0%} ({direction}{delta:.0%})")
    print()


def _approve_proposal(db: DatabaseManager, proposal_id: int):
    prop = db.get_proposal_by_id(proposal_id)
    if not prop:
        print(f"Proposal #{proposal_id} not found.")
        return
    if prop.status != "proposed":
        print(f"Proposal #{proposal_id} is already {prop.status}.")
        return

    changes = prop.proposed_changes or {}
    if not changes:
        print(f"Proposal #{proposal_id} has no changes to apply.")
        db.update_proposal_status(proposal_id, "approved")
        return

    cfg = CONFIG
    current = cfg.weights.as_dict()
    for key, value in changes.items():
        if hasattr(cfg.weights, key):
            setattr(cfg.weights, key, float(value))
            print(f"  {key}: {current.get(key, 0):.0%} -> {float(value):.0%}")

    cfg.save()
    db.update_proposal_status(proposal_id, "approved")
    print(f"\nProposal #{proposal_id} approved and applied. Config saved.")
    print("NOTE: Changes take effect on next pipeline run.")


def _reject_proposal(db: DatabaseManager, proposal_id: int, reason: str = ""):
    prop = db.get_proposal_by_id(proposal_id)
    if not prop:
        print(f"Proposal #{proposal_id} not found.")
        return
    db.update_proposal_status(proposal_id, "rejected")
    if reason:
        db.add_proposal_argument(proposal_id, "against", reason, agent_name="human")
    print(f"Proposal #{proposal_id} rejected.")
    if reason:
        print(f"  Reason: {reason}")


def _add_argument(db: DatabaseManager, proposal_id: int, stance: str, argument: str):
    prop = db.get_proposal_by_id(proposal_id)
    if not prop:
        print(f"Proposal #{proposal_id} not found.")
        return
    if not argument:
        argument = input("Enter your argument: ")
    db.add_proposal_argument(proposal_id, stance, argument, agent_name="human")
    print(f"Argument added to proposal #{proposal_id} ({stance}).")


def _send_test_email(db: DatabaseManager):
    import datetime
    import numpy as np
    from stock_intel.agents.attribution_engine import FACTOR_META, AttributionEngine
    from stock_intel.agents.calibration import CalibrationAnalyzer
    from stock_intel.utils.emailer import EmailReporter
    from stock_intel.utils.allocator import compute_allocations, fetch_benchmark_return

    recs = db.get_latest_recommendations(limit=10)
    evaluated = db.get_evaluated_recommendations(days=365)

    if evaluated:
        returns = [r["actual_return"] for r in evaluated if r["actual_return"] is not None]
        returns_1d = [r["return_1d"] for r in evaluated if r["return_1d"] is not None]
        returns_2d = [r["return_2d"] for r in evaluated if r["return_2d"] is not None]
        returns_5d = [r["return_5d"] for r in evaluated if r["return_5d"] is not None]
        correct = sum(1 for r in evaluated if r["prediction_accurate"])
        correct_1d = sum(1 for r in evaluated if r.get("prediction_accurate_1d"))
        correct_2d = sum(1 for r in evaluated if r.get("prediction_accurate_2d"))
        correct_5d = sum(1 for r in evaluated if r.get("prediction_accurate_5d"))
        spy_returns = [r["spy_return_pct"] for r in evaluated if r["spy_return_pct"] is not None]
        drawdowns = [r["max_drawdown"] for r in evaluated if r["max_drawdown"] is not None]
        perf = {
            "win_rate": round(correct / len(evaluated), 4) if evaluated else 0,
            "win_rate_1d": round(correct_1d / len(returns_1d), 4) if returns_1d else 0,
            "win_rate_2d": round(correct_2d / len(returns_2d), 4) if returns_2d else 0,
            "win_rate_5d": round(correct_5d / len(returns_5d), 4) if returns_5d else 0,
            "avg_return": round(float(np.mean(returns)), 4) if returns else 0,
            "avg_return_1d": round(float(np.mean(returns_1d)), 4) if returns_1d else 0,
            "avg_return_2d": round(float(np.mean(returns_2d)), 4) if returns_2d else 0,
            "avg_return_5d": round(float(np.mean(returns_5d)), 4) if returns_5d else 0,
            "max_drawdown": round(float(np.min(drawdowns)), 4) if drawdowns else 0,
            "spy_return_pct": round(float(np.mean(spy_returns)), 4) if spy_returns else 0,
            "sharpe_ratio": round(float(np.mean(returns) / np.std(returns) * np.sqrt(252)), 4) if len(returns) > 1 and np.std(returns) > 0 else 0,
            "evaluated_count": len(evaluated),
        }
    else:
        perf = {"win_rate": 0, "avg_return": 0, "sharpe_ratio": 0, "evaluated_count": 0}

    engine = AttributionEngine(db)
    attribution = engine.execute({"lookback_days": 90})
    cal = CalibrationAnalyzer(db)
    calibration = cal.execute({"lookback_days": 90})

    factor_perf = {}
    for k, meta in FACTOR_META.items():
        v = attribution.get("factor_attributions", {}).get(k)
        if v and v.get("total_trades", 0) > 0:
            factor_perf[meta["label"]] = {"win_rate": v.get("win_rate", 0),
                                          "avg_return": v.get("avg_return", 0),
                                          "total_trades": v.get("total_trades", 0)}

    open_pos = db.get_open_positions()
    open_pos_data = [
        {
            "ticker": p.ticker,
            "entry_date": p.entry_date.isoformat() if hasattr(p.entry_date, "isoformat") else str(p.entry_date),
            "entry_price": p.entry_price,
            "current_return": p.current_return,
            "days_held": (datetime.date.today() - p.entry_date).days,
            "days_remaining": max(0, (p.planned_exit_date - datetime.date.today()).days),
            "planned_exit": p.planned_exit_date.isoformat() if hasattr(p.planned_exit_date, "isoformat") else str(p.planned_exit_date),
        }
        for p in open_pos if p.status == "open"
    ]
    closed_since = datetime.date.today() - datetime.timedelta(days=1)
    closed_pos = db.get_closed_positions_since(closed_since)
    closed_pos_data = [
        {
            "ticker": p.ticker,
            "entry_date": p.entry_date.isoformat() if hasattr(p.entry_date, "isoformat") else str(p.entry_date),
            "exit_date": p.exit_date.isoformat() if hasattr(p.exit_date, "isoformat") else str(p.exit_date),
            "entry_price": p.entry_price,
            "exit_price": p.exit_price,
            "trade_return": p.trade_return,
            "holding_days": (p.exit_date - p.entry_date).days if p.exit_date and p.entry_date else 0,
        }
        for p in closed_pos
    ]
    portfolio_data = {"total_invested": 0, "portfolio_value": 0, "cumulative_return": 0,
                      "open_positions": len(open_pos_data), "total_closed_positions": len(closed_pos_data),
                      "closed_positions_today": len(closed_pos_data)}

    rec_dicts = [dict(r) for r in recs]
    alloc_list, cash_pct_alloc, _ = compute_allocations(rec_dicts)
    bmark_return = fetch_benchmark_return(days=5)

    context = {
        "date": datetime.date.today().isoformat(),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks_scanned": db.get_all_stored_tickers().__len__(),
        "recommendations_count": len(recs),
        "elapsed_seconds": 123.4,
        "recommendations": recs,
        "performance": perf,
        "research_insights": {"proposals_list": [],
                              "patterns_found": 0,
                              "calibration_summary": {"ece": calibration.get("ece", 0),
                                                       "accuracy": calibration.get("overall_accuracy", 0)}},
        "factor_performance": factor_perf,
        "open_positions": open_pos_data,
        "closed_positions": closed_pos_data,
        "portfolio": portfolio_data,
        "allocation_data": alloc_list,
        "cash_pct": cash_pct_alloc,
    }

    emailer = EmailReporter()
    if not emailer.enabled:
        print("\n  Email is disabled. Set email.enabled=true in config.json and provide SMTP credentials.\n")
        print("  Required config (in stock_intel/data/config.json):")
        print('    "email": {')
        print('      "smtp_server": "smtp.gmail.com",')
        print('      "smtp_port": 587,')
        print('      "sender_email": "your@gmail.com",')
        print('      "recipient_email": "you@example.com",')
        print('      "enabled": true')
        print("    }")
        print("\n  Or set environment variable:")
        print("    $env:STOCKINTEL_EMAIL_PASSWORD = 'your-app-password'")
        print()
        return

    result = emailer.send_daily_report(context)
    if result:
        print(f"\n  Test email sent! Check {CONFIG.email.recipient_email}\n")
    else:
        print(f"\n  Email send FAILED. Check logs above for details.\n")


if __name__ == "__main__":
    main()
