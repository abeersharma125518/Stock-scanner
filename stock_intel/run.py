#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
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


if __name__ == "__main__":
    main()
