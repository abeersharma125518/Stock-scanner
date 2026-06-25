import datetime
import logging
import os
import time
from typing import Dict, Optional
from stock_intel.config.settings import CONFIG
from stock_intel.db.database import DatabaseManager
from stock_intel.agents.scanner_agent import ScannerAgent
from stock_intel.agents.news_analyst import NewsAnalyst
from stock_intel.agents.sentiment_analyst import SentimentAnalyst
from stock_intel.agents.scoring_engine import ScoringEngine
from stock_intel.agents.performance_tracker import PerformanceTracker
from stock_intel.agents.attribution_engine import AttributionEngine
from stock_intel.agents.calibration import CalibrationAnalyzer
from stock_intel.agents.post_mortem import PostMortemEngine
from stock_intel.agents.periodic_report import PeriodicReporter
from stock_intel.agents.research_agent import ResearchAgent
from stock_intel.agents.pattern_discovery import PatternDiscoveryAgent
from stock_intel.agents.improvement_engine import ImprovementEngine
from stock_intel.utils.dashboard import Dashboard
from stock_intel.utils.emailer import EmailReporter
from stock_intel.agents.attribution_engine import FACTOR_META

logger = logging.getLogger(__name__)


class StockIntelPipeline:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self.scanner = ScannerAgent(self.db)
        self.news_analyst = NewsAnalyst(self.db)
        self.sentiment_analyst = SentimentAnalyst(self.db)
        self.scoring_engine = ScoringEngine(self.db)
        self.performance_tracker = PerformanceTracker(self.db)
        self.attribution_engine = AttributionEngine(self.db)
        self.calibration_analyzer = CalibrationAnalyzer(self.db)
        self.post_mortem_engine = PostMortemEngine(self.db)
        self.periodic_reporter = PeriodicReporter(self.db)
        self.research_agent = ResearchAgent(self.db)
        self.pattern_discovery = PatternDiscoveryAgent(self.db)
        self.improvement_engine = ImprovementEngine(self.db)
        self.dashboard = Dashboard(self.db)
        self.emailer = EmailReporter()
        self.context: Dict = {}

    def run_scan_phase(self) -> Dict:
        logger.info("=== PHASE 1: Scan ===")
        results = self.scanner.execute()
        self.context["scanner_results"] = results
        logger.info(f"Scan complete: {len(results.get('all_snapshots', {}))} snapshots, "
                     f"{len(results.get('unusual_volume', []))} unusual volume, "
                     f"{len(results.get('premarket_movers', []))} premarket movers")
        return results

    def run_news_phase(self) -> Dict:
        logger.info("=== PHASE 2: News Analysis ===")
        results = self.news_analyst.execute({"scanner_results": self.context.get("scanner_results", {})})
        self.context["news_results"] = results
        logger.info(f"News complete: {results.get('total_articles', 0)} articles, "
                     f"{len(results.get('catalyst_stocks', []))} catalysts")
        return results

    def run_sentiment_phase(self) -> Dict:
        logger.info("=== PHASE 3: Sentiment Analysis ===")
        results = self.sentiment_analyst.execute({
            "scanner_results": self.context.get("scanner_results", {}),
            "news_results": self.context.get("news_results", {}),
        })
        self.context["sentiment_results"] = results
        logger.info(f"Sentiment complete: {len(results.get('sentiment_scores', {}))} tickers scored")
        return results

    def run_scoring_phase(self) -> Dict:
        logger.info("=== PHASE 4: Scoring ===")
        results = self.scoring_engine.execute(self.context)
        self.context["scoring_results"] = results
        logger.info(f"Scoring complete: {results.get('total_scored', 0)} scored, "
                     f"{len(results.get('recommendations', []))} recommended")
        return results

    def run_calibration_phase(self) -> Dict:
        logger.info("=== PHASE 5a: Confidence Calibration ===")
        results = self.calibration_analyzer.execute({"lookback_days": 90})
        self.context["calibration_results"] = results
        self.calibration_analyzer.print_report(results)
        return results

    def run_attribution_phase(self) -> Dict:
        logger.info("=== PHASE 5b: Signal Attribution ===")
        results = self.attribution_engine.execute({"lookback_days": 90})
        self.context["attribution_results"] = results
        self.attribution_engine.print_report(results)
        return results

    def run_report_phase(self) -> Dict:
        logger.info("=== PHASE 5: Report & Dashboard ===")
        attribution = self.attribution_engine.execute({"lookback_days": 90})
        self.context["attribution_results"] = attribution
        self.dashboard.generate_html_report()
        self.dashboard.print_cli_dashboard()
        self.dashboard.save_json()
        recs = self.context.get("scoring_results", {}).get("recommendations", [])
        logger.info(f"Dashboard generated, {len(recs)} recommendations")
        return {"dashboard_generated": True, "recommendations_count": len(recs)}

    def run_evaluation_phase(self) -> Dict:
        logger.info("=== PHASE 6: Post-Market Evaluation ===")
        results = self.performance_tracker.execute()
        self.context["evaluation_results"] = results
        logger.info(f"Evaluation complete: {results.get('evaluated_count', 0)} evaluated, "
                     f"win rate {results.get('win_rate', 0):.1%}")
        return results

    MIN_EVALUATED_FOR_RESEARCH = 10

    def _has_enough_data(self) -> bool:
        recs = self.db.get_all_evaluated_recommendations()
        return len(recs) >= self.MIN_EVALUATED_FOR_RESEARCH

    def run_research_phase(self) -> Dict:
        logger.info("=== PHASE 8: Research & Pattern Discovery ===")
        if not self._has_enough_data():
            logger.info(f"Skipping research: need {self.MIN_EVALUATED_FOR_RESEARCH}+ evaluated recs")
            return {"skipped": True, "reason": "insufficient_data"}

        research = self.research_agent.execute({"period": "monthly", "lookback_days": 90})
        self.context["research_results"] = research
        self.research_agent.print_research_summary(research)
        logger.info(f"Research: {research.get('total_recommendations', 0)} recs, "
                     f"{len(research.get('proposals', []))} proposals")
        return research

    def run_pattern_phase(self) -> Dict:
        logger.info("=== PHASE 8b: Pattern Discovery ===")
        if not self._has_enough_data():
            return {"skipped": True, "reason": "insufficient_data"}

        patterns = self.pattern_discovery.execute({"lookback_days": 90})
        self.context["pattern_results"] = patterns
        logger.info(f"Pattern discovery: {len(patterns.get('discoveries', {}).get('multi_factor_rules', {}).get('rules', []))} rules")
        return patterns

    def run_improvement_phase(self) -> Dict:
        logger.info("=== PHASE 8c: Improvement Engine ===")
        if not self._has_enough_data():
            return {"skipped": True, "reason": "insufficient_data"}

        results = self.improvement_engine.execute({
            "research_results": self.context.get("research_results", {}),
            "pattern_results": self.context.get("pattern_results", {}),
            "calibration_results": self.context.get("calibration_results", {}),
            "post_mortem_results": self.context.get("post_mortem_results", {}),
        })
        self.context["improvement_results"] = results
        self.improvement_engine.print_report(results)

        for prop_data in results.get("proposals", []):
            report_id = self.context.get("research_results", {}).get("report_id")
            if report_id:
                self.db.save_strategy_proposal(report_id, prop_data)

        logger.info(f"Improvement: {results.get('total', 0)} proposals")
        return results

    def run_post_mortem_phase(self) -> Dict:
        logger.info("=== PHASE 7: Post-Mortem (Why Was I Wrong?) ===")
        results = self.post_mortem_engine.execute({"lookback_days": 90})
        self.context["post_mortem_results"] = results
        self.post_mortem_engine.print_post_mortems(90)
        logger.info(f"Post-mortem: {results.get('analyzed', 0)} failures analyzed")
        return results

    def run_full(self) -> Dict:
        logger.info("=" * 60)
        logger.info("STOCKINTEL FULL PIPELINE START")
        logger.info("=" * 60)
        start = time.time()
        self.run_scan_phase()
        self.run_news_phase()
        self.run_sentiment_phase()
        self.run_scoring_phase()
        self.run_report_phase()
        self.run_calibration_phase()
        self.run_attribution_phase()
        self.run_post_mortem_phase()

        self.run_evaluation_phase()
        self.run_research_phase()
        self.run_pattern_phase()
        self.run_improvement_phase()

        self.run_dashboard_only()
        elapsed = time.time() - start
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        self.context["elapsed_seconds"] = elapsed
        self._send_daily_email()
        return self.context

    def _send_daily_email(self):
        try:
            scanner_results = self.context.get("scanner_results", {})
            scoring_results = self.context.get("scoring_results", {})
            eval_results = self.context.get("evaluation_results", {})
            calibration_results = self.context.get("calibration_results", {})
            attribution_results = self.context.get("attribution_results", {})
            research_results = self.context.get("research_results", {})
            pattern_results = self.context.get("pattern_results", {})

            factor_perf = {}
            for k, meta in FACTOR_META.items():
                v = attribution_results.get("factor_attributions", {}).get(k)
                if v and v.get("total_trades", 0) > 0:
                    factor_perf[meta["label"]] = {"win_rate": v.get("win_rate", 0),
                                                  "avg_return": v.get("avg_return", 0),
                                                  "total_trades": v.get("total_trades", 0)}

            research_insights = {}
            if research_results:
                research_insights["proposals_list"] = research_results.get("proposals", [])
                research_insights["patterns_found"] = len(pattern_results.get("discoveries", {}).get("multi_factor_rules", {}).get("rules", []))
                if calibration_results:
                    research_insights["calibration_summary"] = {
                        "ece": calibration_results.get("ece", 0),
                        "accuracy": calibration_results.get("overall_accuracy", 0),
                    }

            email_context = {
                "date": datetime.date.today().isoformat(),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stocks_scanned": len(scanner_results.get("all_snapshots", {})),
                "recommendations_count": len(scoring_results.get("recommendations", [])),
                "elapsed_seconds": self.context.get("elapsed_seconds", 0),
                "recommendations": scoring_results.get("recommendations", []),
                "performance": eval_results,
                "research_insights": research_insights,
                "factor_performance": factor_perf,
                "open_positions": eval_results.get("open_positions_data", []),
                "closed_positions": eval_results.get("closed_positions_data", []),
                "portfolio": eval_results.get("portfolio", {}),
                "allocation_data": eval_results.get("allocation_data", []),
                "cash_pct": eval_results.get("cash_pct", 0),
            }
            self.emailer.send_daily_report(email_context)
        except Exception as e:
            logger.error(f"Failed to send daily email: {e}")

    def run_scan_only(self) -> Dict:
        return self.run_scan_phase()

    def run_score_only(self) -> Dict:
        self.run_scan_phase()
        self.run_news_phase()
        self.run_sentiment_phase()
        return self.run_scoring_phase()

    def run_evaluate_only(self) -> Dict:
        return self.run_evaluation_phase()

    def run_attribution_only(self) -> Dict:
        return self.run_attribution_phase()

    def run_calibration_only(self) -> Dict:
        return self.run_calibration_phase()

    def run_post_mortem_only(self) -> Dict:
        return self.run_post_mortem_phase()

    def run_periodic_report(self, period: str = "weekly") -> Dict:
        logger.info(f"=== Periodic Report ({period}) ===")
        results = self.periodic_reporter.execute({"period": period})
        self.periodic_reporter.print_report(results)
        html = self.periodic_reporter.generate_html(results)
        output_path = os.path.join(CONFIG.data_dir, f"periodic_report_{period}.html")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Periodic report written to {output_path}")
        return results

    def run_dashboard_only(self) -> Dict:
        self.dashboard.generate_html_report()
        self.dashboard.print_cli_dashboard()
        return {}
