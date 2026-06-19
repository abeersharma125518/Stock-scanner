import datetime
import logging
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Generator
from sqlalchemy import create_engine, func, and_, desc, asc
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from stock_intel.config.settings import CONFIG
from stock_intel.db.models import (
    Base, Stock, DailySnapshot, Recommendation, InsiderTransaction,
    EarningsEvent, NewsArticle, RedditMention, StockSignal,
    PerformanceSummary, SignalPerformance, UserAlert,
    ResearchReport, StrategyProposal, ProposalArgument, create_all_tables,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or CONFIG.db.path
        self._engine = None
        self._session_factory = None
        self.connect()

    def connect(self):
        self._engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=CONFIG.db.echo,
            poolclass=QueuePool,
            pool_size=CONFIG.db.pool_size,
            max_overflow=CONFIG.db.max_overflow,
        )
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        create_all_tables(self._engine)
        self._migrate_schema()
        logger.info(f"Database connected: {self.db_path}")

    def _migrate_schema(self):
        with self.session() as session:
            from sqlalchemy import inspect
            inspector = inspect(self._engine)
            rec_cols = [c["name"] for c in inspector.get_columns("recommendations")]
            if "failure_reason" not in rec_cols:
                session.execute("ALTER TABLE recommendations ADD COLUMN failure_reason TEXT")
                session.execute("ALTER TABLE recommendations ADD COLUMN failure_category VARCHAR(50)")
                logger.info("Schema migration: added failure_reason, failure_category to recommendations")
            snap_cols = [c["name"] for c in inspector.get_columns("daily_snapshots")]
            for col in ("short_term_momentum", "mid_term_momentum"):
                if col not in snap_cols:
                    session.execute(f"ALTER TABLE daily_snapshots ADD COLUMN {col} FLOAT")
                    logger.info(f"Schema migration: added {col} to daily_snapshots")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_or_create_stock(self, ticker: str, session: Session = None) -> Stock:
        if session is None:
            with self.session() as s:
                return self._get_or_create_stock(ticker, s)
        return self._get_or_create_stock(ticker, session)

    def get_or_create_stock_id(self, ticker: str) -> int:
        with self.session() as session:
            stock = self._get_or_create_stock(ticker, session)
            session.flush()
            return stock.id

    def _get_or_create_stock(self, ticker: str, session: Session) -> Stock:
        stock = session.query(Stock).filter_by(ticker=ticker.upper()).first()
        if stock is None:
            stock = Stock(ticker=ticker.upper(), last_updated=datetime.datetime.utcnow())
            session.add(stock)
            session.flush()
        return stock

    def bulk_upsert_stocks(self, tickers: List[str]) -> Dict[str, Stock]:
        result = {}
        with self.session() as session:
            for ticker in tickers:
                stock = self._get_or_create_stock(ticker, session)
                result[ticker.upper()] = stock
        return result

    def _get_model_columns(self, model_class) -> set:
        return {c.name for c in model_class.__table__.columns}

    def save_snapshot(self, stock_id: int, date: Date, data: dict) -> DailySnapshot:
        with self.session() as session:
            existing = session.query(DailySnapshot).filter_by(
                stock_id=stock_id, date=date
            ).first()
            valid_cols = self._get_model_columns(DailySnapshot)
            data_copy = {k: v for k, v in data.items() if k in valid_cols and k != "date" and k != "id"}
            if existing:
                for k, v in data_copy.items():
                    if v is not None:
                        setattr(existing, k, v)
                snapshot = existing
            else:
                snapshot = DailySnapshot(stock_id=stock_id, date=date, **data_copy)
                session.add(snapshot)
            session.flush()
            return snapshot

    def save_recommendation(self, stock_id: int, date: Date, rank: int,
                            data: dict) -> Recommendation:
        with self.session() as session:
            data_copy = {k: v for k, v in data.items() if k != "date"}
            rec = Recommendation(stock_id=stock_id, date=date, rank=rank, **data_copy)
            session.add(rec)
            session.flush()
            return rec

    def get_recommendations_by_date(self, date: Date) -> List[Recommendation]:
        with self.session() as session:
            return session.query(Recommendation).filter(
                Recommendation.date == date
            ).order_by(Recommendation.rank).all()

    def get_latest_recommendations(self, limit: int = 10) -> List[dict]:
        with self.session() as session:
            subq = session.query(
                Recommendation.date.label("max_date")
            ).order_by(desc(Recommendation.date)).limit(1).subquery()
            recs = session.query(Recommendation).filter(
                Recommendation.date == subq.c.max_date
            ).order_by(Recommendation.rank).limit(limit).all()
            result = []
            for r in recs:
                stock = session.query(Stock).filter_by(id=r.stock_id).first()
                result.append({
                    "rank": r.rank,
                    "ticker": stock.ticker if stock else "?",
                    "name": stock.name if stock else "",
                    "score": r.total_score,
                    "explanation": r.explanation,
                    "signals": r.signals,
                    "predicted_direction": r.predicted_direction,
                    "predicted_gap_pct": r.predicted_gap_pct,
                })
            return result

    def get_recommendations_by_date_range(self, start_date, end_date) -> List[dict]:
        with self.session() as session:
            recs = session.query(Recommendation).filter(
                Recommendation.date.between(start_date, end_date),
                Recommendation.evaluated == True,
            ).order_by(Recommendation.date, Recommendation.rank).all()
            result = []
            for r in recs:
                stock = session.query(Stock).filter_by(id=r.stock_id).first()
                result.append({
                    "date": r.date.isoformat() if hasattr(r.date, "isoformat") else str(r.date),
                    "ticker": stock.ticker if stock else "?",
                    "sector": stock.sector if stock else None,
                    "rank": r.rank,
                    "total_score": r.total_score,
                    "predicted_direction": r.predicted_direction,
                    "predicted_gap_pct": r.predicted_gap_pct,
                    "actual_close_pct": r.actual_close_pct,
                    "prediction_accurate": r.prediction_accurate,
                    "volume_score": r.volume_score,
                    "sentiment_score": r.sentiment_score,
                    "news_catalyst_score": r.news_catalyst_score,
                    "momentum_score": r.momentum_score,
                    "earnings_score": r.earnings_score,
                    "float_score": r.float_score,
                    "insider_score": r.insider_score,
                    "technical_score": r.technical_score,
                    "premarket_score": r.premarket_score,
                    "failure_reason": r.failure_reason,
                    "failure_category": r.failure_category,
                })
            return result

    def save_insider_transactions(self, stock_id: int, transactions: List[dict]):
        with self.session() as session:
            for t in transactions:
                existing = session.query(InsiderTransaction).filter_by(
                    stock_id=stock_id,
                    filing_date=t.get("filing_date"),
                    insider_name=t.get("insider_name"),
                    transaction_type=t.get("transaction_type"),
                ).first()
                if not existing:
                    tx = InsiderTransaction(stock_id=stock_id, **t)
                    session.add(tx)

    def save_earnings_event(self, stock_id: int, data: dict) -> EarningsEvent:
        with self.session() as session:
            event = EarningsEvent(stock_id=stock_id, **data)
            session.add(event)
            session.flush()
            return event

    def save_news_article(self, stock_id: int, data: dict) -> NewsArticle:
        with self.session() as session:
            article = NewsArticle(stock_id=stock_id, **data)
            session.add(article)
            session.flush()
            return article

    def save_reddit_mention(self, stock_id: int, data: dict) -> RedditMention:
        with self.session() as session:
            mention = RedditMention(stock_id=stock_id, **data)
            session.add(mention)
            session.flush()
            return mention

    def save_signal(self, stock_id: int, date: Date, signal_type: str,
                    strength: float, data: dict = None) -> StockSignal:
        with self.session() as session:
            signal = StockSignal(
                stock_id=stock_id, date=date, signal_type=signal_type,
                signal_strength=strength, signal_data=data or {},
            )
            session.add(signal)
            session.flush()
            return signal

    def save_performance_summary(self, date: Date, data: dict) -> PerformanceSummary:
        with self.session() as session:
            existing = session.query(PerformanceSummary).filter_by(date=date).first()
            if existing:
                for k, v in data.items():
                    if hasattr(existing, k):
                        setattr(existing, k, v)
                summary = existing
            else:
                summary = PerformanceSummary(date=date, **data)
                session.add(summary)
            session.flush()
            return summary

    def save_signal_performance(self, summary_id: int, signal_type: str,
                                 data: dict) -> SignalPerformance:
        with self.session() as session:
            existing = session.query(SignalPerformance).filter_by(
                summary_id=summary_id, signal_type=signal_type
            ).first()
            if existing:
                for k, v in data.items():
                    if hasattr(existing, k):
                        setattr(existing, k, v)
                sp = existing
            else:
                sp = SignalPerformance(summary_id=summary_id, signal_type=signal_type, **data)
                session.add(sp)
            session.flush()
            return sp

    def create_alert(self, alert_type: str, ticker: str, message: str,
                     priority: str = "normal") -> UserAlert:
        with self.session() as session:
            alert = UserAlert(
                alert_type=alert_type, ticker=ticker, message=message,
                priority=priority,
            )
            session.add(alert)
            session.flush()
            return alert

    def get_unread_alerts(self) -> List[UserAlert]:
        with self.session() as session:
            return session.query(UserAlert).filter_by(is_read=False).all()

    def get_stock_snapshots_by_date_range(self, start_date, end_date) -> List[dict]:
        with self.session() as session:
            snapshots = session.query(DailySnapshot).filter(
                DailySnapshot.date.between(start_date, end_date),
            ).order_by(DailySnapshot.date).all()
            return [
                {
                    "stock_id": s.stock_id, "date": s.date.isoformat() if hasattr(s.date, "isoformat") else str(s.date),
                    "volume_ratio": s.volume_ratio, "rsi_14": s.rsi_14,
                    "premarket_change_pct": s.premarket_change_pct,
                    "short_term_momentum": s.short_term_momentum,
                    "atr_14": s.atr_14, "volume": s.volume,
                    "close_price": s.close_price,
                }
                for s in snapshots
            ]

    def get_stock_snapshots(self, ticker: str, days: int = 20) -> List[DailySnapshot]:
        with self.session() as session:
            stock = session.query(Stock).filter_by(ticker=ticker.upper()).first()
            if not stock:
                return []
            cutoff = datetime.date.today() - datetime.timedelta(days=days)
            return session.query(DailySnapshot).filter(
                DailySnapshot.stock_id == stock.id,
                DailySnapshot.date >= cutoff,
            ).order_by(DailySnapshot.date).all()

    def get_upcoming_earnings(self, days_ahead: int = 7) -> List[dict]:
        with self.session() as session:
            today = datetime.date.today()
            end = today + datetime.timedelta(days=days_ahead)
            events = session.query(EarningsEvent).filter(
                EarningsEvent.report_date.between(today, end),
                EarningsEvent.is_confirmed == True,
            ).all()
            result = []
            for e in events:
                stock = session.query(Stock).filter_by(id=e.stock_id).first()
                result.append({
                    "ticker": stock.ticker if stock else "?",
                    "date": e.report_date,
                    "eps_estimate": e.eps_estimate,
                    "revenue_estimate": e.revenue_estimate,
                })
            return result

    def get_performance_history(self, days: int = 90) -> List[dict]:
        with self.session() as session:
            cutoff = datetime.date.today() - datetime.timedelta(days=days)
            summaries = session.query(PerformanceSummary).filter(
                PerformanceSummary.date >= cutoff
            ).order_by(PerformanceSummary.date).all()
            return [
                {
                    "date": s.date,
                    "win_rate": s.win_rate,
                    "avg_return": s.avg_return_pct,
                    "total_recs": s.total_recommendations,
                    "correct": s.correct_predictions,
                    "incorrect": s.incorrect_predictions,
                    "cumulative_win_rate": s.cumulative_win_rate,
                    "sharpe_ratio": s.sharpe_ratio,
                    "avg_total_score": s.avg_total_score,
                }
                for s in summaries
            ]

    def get_signal_performance_history(self, days: int = 90) -> Dict[str, List[dict]]:
        with self.session() as session:
            cutoff = datetime.date.today() - datetime.timedelta(days=days)
            summaries = session.query(PerformanceSummary).filter(
                PerformanceSummary.date >= cutoff
            ).all()
            signal_data = {}
            for s in summaries:
                for sp in s.signal_performances:
                    if sp.signal_type not in signal_data:
                        signal_data[sp.signal_type] = []
                    signal_data[sp.signal_type].append({
                        "date": s.date,
                        "win_rate": sp.win_rate,
                        "avg_return": sp.avg_return_pct,
                        "occurrences": sp.total_occurrences,
                    })
            return signal_data

    def get_all_stored_tickers(self) -> List[str]:
        with self.session() as session:
            return [s.ticker for s in session.query(Stock.ticker).all()]

    def get_recent_news_for_ticker(self, ticker: str, days: int = 3) -> List[NewsArticle]:
        with self.session() as session:
            stock = session.query(Stock).filter_by(ticker=ticker.upper()).first()
            if not stock:
                return []
            cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
            return session.query(NewsArticle).filter(
                NewsArticle.stock_id == stock.id,
                NewsArticle.published_at >= cutoff,
            ).order_by(desc(NewsArticle.published_at)).all()

    def get_recent_reddit_mentions(self, ticker: str, days: int = 2) -> List[RedditMention]:
        with self.session() as session:
            stock = session.query(Stock).filter_by(ticker=ticker.upper()).first()
            if not stock:
                return []
            cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
            return session.query(RedditMention).filter(
                RedditMention.stock_id == stock.id,
                RedditMention.created_at >= cutoff,
            ).order_by(desc(RedditMention.created_at)).all()

    def get_un_evaluated_recommendations(self) -> List[Recommendation]:
        with self.session() as session:
            from sqlalchemy.orm import joinedload
            return session.query(Recommendation).options(
                joinedload(Recommendation.stock)
            ).filter_by(evaluated=False).all()

    def get_attribution_detail(self, days: int = 90) -> List[dict]:
        with self.session() as session:
            cutoff = datetime.date.today() - datetime.timedelta(days=days)
            recs = session.query(Recommendation).filter(
                Recommendation.evaluated == True,
                Recommendation.date >= cutoff,
                Recommendation.actual_close_pct.isnot(None),
            ).order_by(desc(Recommendation.date)).all()
            results = []
            for r in recs:
                stock = session.query(Stock).filter_by(id=r.stock_id).first()
                scores = {
                    "volume_score": r.volume_score or 0,
                    "news_score": r.news_catalyst_score or 0,
                    "sentiment_score": r.sentiment_score or 0,
                    "momentum_score": r.momentum_score or 0,
                    "premarket_score": r.premarket_score or 0,
                    "earnings_score": r.earnings_score or 0,
                    "float_score": r.float_score or 0,
                    "insider_score": r.insider_score or 0,
                    "technical_score": r.technical_score or 0,
                }
                max_score = max(scores.values()) if scores else 0
                dominant = [k for k, v in scores.items() if v == max_score] if max_score > 0 else []
                results.append({
                    "date": r.date.isoformat() if hasattr(r.date, "isoformat") else str(r.date),
                    "ticker": stock.ticker if stock else "?",
                    "rank": r.rank,
                    "total_score": r.total_score,
                    "dominant_factors": dominant,
                    **scores,
                    "actual_return": r.actual_close_pct,
                    "prediction_accurate": r.prediction_accurate,
                    "predicted_direction": r.predicted_direction,
                })
            return results

    def get_evaluated_recommendations(self, days: int = 90) -> List[dict]:
        with self.session() as session:
            cutoff = datetime.date.today() - datetime.timedelta(days=days)
            recs = session.query(Recommendation).filter(
                Recommendation.evaluated == True,
                Recommendation.date >= cutoff,
                Recommendation.actual_close_pct.isnot(None),
            ).order_by(desc(Recommendation.date)).all()
            results = []
            for r in recs:
                stock = session.query(Stock).filter_by(id=r.stock_id).first()
                results.append({
                    "date": r.date.isoformat() if hasattr(r.date, "isoformat") else str(r.date),
                    "ticker": stock.ticker if stock else "?",
                    "total_score": r.total_score,
                    "volume_score": r.volume_score or 0,
                    "premarket_score": r.premarket_score or 0,
                    "sentiment_score": r.sentiment_score or 0,
                    "news_score": r.news_catalyst_score or 0,
                    "insider_score": r.insider_score or 0,
                    "earnings_score": r.earnings_score or 0,
                    "technical_score": r.technical_score or 0,
                    "momentum_score": r.momentum_score or 0,
                    "float_score": r.float_score or 0,
                    "actual_return": r.actual_close_pct,
                    "prediction_accurate": r.prediction_accurate,
                    "predicted_direction": r.predicted_direction,
                    "rank": r.rank,
                })
            return results

    def get_all_evaluated_recommendations(self) -> List[dict]:
        with self.session() as session:
            recs = session.query(Recommendation).filter(
                Recommendation.evaluated == True,
                Recommendation.actual_close_pct.isnot(None),
            ).order_by(Recommendation.date).all()
            results = []
            for r in recs:
                stock = session.query(Stock).filter_by(id=r.stock_id).first()
                results.append({
                    "id": r.id,
                    "date": r.date.isoformat() if hasattr(r.date, "isoformat") else str(r.date),
                    "ticker": stock.ticker if stock else "?",
                    "sector": stock.sector if stock else None,
                    "rank": r.rank,
                    "total_score": r.total_score,
                    "predicted_direction": r.predicted_direction,
                    "predicted_gap_pct": r.predicted_gap_pct,
                    "actual_return": r.actual_close_pct,
                    "prediction_accurate": r.prediction_accurate,
                    "volume_score": r.volume_score or 0,
                    "premarket_score": r.premarket_score or 0,
                    "sentiment_score": r.sentiment_score or 0,
                    "news_score": r.news_catalyst_score or 0,
                    "insider_score": r.insider_score or 0,
                    "earnings_score": r.earnings_score or 0,
                    "technical_score": r.technical_score or 0,
                    "momentum_score": r.momentum_score or 0,
                    "float_score": r.float_score or 0,
                    "failure_reason": r.failure_reason,
                    "failure_category": r.failure_category,
                })
            return results

    def save_research_report(self, report_type: str, period_start, period_end, data: dict) -> ResearchReport:
        with self.session() as session:
            report = ResearchReport(
                report_type=report_type, period_start=period_start, period_end=period_end,
                total_recommendations=data.get("total_recommendations"),
                evaluated_count=data.get("evaluated_count"),
                overall_win_rate=data.get("overall_win_rate"),
                overall_avg_return=data.get("overall_avg_return"),
                summary=data.get("summary", ""),
                findings_json=data.get("findings", {}),
            )
            session.add(report)
            session.flush()
            return report

    def save_strategy_proposal(self, report_id: int, data: dict) -> StrategyProposal:
        with self.session() as session:
            proposal = StrategyProposal(
                report_id=report_id, title=data["title"],
                description=data.get("description", ""),
                proposed_changes=data.get("proposed_changes", {}),
                current_weights=data.get("current_weights", {}),
                statistical_justification=data.get("statistical_justification", ""),
                sample_sizes=data.get("sample_sizes", {}),
                confidence_metrics=data.get("confidence_metrics", {}),
                expected_impact=data.get("expected_impact", {}),
                evidence_details=data.get("evidence_details", {}),
                status="proposed",
            )
            session.add(proposal)
            session.flush()
            return proposal

    def get_research_reports(self, limit: int = 10) -> List[ResearchReport]:
        with self.session() as session:
            from sqlalchemy.orm import joinedload
            return session.query(ResearchReport).options(
                joinedload(ResearchReport.proposals).joinedload(StrategyProposal.arguments)
            ).order_by(desc(ResearchReport.created_at)).limit(limit).all()

    def get_proposal_by_id(self, proposal_id: int) -> Optional[StrategyProposal]:
        with self.session() as session:
            from sqlalchemy.orm import joinedload
            return session.query(StrategyProposal).options(
                joinedload(StrategyProposal.arguments)
            ).filter_by(id=proposal_id).first()

    def get_proposals(self, status: Optional[str] = None) -> List[StrategyProposal]:
        with self.session() as session:
            from sqlalchemy.orm import joinedload
            q = session.query(StrategyProposal).options(
                joinedload(StrategyProposal.arguments)
            ).order_by(desc(StrategyProposal.created_at))
            if status:
                q = q.filter(StrategyProposal.status == status)
            return q.all()

    def update_proposal_status(self, proposal_id: int, status: str) -> bool:
        with self.session() as session:
            prop = session.query(StrategyProposal).filter_by(id=proposal_id).first()
            if not prop:
                return False
            prop.status = status
            return True

    def add_proposal_argument(self, proposal_id: int, stance: str, argument: str,
                               evidence: str = "", confidence: float = None,
                               agent_name: str = "human") -> ProposalArgument:
        with self.session() as session:
            arg = ProposalArgument(
                proposal_id=proposal_id, stance=stance, argument=argument,
                evidence=evidence, confidence=confidence, agent_name=agent_name,
            )
            session.add(arg)
            session.flush()
            return arg
