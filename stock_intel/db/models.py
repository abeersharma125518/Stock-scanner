import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime, Date, Boolean,
    ForeignKey, Index, UniqueConstraint, create_engine, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    exchange = Column(String(50))
    ipo_year = Column(Integer)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow)

    snapshots = relationship("DailySnapshot", back_populates="stock", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="stock", cascade="all, delete-orphan")
    insider_transactions = relationship("InsiderTransaction", back_populates="stock", cascade="all, delete-orphan")
    earnings_events = relationship("EarningsEvent", back_populates="stock", cascade="all, delete-orphan")
    news_articles = relationship("NewsArticle", back_populates="stock", cascade="all, delete-orphan")
    reddit_mentions = relationship("RedditMention", back_populates="stock", cascade="all, delete-orphan")
    signals = relationship("StockSignal", back_populates="stock", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Stock(ticker='{self.ticker}', name='{self.name}')>"


class DailySnapshot(Base):
    __tablename__ = "daily_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    avg_volume_20d = Column(Float)
    volume_ratio = Column(Float)
    premarket_change_pct = Column(Float)
    premarket_volume = Column(Integer)
    after_hours_change_pct = Column(Float)
    gap_pct = Column(Float)
    intraday_high_pct = Column(Float)
    intraday_low_pct = Column(Float)
    vwap = Column(Float)
    atr_14 = Column(Float)
    rsi_14 = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    beta = Column(Float)
    relative_volume = Column(Float)
    dollar_volume = Column(Float)
    float_shares = Column(Float)
    short_ratio = Column(Float)
    short_term_momentum = Column(Float)
    mid_term_momentum = Column(Float)
    is_market_hours = Column(Boolean, default=False)

    stock = relationship("Stock", back_populates="snapshots")

    __table_args__ = (
        Index("idx_snapshot_stock_date", "stock_id", "date", unique=True),
    )


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False)
    rank = Column(Integer)
    total_score = Column(Float)
    volume_score = Column(Float)
    premarket_score = Column(Float)
    sentiment_score = Column(Float)
    news_catalyst_score = Column(Float)
    insider_score = Column(Float)
    earnings_score = Column(Float)
    technical_score = Column(Float)
    momentum_score = Column(Float)
    float_score = Column(Float)
    signals = Column(JSON)
    key_drivers = Column(Text)
    explanation = Column(Text)
    predicted_direction = Column(String(10))
    predicted_gap_pct = Column(Float)
    actual_gap_pct = Column(Float, nullable=True)
    actual_close_pct = Column(Float, nullable=True)
    prediction_accurate = Column(Boolean, nullable=True)
    evaluated = Column(Boolean, default=False)
    failure_reason = Column(Text, nullable=True)
    failure_category = Column(String(50), nullable=True)

    stock = relationship("Stock", back_populates="recommendations")

    __table_args__ = (
        Index("idx_rec_date_rank", "date", "rank"),
        Index("idx_rec_stock_date", "stock_id", "date"),
    )


class InsiderTransaction(Base):
    __tablename__ = "insider_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    filing_date = Column(Date, nullable=False)
    transaction_date = Column(Date)
    insider_name = Column(String(255))
    title = Column(String(255))
    transaction_type = Column(String(50))
    shares_traded = Column(Float)
    price = Column(Float)
    shares_held = Column(Float)
    trade_value = Column(Float)
    is_direct = Column(Boolean, default=True)

    stock = relationship("Stock", back_populates="insider_transactions")

    __table_args__ = (
        Index("idx_insider_stock_date", "stock_id", "filing_date"),
    )


class EarningsEvent(Base):
    __tablename__ = "earnings_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    report_date = Column(Date, nullable=False)
    fiscal_quarter = Column(String(10))
    fiscal_year = Column(Integer)
    eps_estimate = Column(Float)
    eps_actual = Column(Float)
    eps_surprise_pct = Column(Float)
    revenue_estimate = Column(Float)
    revenue_actual = Column(Float)
    revenue_surprise_pct = Column(Float)
    is_beat = Column(Boolean)
    next_report_date = Column(Date)
    is_confirmed = Column(Boolean, default=True)

    stock = relationship("Stock", back_populates="earnings_events")

    __table_args__ = (
        Index("idx_earnings_stock_date", "stock_id", "report_date"),
    )


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    published_at = Column(DateTime, nullable=False)
    title = Column(Text)
    summary = Column(Text)
    source = Column(String(100))
    url = Column(String(500))
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    relevance_score = Column(Float, default=1.0)
    topics = Column(JSON)
    is_earnings_related = Column(Boolean, default=False)
    is_insider_related = Column(Boolean, default=False)
    is_catalyst = Column(Boolean, default=False)

    stock = relationship("Stock", back_populates="news_articles")

    __table_args__ = (
        Index("idx_news_stock_date", "stock_id", "published_at"),
        Index("idx_news_date", "published_at"),
    )


class RedditMention(Base):
    __tablename__ = "reddit_mentions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    created_at = Column(DateTime, nullable=False)
    subreddit = Column(String(100))
    post_title = Column(Text)
    post_url = Column(String(500))
    post_score = Column(Integer)
    num_comments = Column(Integer)
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    mention_count_in_post = Column(Integer, default=1)
    is_self_post = Column(Boolean, default=True)
    upvote_ratio = Column(Float)

    stock = relationship("Stock", back_populates="reddit_mentions")

    __table_args__ = (
        Index("idx_reddit_stock_date", "stock_id", "created_at"),
        Index("idx_reddit_date", "created_at"),
    )


class StockSignal(Base):
    __tablename__ = "stock_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False)
    signal_type = Column(String(50), nullable=False)
    signal_strength = Column(Float, default=0.0)
    signal_data = Column(JSON)
    is_active = Column(Boolean, default=True)

    stock = relationship("Stock", back_populates="signals")

    __table_args__ = (
        Index("idx_signal_type_date", "signal_type", "date"),
        Index("idx_signal_stock_date", "stock_id", "date"),
    )


class PerformanceSummary(Base):
    __tablename__ = "performance_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True)
    total_recommendations = Column(Integer)
    correct_predictions = Column(Integer)
    incorrect_predictions = Column(Integer)
    win_rate = Column(Float)
    avg_return_pct = Column(Float)
    median_return_pct = Column(Float)
    total_gain_pct = Column(Float)
    max_gain_pct = Column(Float)
    max_loss_pct = Column(Float)
    std_return_pct = Column(Float)
    sharpe_ratio = Column(Float)
    cumulative_win_rate = Column(Float)
    risk_reward_ratio = Column(Float)
    avg_volume_score = Column(Float)
    avg_premarket_score = Column(Float)
    avg_sentiment_score = Column(Float)
    avg_news_score = Column(Float)
    avg_insider_score = Column(Float)
    avg_earnings_score = Column(Float)
    avg_technical_score = Column(Float)
    avg_total_score = Column(Float)
    avg_float_score = Column(Float)

    signal_performances = relationship("SignalPerformance", back_populates="summary", cascade="all, delete-orphan")


class SignalPerformance(Base):
    __tablename__ = "signal_performances"

    id = Column(Integer, primary_key=True, autoincrement=True)
    summary_id = Column(Integer, ForeignKey("performance_summaries.id"), nullable=False)
    signal_type = Column(String(50), nullable=False)
    total_occurrences = Column(Integer)
    correct_predictions = Column(Integer)
    win_rate = Column(Float)
    avg_return_pct = Column(Float)
    total_return_pct = Column(Float)

    summary = relationship("PerformanceSummary", back_populates="signal_performances")

    __table_args__ = (
        Index("idx_sigperf_summary_type", "summary_id", "signal_type", unique=True),
    )


class UserAlert(Base):
    __tablename__ = "user_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    alert_type = Column(String(50))
    ticker = Column(String(10))
    message = Column(Text)
    priority = Column(String(20), default="normal")
    is_read = Column(Boolean, default=False)
    sent_email = Column(Boolean, default=False)
    email_sent_at = Column(DateTime, nullable=True)


class ResearchReport(Base):
    __tablename__ = "research_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    report_type = Column(String(20), nullable=False)
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    total_recommendations = Column(Integer)
    evaluated_count = Column(Integer)
    overall_win_rate = Column(Float)
    overall_avg_return = Column(Float)
    summary = Column(Text)
    findings_json = Column(JSON)
    proposals = relationship("StrategyProposal", back_populates="report", cascade="all, delete-orphan")


class StrategyProposal(Base):
    __tablename__ = "strategy_proposals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey("research_reports.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    proposed_changes = Column(JSON)
    current_weights = Column(JSON)
    statistical_justification = Column(Text)
    sample_sizes = Column(JSON)
    confidence_metrics = Column(JSON)
    expected_impact = Column(JSON)
    evidence_details = Column(JSON)
    status = Column(String(20), default="proposed")

    report = relationship("ResearchReport", back_populates="proposals")
    arguments = relationship("ProposalArgument", back_populates="proposal", cascade="all, delete-orphan")


class ProposalArgument(Base):
    __tablename__ = "proposal_arguments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    proposal_id = Column(Integer, ForeignKey("strategy_proposals.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    stance = Column(String(10), nullable=False)
    argument = Column(Text, nullable=False)
    evidence = Column(Text)
    confidence = Column(Float)
    agent_name = Column(String(100), default="human")

    proposal = relationship("StrategyProposal", back_populates="arguments")


def create_all_tables(engine):
    Base.metadata.create_all(engine)
