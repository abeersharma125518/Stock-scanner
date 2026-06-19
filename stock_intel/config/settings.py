import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class DatabaseConfig:
    path: str = str(Path(__file__).parent.parent.parent / "stock_intel" / "data" / "stock_intel.db")
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class ScannerConfig:
    min_price: float = 2.0
    max_price: float = 500.0
    min_volume: int = 100000
    volume_surge_threshold: float = 1.5
    premarket_change_threshold: float = 2.0
    max_stocks_to_scan: int = 3000
    lookback_days: int = 20
    unusual_volume_lookback: int = 5


@dataclass
class SentimentConfig:
    reddit_subreddits: List[str] = field(default_factory=lambda: [
        "wallstreetbets", "stocks", "investing", "StockMarket",
        "pennystocks", "options", "trading", "Daytrading"
    ])
    reddit_posts_limit: int = 500
    news_sources_weight: float = 0.5
    reddit_weight: float = 0.5
    positive_threshold: float = 0.15
    negative_threshold: float = -0.15


@dataclass
class ScoringWeights:
    volume_score: float = 0.25
    news_catalyst_score: float = 0.20
    momentum_score: float = 0.15
    float_score: float = 0.10
    earnings_momentum_score: float = 0.10
    sentiment_score: float = 0.08
    technical_score: float = 0.07
    premarket_score: float = 0.03
    insider_score: float = 0.02

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EmailConfig:
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    recipient_email: str = ""
    enabled: bool = False


@dataclass
class CollectorConfig:
    yahoo_rate_limit: float = 0.5
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "StockIntel/1.0"
    finviz_timeout: int = 30
    sec_rate_limit: float = 0.1


@dataclass
class AppConfig:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    email: EmailConfig = field(default_factory=EmailConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    top_n_recommendations: int = 10
    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0
    premarket_start_hour: int = 4
    premarket_start_minute: int = 0
    data_dir: str = str(Path(__file__).parent.parent.parent / "stock_intel" / "data")

    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.data_dir, "config.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> dict:
        return {
            "db": asdict(self.db),
            "scanner": asdict(self.scanner),
            "sentiment": asdict(self.sentiment),
            "weights": asdict(self.weights),
            "email": asdict(self.email),
            "collector": asdict(self.collector),
            "top_n_recommendations": self.top_n_recommendations,
            "market_open_hour": self.market_open_hour,
            "market_open_minute": self.market_open_minute,
            "market_close_hour": self.market_close_hour,
            "market_close_minute": self.market_close_minute,
            "premarket_start_hour": self.premarket_start_hour,
            "premarket_start_minute": self.premarket_start_minute,
            "data_dir": self.data_dir,
        }

    @classmethod
    def load(cls, path: Optional[str] = None):
        cfg = cls()
        if path is None:
            path = os.path.join(cfg.data_dir, "config.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if "db" in data:
                cfg.db = DatabaseConfig(**data["db"])
            if "scanner" in data:
                cfg.scanner = ScannerConfig(**data["scanner"])
            if "sentiment" in data:
                cfg.sentiment = SentimentConfig(**data["sentiment"])
            if "weights" in data:
                cfg.weights = ScoringWeights.from_dict(data["weights"])
            if "email" in data:
                cfg.email = EmailConfig(**data["email"])
            if "collector" in data:
                cfg.collector = CollectorConfig(**data["collector"])
            if "top_n_recommendations" in data:
                cfg.top_n_recommendations = data["top_n_recommendations"]
        return cfg

    @classmethod
    def default_weights(cls) -> Dict[str, float]:
        return asdict(ScoringWeights())


CONFIG = AppConfig.load()
