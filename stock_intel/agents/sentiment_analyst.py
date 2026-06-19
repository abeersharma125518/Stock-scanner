import datetime
import logging
from typing import Dict, List, Optional, Any
from stock_intel.agents.base_agent import BaseAgent
from stock_intel.collectors.reddit_collector import RedditCollector
from stock_intel.db.database import DatabaseManager
from stock_intel.config.settings import CONFIG

logger = logging.getLogger(__name__)


class SentimentAnalyst(BaseAgent):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        super().__init__(db, config)
        self.reddit = RedditCollector()
        self.reddit_weight = CONFIG.sentiment.reddit_weight
        self.news_weight = CONFIG.sentiment.news_sources_weight

    def validate(self) -> bool:
        return True

    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.log_start()
        results = {
            "reddit_mentions": {}, "sentiment_scores": {},
            "top_bullish": [], "top_bearish": [], "most_discussed": [],
            "timestamp": datetime.datetime.now().isoformat(),
        }

        reddit_data = self.reddit.fetch_consolidated()
        results["reddit_mentions"] = reddit_data

        for ticker, mentions in reddit_data.items():
            for m in mentions:
                if ticker and mentions:
                    stock_id = self.db.get_or_create_stock_id(ticker)
                    if stock_id:
                        self.db.save_reddit_mention(stock_id, m)

        for ticker, mentions in reddit_data.items():
            sentiment = self._compute_aggregate_sentiment(ticker, mentions)
            recent_news = self.db.get_recent_news_for_ticker(ticker, days=2)
            news_sentiment = 0.0
            if recent_news:
                news_sentiment = sum(n.sentiment_score or 0 for n in recent_news) / len(recent_news)

            combined = (sentiment["avg_sentiment"] * self.reddit_weight + news_sentiment * self.news_weight)
            combined_weight = self.reddit_weight + self.news_weight
            if combined_weight > 0:
                combined /= combined_weight

            results["sentiment_scores"][ticker] = {
                "reddit_sentiment": sentiment["avg_sentiment"],
                "news_sentiment": news_sentiment,
                "combined_sentiment": combined,
                "mention_count": sentiment["mention_count"],
                "total_comments": sentiment["total_comments"],
                "total_score": sentiment["total_score"],
                "subreddits": sentiment["subreddits"],
                "label": "bullish" if combined > 0.15 else "bearish" if combined < -0.15 else "neutral",
            }

            stock_id = self.db.get_or_create_stock_id(ticker)
            if stock_id and sentiment["mention_count"] > 0:
                self.db.save_signal(stock_id, datetime.date.today(), "sentiment", float(combined), {
                    "combined_sentiment": combined, "reddit_sentiment": sentiment["avg_sentiment"],
                    "news_sentiment": news_sentiment, "mention_count": sentiment["mention_count"],
                })

        sorted_scores = sorted(results["sentiment_scores"].items(), key=lambda x: x[1]["combined_sentiment"], reverse=True)
        results["top_bullish"] = [{"ticker": t, **s} for t, s in sorted_scores[:20] if s["combined_sentiment"] > 0.15]
        results["top_bearish"] = [{"ticker": t, **s} for t, s in reversed(sorted_scores[-20:]) if s["combined_sentiment"] < -0.15]

        most_discussed = sorted(results["sentiment_scores"].items(), key=lambda x: x[1]["mention_count"], reverse=True)
        results["most_discussed"] = [{"ticker": t, **s} for t, s in most_discussed[:20]]

        self.results = results
        self.log_end()
        return results

    def _compute_aggregate_sentiment(self, ticker: str, mentions: List[Dict]) -> Dict:
        if not mentions:
            return {"avg_sentiment": 0.0, "mention_count": 0, "total_comments": 0, "total_score": 0, "subreddits": []}
        weighted_sum = 0
        total_weight = 0
        for m in mentions:
            weight = 1.0
            if m.get("post_score") and m["post_score"] > 0:
                weight *= min(m["post_score"] / 100, 5.0)
            if m.get("num_comments") and m["num_comments"] > 0:
                weight *= min(m["num_comments"] / 50, 3.0)
            weighted_sum += (m.get("sentiment_score", 0) or 0) * weight
            total_weight += weight
        avg_sent = weighted_sum / total_weight if total_weight > 0 else 0
        return {
            "avg_sentiment": avg_sent, "mention_count": len(mentions),
            "total_comments": sum(m.get("num_comments", 0) or 0 for m in mentions),
            "total_score": sum(m.get("post_score", 0) or 0 for m in mentions),
            "subreddits": list(set(m.get("subreddit", "") for m in mentions)),
        }
