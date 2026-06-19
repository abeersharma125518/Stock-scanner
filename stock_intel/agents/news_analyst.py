import datetime
import logging
import time
from typing import Dict, List, Optional, Any
from stock_intel.agents.base_agent import BaseAgent
from stock_intel.db.database import DatabaseManager

logger = logging.getLogger(__name__)


class NewsAnalyst(BaseAgent):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        super().__init__(db, config)
        self.news_cache: Dict[str, List[Dict]] = {}

    def validate(self) -> bool:
        return True

    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.log_start()
        all_tickers = self.db.get_all_stored_tickers()
        news_results = {}
        catalyst_stocks = []

        for i, ticker in enumerate(all_tickers):
            try:
                articles = self._fetch_news_for_ticker(ticker)
                if articles:
                    news_results[ticker] = articles
                    for article in articles:
                        stock_id = self.db.get_or_create_stock_id(ticker)
                        if stock_id:
                            self.db.save_news_article(stock_id, article)

                    catalyst = self._detect_catalyst(articles)
                    if catalyst:
                        catalyst_stocks.append({
                            "ticker": ticker, "catalyst_type": catalyst["type"],
                            "catalyst_description": catalyst["description"],
                            "articles_count": len(articles),
                            "avg_sentiment": catalyst["avg_sentiment"],
                            "confidence": catalyst["confidence"],
                        })
                        stock_id = self.db.get_or_create_stock_id(ticker)
                        if stock_id:
                            self.db.save_signal(stock_id, datetime.date.today(),
                                                f"catalyst_{catalyst['type']}",
                                                catalyst["confidence"], catalyst)

                if i > 0 and i % 100 == 0:
                    logger.info(f"News: processed {i}/{len(all_tickers)} tickers")
                    time.sleep(0.5)
            except Exception as e:
                logger.warning(f"News fetch failed for {ticker}: {e}")
                continue

        catalyst_stocks.sort(key=lambda x: x["confidence"], reverse=True)
        self.results = {
            "total_articles": sum(len(v) for v in news_results.values()),
            "tickers_with_news": len(news_results),
            "catalyst_stocks": catalyst_stocks,
            "all_news": news_results,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.log_end()
        return self.results

    def _fetch_news_for_ticker(self, ticker: str) -> List[Dict]:
        articles = []
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            news = stock.news
            if news:
                for item in news[:10]:
                    title = item.get("title", "")
                    sentiment = self._calc_sentiment(title)
                    articles.append({
                        "published_at": datetime.datetime.fromtimestamp(item.get("providerPublishTime", 0))
                                        if item.get("providerPublishTime") else datetime.datetime.now(),
                        "title": title, "summary": item.get("summary", ""),
                        "source": item.get("publisher", "Yahoo"), "url": item.get("link", ""),
                        "sentiment_score": sentiment,
                        "sentiment_label": "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral",
                        "is_catalyst": self._is_catalyst_title(title),
                        "topics": self._extract_topics(title),
                    })
        except Exception:
            pass

        seen = set()
        unique = []
        for a in articles:
            if a["title"] not in seen:
                seen.add(a["title"])
                unique.append(a)
        return unique

    def _detect_catalyst(self, articles: List[Dict]) -> Optional[Dict]:
        catalyst_keywords = {
            "earnings": ["earnings", "quarterly result", "profit", "revenue", "EPS", "fiscal"],
            "merger": ["merger", "acquisition", "buyout", "takeover", "merge"],
            "partnership": ["partnership", "collaboration", "alliance", "joint venture"],
            "product_launch": ["launch", "unveil", "introduce", "new product", "FDA"],
            "contract": ["contract", "awarded", "government grant", "approval"],
            "stock_split": ["stock split", "forward split", "reverse split"],
            "dividend": ["dividend", "buyback", "share repurchase"],
            "legal": ["lawsuit", "settlement", "investigation", "SEC", "DOJ"],
            "analyst_upgrade": ["upgrade", "outperform", "buy rating", "target raise"],
            "analyst_downgrade": ["downgrade", "sell rating", "target cut"],
        }
        detected_types = []
        total_sentiment = 0
        catalyst_count = 0
        for article in articles:
            title = article.get("title", "").lower()
            for ctype, keywords in catalyst_keywords.items():
                if any(kw.lower() in title for kw in keywords):
                    detected_types.append(ctype)
                    total_sentiment += article.get("sentiment_score", 0)
                    catalyst_count += 1
                    break
        if not detected_types:
            return None
        from collections import Counter
        type_counts = Counter(detected_types)
        primary_type = type_counts.most_common(1)[0][0]
        avg_sent = total_sentiment / max(catalyst_count, 1)
        confidence = min(catalyst_count / 5.0, 1.0)
        return {
            "type": primary_type,
            "description": f"{primary_type.replace('_', ' ').title()} catalyst detected ({catalyst_count} articles)",
            "avg_sentiment": avg_sent, "confidence": confidence,
            "all_types": list(type_counts.keys()), "catalyst_count": catalyst_count,
        }

    @staticmethod
    def _calc_sentiment(text: str) -> float:
        positive = {"surge","soar","gain","bullish","upgrade","beat","outperform","positive","growth","profit","strong","rally","breakout","opportunity","innovation","record","expansion","launch","partnership","approval","momentum","leadership","award","success","milestone","breakthrough","dividend","buyback"}
        negative = {"decline","drop","fall","loss","bearish","downgrade","miss","underperform","negative","weak","selloff","crash","plunge","risk","warning","lawsuit","investigation","fraud","penalty","cut","reduction","layoff","restructuring","debt","default","bankruptcy","downturn","recession","inflation","volatility"}
        words = set(text.lower().split())
        pos = len(words & positive)
        neg = len(words & negative)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    @staticmethod
    def _is_catalyst_title(title: str) -> bool:
        return any(kw.lower() in title.lower() for kw in [
            "earnings","merger","acquisition","FDA","approval","launch","partnership",
            "contract","upgrade","downgrade","dividend","buyback","split",
            "lawsuit","settlement","investigation","surge","plunge","crash","rally","breakout",
        ])

    @staticmethod
    def _extract_topics(title: str) -> List[str]:
        topics_map = {
            "earnings": ["earnings","revenue","profit","quarterly","fiscal","EPS"],
            "technology": ["AI","tech","software","cloud","digital","data","chip"],
            "healthcare": ["FDA","drug","trial","biotech","medical","health"],
            "energy": ["oil","gas","energy","renewable","solar","wind"],
            "finance": ["bank","rate","interest","loan","credit","mortgage"],
            "mergers": ["merger","acquisition","takeover","buyout"],
            "macro": ["economy","inflation","GDP","jobs","consumer"],
        }
        text_lower = title.lower()
        found = []
        for topic, keywords in topics_map.items():
            if any(kw.lower() in text_lower for kw in keywords):
                found.append(topic)
        return found
