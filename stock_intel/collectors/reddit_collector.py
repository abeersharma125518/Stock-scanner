import datetime
import logging
import re
import time
from typing import List, Dict, Optional
from stock_intel.config.settings import CONFIG

logger = logging.getLogger(__name__)

TICKER_PATTERN = re.compile(r'\b[A-Z]{1,5}\b')
STOCK_CASHTAG = re.compile(r'\$([A-Z]{1,5})')
KNOWN_TICKERS = None


def load_ticker_set() -> set:
    global KNOWN_TICKERS
    if KNOWN_TICKERS is not None:
        return KNOWN_TICKERS
    try:
        KNOWN_TICKERS = {
            "AAPL","MSFT","GOOGL","AMZN","NVDA","META","BRK.B","TSLA","UNH","JPM",
            "V","XOM","JNJ","WMT","PG","MA","CVX","HD","MRK","ABBV",
            "BAC","KO","PEP","COST","CRM","AVGO","CSCO","MCD","ADBE","NFLX",
            "TMO","WFC","ABT","DHR","CMCSA","AMD","LIN","NKE","DIS","PM",
            "NEE","VZ","TXN","INTU","IBM","RTX","AMGN","HON","QCOM","UPS",
            "SBUX","BA","MS","GS","C","AXP","BLK","MMM","CAT","SPGI",
            "DE","GE","LMT","PLD","BKNG","SYK","MDT","ADP","GILD","ISRG",
            "TJX","LRCX","AMAT","CB","EL","MO","CL","ZTS","PGR","EOG",
            "MCK","TMUS","SO","DUK","AEP","SRE","EXC","PEG","ED","WELL",
            "PSA","EQIX","DLR","AVB","EIX","XEL","ES","DTE","AEE","CMS",
            "PLTR","SMCI","SNOW","DASH","ABNB","UBER","LYFT","DKNG","RBLX","COIN",
            "HOOD","SOFI","MSTR","RDDT","ARM","CVNA","CHWY","GME","AMC","BB",
            "F","GM","CCL","NCLH","RCL","AAL","UAL","DAL","LUV","BA",
            "TSM","BABA","PDD","NIO","LI","XPEV","BIDU","JD","TCEHY","NTES",
        }
    except Exception:
        KNOWN_TICKERS = set()
    return KNOWN_TICKERS


def extract_tickers(text: str) -> List[str]:
    tickers = set()
    for match in STOCK_CASHTAG.finditer(text):
        tickers.add(match.group(1))
    known = load_ticker_set()
    for match in TICKER_PATTERN.finditer(text):
        token = match.group(0)
        if token in known and token not in ("A","I","IT","AT","AS","IS","TO","BE","GO","NO","ON","FOR","ARE"):
            tickers.add(token)
    return list(tickers)


class RedditCollector:
    def __init__(self):
        self.reddit = None
        self._initialized = False
        self.subreddits = CONFIG.sentiment.reddit_subreddits
        self.posts_limit = CONFIG.sentiment.reddit_posts_limit
        self._init_client()

    def _init_client(self):
        cid = CONFIG.collector.reddit_client_id
        secret = CONFIG.collector.reddit_client_secret
        if cid and secret:
            try:
                import praw
                self.reddit = praw.Reddit(client_id=cid, client_secret=secret, user_agent=CONFIG.collector.reddit_user_agent)
                self._initialized = True
                logger.info("Reddit client initialized")
            except Exception as e:
                logger.warning(f"Failed to init Reddit client: {e}")

    def fetch_mentions(self, subreddit_name: str = "wallstreetbets", limit: int = None) -> List[Dict]:
        if not self._initialized or not self.reddit:
            logger.warning("Reddit client not available, using fallback")
            return self._fallback_fetch(subreddit_name, limit or self.posts_limit)
        mentions = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            for post in subreddit.hot(limit=limit or self.posts_limit):
                tickers = extract_tickers(post.title + " " + (post.selftext or ""))
                if tickers:
                    sentiment = self._simple_sentiment(post.title)
                    mentions.append({
                        "ticker": tickers[0],
                        "all_tickers": tickers,
                        "subreddit": subreddit_name,
                        "post_title": post.title,
                        "post_url": f"https://reddit.com{post.permalink}",
                        "post_score": post.score,
                        "num_comments": post.num_comments,
                        "upvote_ratio": post.upvote_ratio,
                        "sentiment_score": sentiment,
                        "sentiment_label": "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral",
                        "created_at": datetime.datetime.fromtimestamp(post.created_utc),
                        "is_self_post": hasattr(post, "is_self") and post.is_self,
                    })
                time.sleep(0.3)
        except Exception as e:
            logger.warning(f"Reddit fetch error for {subreddit_name}: {e}")
        return mentions

    def fetch_all_subreddits(self) -> Dict[str, List[Dict]]:
        results = {}
        for sub in self.subreddits:
            try:
                mentions = self.fetch_mentions(sub)
                results[sub] = mentions
            except Exception as e:
                logger.error(f"Failed to fetch {sub}: {e}")
                results[sub] = []
        return results

    def fetch_consolidated(self) -> Dict[str, List[Dict]]:
        all_mentions = self.fetch_all_subreddits()
        consolidated = {}
        for sub, mentions in all_mentions.items():
            for m in mentions:
                ticker = m["ticker"]
                if ticker not in consolidated:
                    consolidated[ticker] = []
                consolidated[ticker].append(m)
        return consolidated

    def get_sentiment_summary(self, ticker: str, mentions: List[Dict]) -> Optional[Dict]:
        if not mentions:
            return None
        scores = [m["sentiment_score"] for m in mentions if m["sentiment_score"] is not None]
        if not scores:
            return None
        avg_sentiment = sum(scores) / len(scores)
        total_score = sum(m["post_score"] for m in mentions if m["post_score"])
        total_comments = sum(m["num_comments"] for m in mentions if m["num_comments"])
        return {
            "ticker": ticker, "mention_count": len(mentions),
            "avg_sentiment": avg_sentiment, "total_score": total_score,
            "total_comments": total_comments, "subreddits": list(set(m["subreddit"] for m in mentions)),
            "sentiment_label": "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral",
        }

    def _fallback_fetch(self, subreddit_name: str, limit: int) -> List[Dict]:
        try:
            import requests
            url = f"https://www.reddit.com/r/{subreddit_name}/hot.json?limit={min(limit, 100)}"
            headers = {"User-Agent": CONFIG.collector.reddit_user_agent}
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                return []
            data = resp.json()
            mentions = []
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                text = title + " " + selftext
                tickers = extract_tickers(text)
                if tickers:
                    sentiment = self._simple_sentiment(title)
                    created = datetime.datetime.fromtimestamp(post.get("created_utc", 0))
                    mentions.append({
                        "ticker": tickers[0], "all_tickers": tickers, "subreddit": subreddit_name,
                        "post_title": title, "post_url": f"https://reddit.com{post.get('permalink', '')}",
                        "post_score": post.get("score", 0), "num_comments": post.get("num_comments", 0),
                        "upvote_ratio": post.get("upvote_ratio", 0.5),
                        "sentiment_score": sentiment,
                        "sentiment_label": "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral",
                        "created_at": created, "is_self_post": post.get("is_self", False),
                    })
            return mentions
        except Exception as e:
            logger.warning(f"Fallback Reddit fetch error: {e}")
            return []

    @staticmethod
    def _simple_sentiment(text: str) -> float:
        positive_words = {
            "moon","rocket","bullish","boom","pump","gain","profit","rich",
            "calls","long","buy","breakout","squeeze","tendies","rip",
            "green","up","high","growth","strong","beat","surge","rally",
            "opportunity","value","hodl","yolo","diamond","hands","launch",
        }
        negative_words = {
            "dump","bearish","crash","loss","bagholder","shorts","sell",
            "put","decline","drop","fall","weak","downgrade","fear",
            "red","down","low","panic","selloff","plunge","tank",
            "capitulation","recession","bankrupt","fraud","lawsuit","short",
        }
        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total
