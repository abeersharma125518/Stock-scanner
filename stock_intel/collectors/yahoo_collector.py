import datetime
import logging
import time
from typing import List, Dict, Optional, Tuple, Set
import yfinance as yf
import pandas as pd
import numpy as np

from stock_intel.config.settings import CONFIG

logger = logging.getLogger(__name__)

_TICKER_CACHE: Optional[List[str]] = None


def _fetch_wikipedia_table(url: str, table_index: int = 0, column: int = 0) -> List[str]:
    try:
        import requests
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        if table_index < len(tables):
            df = tables[table_index]
            tickers = df.iloc[:, column].dropna().astype(str).str.strip().str.upper().tolist()
            valid = [t for t in tickers if t and len(t) <= 5 and t != "N/A" and not t.startswith("http")]
            logger.info(f"Fetched {len(valid)} tickers from {url.split('/')[-1]}")
            return valid
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
    return []


def _fetch_sp500() -> List[str]:
    return _fetch_wikipedia_table(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        table_index=0, column=0
    )


def _fetch_nasdaq100() -> List[str]:
    tickers = _fetch_wikipedia_table("https://en.wikipedia.org/wiki/Nasdaq-100", table_index=4, column=1)
    if not tickers:
        tickers = _fetch_wikipedia_table("https://en.wikipedia.org/wiki/Nasdaq-100", table_index=2, column=1)
    return tickers


def _fetch_dow30() -> List[str]:
    return _fetch_wikipedia_table(
        "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        table_index=1, column=2
    )


def _fetch_russell2000() -> List[str]:
    tickers = _fetch_wikipedia_table("https://en.wikipedia.org/wiki/Russell_2000_Index", table_index=2, column=0)
    if not tickers:
        tickers = _fetch_wikipedia_table("https://en.wikipedia.org/wiki/Russell_2000_Index", table_index=1, column=0)
    return tickers


def _fetch_russell1000() -> List[str]:
    tickers = _fetch_wikipedia_table("https://en.wikipedia.org/wiki/Russell_1000_Index", table_index=2, column=0)
    if not tickers:
        tickers = _fetch_wikipedia_table("https://en.wikipedia.org/wiki/Russell_1000_Index", table_index=1, column=0)
    return tickers


def _load_hardcoded_fallback() -> List[str]:
    return [
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
        "TEAM","WDAY","NET","CFLT","MDB","ESTC","OKTA","ZS","CRWD","DDOG",
        "TOST","GTLB","FOUR","BILL","WU","SQ","PYPL","TWLO","VMEO","FVRR",
        "ASML","SAP","NSRGY","TM","SONY","MUFG","SMFG","HMC","MT","RIO",
        "RDDT","AI","BBAI","IONQ","RGTI","QBTS","MSTY","PATH","AFRM","UPST",
        "JETS","XBI","IBB","XLF","XLE","XLK","XLV","XLU","XLI","XLP",
        "XLY","XLRE","XLC","VOO","SPY","QQQ","IWM","DIA","TLT","GLD",
        "SLV","USO","UNG","ARKK","ARKW",
    ]


def _clean_ticker(t: str) -> str:
    t = t.upper().replace(".", "-").replace("'", "").strip()
    if "." in t:
        t = t.split(".")[0]
    return t


def get_univers_tickers(force_refresh: bool = False) -> List[str]:
    global _TICKER_CACHE
    if _TICKER_CACHE is not None and not force_refresh:
        return _TICKER_CACHE

    all_tickers: Set[str] = set()
    sources = []

    sp500 = _fetch_sp500()
    if sp500:
        sources.append(("S&P 500", sp500))
        all_tickers.update(sp500)

    nasdaq100 = _fetch_nasdaq100()
    if nasdaq100:
        sources.append(("NASDAQ-100", nasdaq100))
        all_tickers.update(nasdaq100)

    dow30 = _fetch_dow30()
    if dow30:
        sources.append(("DJIA", dow30))
        all_tickers.update(dow30)

    russell1000 = _fetch_russell1000()
    if russell1000:
        sources.append(("Russell 1000", russell1000))
        all_tickers.update(russell1000)

    russell2000 = _fetch_russell2000()
    if russell2000:
        sources.append(("Russell 2000", russell2000))
        all_tickers.update(russell2000)

    if not all_tickers:
        logger.warning("Wikipedia fetch failed, using hardcoded fallback")
        fallback = _load_hardcoded_fallback()
        all_tickers.update(fallback)
        sources.append(("fallback", fallback))

    cleaned = sorted({_clean_ticker(t) for t in all_tickers if _clean_ticker(t) and len(t) <= 5})
    final = cleaned[:CONFIG.scanner.max_stocks_to_scan]

    source_summary = "; ".join(f"{name}: {len(tks)}" for name, tks in sources)
    logger.info(f"Universe: {len(final)} tickers ({source_summary})")

    _TICKER_CACHE = final
    return final


class YahooCollector:
    def __init__(self):
        self.rate_limit = CONFIG.collector.yahoo_rate_limit
        self.universe = get_univers_tickers()

    def fetch_batch_data(self, tickers: List[str]) -> pd.DataFrame:
        valid_tickers = [t for t in tickers if t and isinstance(t, str)]
        if not valid_tickers:
            return pd.DataFrame()
        try:
            data = yf.download(valid_tickers, period="1mo", group_by="ticker", progress=False, threads=True, auto_adjust=True)
            return data
        except Exception as e:
            logger.error(f"Error fetching batch data: {e}")
            return pd.DataFrame()

    def fetch_snapshot(self, ticker: str) -> Optional[Dict]:
        try:
            time.sleep(self.rate_limit)
            stock = yf.Ticker(ticker)
            info = stock.info if stock.info else {}
            hist = stock.history(period="1mo")
            pre = stock.history(period="5d", interval="1m")
            if hist.empty:
                logger.warning(f"No history for {ticker}")
                return None

            latest = hist.iloc[-1]
            avg_vol = hist["Volume"].tail(20).mean() if len(hist) >= 20 else hist["Volume"].mean()
            volume_ratio = latest["Volume"] / avg_vol if avg_vol > 0 else 1.0

            closes = hist["Close"].values
            rsi = self._calc_rsi(closes, 14)

            sma_20 = hist["Close"].tail(20).mean() if len(hist) >= 20 else None
            sma_50 = hist["Close"].tail(50).mean() if len(hist) >= 50 else None
            sma_200 = hist["Close"].tail(200).mean() if len(hist) >= 200 else None

            highs = hist["High"].values
            lows = hist["Low"].values
            atr = self._calc_atr(highs, lows, closes, 14) if len(highs) >= 14 else None

            premarket_change = None
            premarket_vol = None
            if not pre.empty and len(pre) > 0:
                pre_close = hist["Close"].iloc[-1] if len(hist) > 1 else latest["Close"]
                pre_open = pre.iloc[0]["Open"] if "Open" in pre.columns else None
                if pre_open and pre_close > 0:
                    premarket_change = ((pre_open - pre_close) / pre_close) * 100
                premarket_vol = int(pre["Volume"].sum()) if "Volume" in pre.columns else None

            pct_changes = hist["Close"].pct_change().dropna().values
            short_term_momentum = float(np.mean(pct_changes[-5:])) * 100 if len(pct_changes) >= 5 else 0
            mid_term_momentum = float(np.mean(pct_changes[-20:])) * 100 if len(pct_changes) >= 20 else 0

            today = datetime.date.today()
            return {
                "date": today,
                "open_price": float(latest["Open"]) if "Open" in latest else None,
                "high_price": float(latest["High"]) if "High" in latest else None,
                "low_price": float(latest["Low"]) if "Low" in latest else None,
                "close_price": float(latest["Close"]) if "Close" in latest else None,
                "volume": int(latest["Volume"]) if "Volume" in latest else 0,
                "avg_volume_20d": float(avg_vol) if avg_vol else None,
                "volume_ratio": float(volume_ratio),
                "premarket_change_pct": float(premarket_change) if premarket_change else None,
                "premarket_volume": premarket_vol,
                "rsi_14": float(rsi) if rsi else None,
                "sma_20": float(sma_20) if sma_20 else None,
                "sma_50": float(sma_50) if sma_50 else None,
                "sma_200": float(sma_200) if sma_200 else None,
                "atr_14": float(atr) if atr else None,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "name": info.get("longName") or info.get("shortName") or ticker,
                "beta": info.get("beta"),
                "short_ratio": info.get("shortRatio"),
                "float_shares": info.get("floatShares"),
                "exchange": info.get("exchange"),
                "short_term_momentum": short_term_momentum,
                "mid_term_momentum": mid_term_momentum,
            }
        except Exception as e:
            logger.warning(f"Yahoo fetch failed for {ticker}: {e}")
            return None

    def fetch_multi_snapshots(self, tickers: List[str]) -> Dict[str, Dict]:
        results = {}
        for i, ticker in enumerate(tickers):
            data = self.fetch_snapshot(ticker)
            if data:
                results[ticker.upper()] = data
            if i > 0 and i % 100 == 0:
                logger.info(f"Yahoo: processed {i}/{len(tickers)} tickers")
        return results

    def scan_universe(self) -> Dict[str, Dict]:
        return self.fetch_multi_snapshots(self.universe)

    def get_premarket_movers(self, min_change_pct: float = None) -> List[Tuple[str, float, float]]:
        if min_change_pct is None:
            min_change_pct = CONFIG.scanner.premarket_change_threshold
        movers = []
        for ticker in self.universe[:500]:
            try:
                time.sleep(self.rate_limit)
                pre = yf.download(ticker, period="2d", interval="1m", progress=False)
                hist = yf.download(ticker, period="5d", progress=False)
                if pre.empty or hist.empty:
                    continue
                prev_close = hist["Close"].iloc[-2] if len(hist) >= 2 else hist["Close"].iloc[-1]
                pre_open = pre.iloc[0]["Open"]
                if prev_close > 0:
                    change_pct = ((pre_open - prev_close) / prev_close) * 100
                    pre_vol = int(pre["Volume"].sum())
                    if abs(change_pct) >= min_change_pct:
                        movers.append((ticker, change_pct, pre_vol))
            except Exception:
                continue
        movers.sort(key=lambda x: abs(x[1]), reverse=True)
        return movers

    def get_upcoming_earnings(self, days_ahead: int = 7) -> List[Dict]:
        try:
            earnings = []
            for ticker in self.universe[:500]:
                try:
                    stock = yf.Ticker(ticker)
                    if stock.calendar and stock.calendar:
                        cal_data = stock.calendar
                        if "Earnings Date" in cal_data:
                            ed = cal_data["Earnings Date"]
                            if isinstance(ed, pd.Timestamp):
                                ed_date = ed.date()
                                if datetime.date.today() <= ed_date <= (datetime.date.today() + datetime.timedelta(days=days_ahead)):
                                    earnings.append({
                                        "ticker": ticker,
                                        "report_date": ed_date,
                                        "eps_estimate": cal_data.get("EPS Estimate"),
                                        "revenue_estimate": cal_data.get("Revenue Estimate"),
                                    })
                except Exception:
                    continue
            return earnings
        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {e}")
            return []

    @staticmethod
    def _calc_rsi(prices: np.ndarray, period: int = 14) -> Optional[float]:
        if len(prices) < period + 1:
            return None
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> Optional[float]:
        if len(highs) < period + 1:
            return None
        trs = []
        for i in range(1, len(highs)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        return float(np.mean(trs[-period:]))
