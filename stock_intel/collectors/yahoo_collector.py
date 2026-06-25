import datetime
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Set
import yfinance as yf
import pandas as pd
import numpy as np

from stock_intel.config.settings import CONFIG

logger = logging.getLogger(__name__)

_TICKER_CACHE: Optional[List[str]] = None
_UNIVERSE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "universe")


def _load_csv_universe(filename: str) -> List[str]:
    path = os.path.join(_UNIVERSE_DIR, filename)
    if not os.path.exists(path):
        logger.warning(f"Universe CSV not found: {path}")
        return []
    try:
        df = pd.read_csv(path)
        if "Symbol" not in df.columns:
            logger.warning(f"Missing 'Symbol' column in {filename}")
            return []
        tickers = df["Symbol"].dropna().astype(str).str.strip().str.upper().tolist()
        valid = sorted({t for t in tickers if t and len(t) <= 5 and t != "N/A"})
        logger.info(f"Loaded {len(valid)} tickers from {filename}")
        return valid
    except Exception as e:
        logger.warning(f"Failed to load {filename}: {e}")
        return []


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

    for fname, label in [("sp500.csv", "S&P 500"), ("nasdaq100.csv", "NASDAQ-100"), ("russell2000.csv", "Russell 2000")]:
        tickers = _load_csv_universe(fname)
        if tickers:
            sources.append((label, tickers))
            all_tickers.update(tickers)

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

    def fetch_batch_data(self, tickers: List[str], chunk_size: int = 200) -> pd.DataFrame:
        valid_tickers = [t for t in tickers if t and isinstance(t, str)]
        if not valid_tickers:
            return pd.DataFrame()
        try:
            if len(valid_tickers) <= chunk_size:
                data = yf.download(valid_tickers, period="1mo", group_by="ticker", progress=False, threads=True, auto_adjust=True)
                return data
            chunks = [valid_tickers[i:i + chunk_size] for i in range(0, len(valid_tickers), chunk_size)]
            logger.info(f"Batch download: {len(valid_tickers)} tickers in {len(chunks)} chunks of {chunk_size}")
            frames = []
            for i, chunk in enumerate(chunks):
                try:
                    chunk_data = yf.download(chunk, period="1mo", group_by="ticker", progress=False, threads=True, auto_adjust=True)
                    if not chunk_data.empty:
                        frames.append(chunk_data)
                except Exception as e:
                    logger.warning(f"Chunk {i + 1}/{len(chunks)} failed: {e}")
            if frames:
                return pd.concat(frames, axis=1)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching batch data: {e}")
            return pd.DataFrame()

    @staticmethod
    def _extract_batch_ticker(batch: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        if batch.empty:
            return None
        if hasattr(batch.columns, "levels"):
            candidates = [ticker, ticker.upper(), ticker.lower()]
            for c in candidates:
                if c in batch.columns.get_level_values(0):
                    return batch[c]
            return None
        else:
            return batch if not batch.empty else None

    def _build_snapshot_from_batch(self, ticker: str, batch: pd.DataFrame) -> Optional[Dict]:
        try:
            time.sleep(self.rate_limit)
            hist = self._extract_batch_ticker(batch, ticker)
            if hist is None or hist.empty or "Close" not in hist.columns:
                return None

            stock = yf.Ticker(ticker)
            info = stock.info if stock.info else {}
            pre = stock.history(period="5d", interval="1m")

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

    def fetch_multi_snapshots(self, tickers: List[str], max_survivors: int = 500) -> Dict[str, Dict]:
        results = {}
        valid = [t for t in tickers if t and isinstance(t, str)]
        if not valid:
            return results

        logger.info(f"Downloading {len(valid)} symbols...")
        batch = self.fetch_batch_data(valid)

        if batch.empty:
            logger.warning("Batch download empty, falling back to individual fetches")
            for i, ticker in enumerate(valid):
                data = self.fetch_snapshot(ticker)
                if data:
                    results[ticker.upper()] = data
                if i > 0 and i % 100 == 0:
                    logger.info(f"Yahoo: processed {i}/{len(valid)} tickers")
            scan_stats = {"scanned": len(valid), "succeeded": len(results), "failed": len(valid) - len(results)}
            logger.info(f"Scan: {scan_stats['succeeded']}/{scan_stats['scanned']} ok, {scan_stats['failed']} failed")
            return results

        candidates = []
        download_fails = 0
        for ticker in valid:
            try:
                hist = self._extract_batch_ticker(batch, ticker)
                if hist is None or hist.empty or "Close" not in hist.columns:
                    download_fails += 1
                    continue
                latest = hist.iloc[-1]
                price = float(latest.get("Close", 0))
                volume = int(latest.get("Volume", 0))
                if price >= CONFIG.scanner.min_price and price <= CONFIG.scanner.max_price and volume >= CONFIG.scanner.min_volume:
                    candidates.append((ticker, price * volume))
            except Exception:
                download_fails += 1
                continue

        candidates.sort(key=lambda x: x[1], reverse=True)
        survivors = [t for t, _ in candidates[:max_survivors]]
        filtered_out = len(candidates) - len(survivors)

        logger.info(f"Downloaded: {len(valid)} | Failed: {download_fails} | Passed filters: {len(candidates)} | Fetching info: {len(survivors)} (sorted by dollar volume, top {max_survivors})")

        _lock = threading.Lock()

        def _fetch(ticker):
            data = self._build_snapshot_from_batch(ticker, batch)
            if data:
                with _lock:
                    results[ticker.upper()] = data

        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(_fetch, survivors))

        scan_stats = {"scanned": len(valid), "succeeded": len(results), "failed": download_fails + (len(survivors) - len(results))}
        logger.info(f"Scan: {scan_stats['succeeded']}/{scan_stats['scanned']} ok, {scan_stats['failed']} failed")
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
