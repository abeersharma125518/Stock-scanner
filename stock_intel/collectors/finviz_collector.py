import datetime
import logging
import re
import time
from typing import List, Dict, Optional, Tuple
from stock_intel.config.settings import CONFIG

logger = logging.getLogger(__name__)


class FinVizCollector:
    def __init__(self):
        self.timeout = CONFIG.collector.finviz_timeout
        self.screeners = {
            "unusual_volume": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o500,sh_price_o2,sh_relvol_o1.5&ft=4",
            "premarket_gainers": "https://finviz.com/screener.ashx?v=111&f=sh_price_o2,sh_relvol_o1.5,ta_premarketgain_o1&ft=4",
            "oversold": "https://finviz.com/screener.ashx?v=111&f=sh_price_o2,sh_relvol_o1,ta_rsi_ob30&ft=4",
            "overbought": "https://finviz.com/screener.ashx?v=111&f=sh_price_o2,sh_relvol_o1,ta_rsi_os70&ft=4",
            "new_high": "https://finviz.com/screener.ashx?v=111&f=sh_price_o2,sh_relvol_o1,ta_highlow20d_ish&ft=4",
            "new_low": "https://finviz.com/screener.ashx?v=111&f=sh_price_o2,sh_relvol_o1,ta_highlow20d_isl&ft=4",
            "gap_up": "https://finviz.com/screener.ashx?v=111&f=sh_price_o2,sh_relvol_o1,ta_gap_u&ft=4",
            "gap_down": "https://finviz.com/screener.ashx?v=111&f=sh_price_o2,sh_relvol_o1,ta_gap_d&ft=4",
            "most_active": "https://finviz.com/screener.ashx?v=111&f=sh_price_o2&ft=4",
        }

    def run_screener(self, screener_name: str = "unusual_volume") -> List[Dict]:
        url = self.screeners.get(screener_name)
        if not url:
            logger.warning(f"Unknown screener: {screener_name}")
            return []
        try:
            import requests
            from bs4 import BeautifulSoup
            headers = {
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/120.0.0.0 Safari/537.36"),
            }
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.select("table[bgcolor='#d3d3d3'] tr")
            if not rows:
                rows = soup.select("tr.table-dark-row-cp")
            results = []
            for row in rows[1:]:
                cols = row.find_all("td")
                if len(cols) < 10:
                    continue
                try:
                    ticker = cols[1].get_text(strip=True) if len(cols) > 1 else ""
                    if not ticker:
                        continue
                    price_text = cols[8].get_text(strip=True) if len(cols) > 8 else "0"
                    change_text = cols[9].get_text(strip=True) if len(cols) > 9 else "0%"
                    volume_text = cols[10].get_text(strip=True) if len(cols) > 10 else "0"
                    relvol_text = cols[11].get_text(strip=True) if len(cols) > 11 else "0"
                    avgvol_text = cols[12].get_text(strip=True) if len(cols) > 12 else "0"
                    market_cap_text = cols[7].get_text(strip=True) if len(cols) > 7 else ""
                    try:
                        price = float(price_text.replace("$", "").replace(",", ""))
                    except ValueError:
                        price = 0.0
                    change_pct = self._parse_pct(change_text)
                    volume = self._parse_number(volume_text)
                    rel_vol = self._parse_number(relvol_text)
                    avg_vol = self._parse_number(avgvol_text)
                    market_cap = self._parse_market_cap(market_cap_text)
                    results.append({
                        "ticker": ticker.upper(), "company": "", "sector": "",
                        "price": price, "change_pct": change_pct, "volume": volume,
                        "relative_volume": rel_vol, "avg_volume": avg_vol,
                        "market_cap": market_cap, "screener": screener_name,
                        "captured_at": datetime.datetime.now(),
                    })
                except (ValueError, IndexError):
                    continue
            logger.info(f"FinViz {screener_name}: {len(results)} results")
            return results
        except ImportError:
            logger.warning("BeautifulSoup not available for FinViz")
            return []
        except Exception as e:
            logger.warning(f"FinViz {screener_name} error: {e}")
            return []

    def run_all_screeners(self) -> Dict[str, List[Dict]]:
        results = {}
        for name in self.screeners:
            try:
                data = self.run_screener(name)
                results[name] = data
                time.sleep(1)
            except Exception as e:
                logger.error(f"Failed screener {name}: {e}")
                results[name] = []
        return results

    def get_consolidated_tickers(self) -> Dict[str, List[str]]:
        results = self.run_all_screeners()
        ticker_data = {}
        for name, stocks in results.items():
            for s in stocks:
                t = s["ticker"]
                if t not in ticker_data:
                    ticker_data[t] = {"screeners": [], "total_signals": 0, "data": s}
                ticker_data[t]["screeners"].append(name)
                ticker_data[t]["total_signals"] += 1
        return ticker_data

    @staticmethod
    def _parse_pct(text: str) -> Optional[float]:
        try:
            text = text.replace("%", "").replace("+", "").strip()
            if text == "-":
                return 0.0
            return float(text)
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _parse_number(text: str) -> Optional[int]:
        try:
            text = text.strip().replace(",", "")
            if text == "-" or not text:
                return None
            if "B" in text:
                return int(float(text.replace("B", "")) * 1_000_000_000)
            if "M" in text:
                return int(float(text.replace("M", "")) * 1_000_000)
            if "K" in text:
                return int(float(text.replace("K", "")) * 1_000)
            return int(float(text))
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _parse_market_cap(text: str) -> Optional[float]:
        try:
            text = text.strip().replace("$", "").replace(",", "")
            if text == "-" or not text:
                return None
            if "B" in text:
                return float(text.replace("B", "")) * 1e9
            if "M" in text:
                return float(text.replace("M", "")) * 1e6
            if "T" in text:
                return float(text.replace("T", "")) * 1e12
            return float(text)
        except (ValueError, AttributeError):
            return None
