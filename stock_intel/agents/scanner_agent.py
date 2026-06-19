import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any
from stock_intel.agents.base_agent import BaseAgent
from stock_intel.collectors.yahoo_collector import YahooCollector
from stock_intel.collectors.finviz_collector import FinVizCollector
from stock_intel.db.database import DatabaseManager
from stock_intel.config.settings import CONFIG

logger = logging.getLogger(__name__)


class ScannerAgent(BaseAgent):
    def __init__(self, db: DatabaseManager, config: Optional[Dict] = None):
        super().__init__(db, config)
        self.yahoo = YahooCollector()
        self.finviz = FinVizCollector()
        self.min_price = CONFIG.scanner.min_price
        self.max_price = CONFIG.scanner.max_price
        self.min_volume = CONFIG.scanner.min_volume
        self.volume_surge = CONFIG.scanner.volume_surge_threshold
        self.premarket_threshold = CONFIG.scanner.premarket_change_threshold

    def validate(self) -> bool:
        return True

    def execute(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.log_start()
        results = {
            "unusual_volume": [],
            "premarket_movers": [],
            "finviz_signals": [],
            "all_snapshots": {},
            "timestamp": datetime.datetime.now().isoformat(),
        }

        finviz_data = self.finviz.run_all_screeners()
        for screener, stocks in finviz_data.items():
            results["finviz_signals"].extend(stocks)
        logger.info(f"FinViz: {sum(len(v) for v in finviz_data.values())} total signals")

        tickers_to_scan = set()
        for stocks in finviz_data.values():
            for s in stocks:
                tickers_to_scan.add(s["ticker"])
        for t in self.yahoo.universe:
            tickers_to_scan.add(t)

        ticker_list = list(tickers_to_scan)[:CONFIG.scanner.max_stocks_to_scan]
        logger.info(f"Scanning {len(ticker_list)} tickers...")

        active_stocks = []

        snapshots = self.yahoo.fetch_multi_snapshots(ticker_list)
        results["all_snapshots"] = snapshots
        logger.info(f"Yahoo: {len(snapshots)} valid snapshots")

        for ticker, data in snapshots.items():
            if not data:
                continue
            price = data.get("close_price") or data.get("open_price", 0)
            volume = data.get("volume", 0)
            vol_ratio = data.get("volume_ratio", 1.0)

            if price is None or price < self.min_price or price > self.max_price:
                continue
            if volume < self.min_volume:
                continue

            stock_data = {
                "ticker": ticker, "price": price, "volume": volume,
                "volume_ratio": vol_ratio, "premarket_change": data.get("premarket_change_pct"),
                "rsi": data.get("rsi_14"), "sma_20": data.get("sma_20"),
                "sma_50": data.get("sma_50"), "atr": data.get("atr_14"),
                "beta": data.get("beta"), "market_cap": data.get("market_cap"),
                "sector": data.get("sector"), "industry": data.get("industry"),
                "exchange": data.get("exchange"),
            }

            stock_id = self.db.get_or_create_stock_id(ticker)
            if stock_id:
                snapshot_data = {k: v for k, v in data.items()
                                 if k not in ("date", "sector", "industry", "market_cap",
                                              "name", "beta", "exchange", "short_ratio", "float_shares")}
                self.db.save_snapshot(stock_id, data["date"], snapshot_data)

            active_stocks.append(stock_data)

            if vol_ratio >= self.volume_surge and stock_id:
                results["unusual_volume"].append(stock_data)
                self.db.save_signal(stock_id, data["date"], "unusual_volume",
                                    float(min(vol_ratio / 3.0, 1.0)),
                                    {"volume_ratio": vol_ratio, "volume": volume})

            pm_change = data.get("premarket_change_pct")
            if pm_change is not None and abs(pm_change) >= self.premarket_threshold and stock_id:
                results["premarket_movers"].append(stock_data)
                self.db.save_signal(stock_id, data["date"], "premarket_mover",
                                    float(min(abs(pm_change) / 10.0, 1.0)),
                                    {"change_pct": pm_change})

        results["unusual_volume"].sort(key=lambda x: x["volume_ratio"], reverse=True)
        results["premarket_movers"].sort(key=lambda x: abs(x.get("premarket_change", 0)), reverse=True)
        self.results = results
        self.log_end()
        return results
