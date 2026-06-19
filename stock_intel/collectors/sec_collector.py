import datetime
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from stock_intel.config.settings import CONFIG

logger = logging.getLogger(__name__)


class SECCollector:
    def __init__(self):
        self.rate_limit = CONFIG.collector.sec_rate_limit
        self.base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        self.headers = {
            "User-Agent": CONFIG.collector.reddit_user_agent or "StockIntel/1.0 (contact@example.com)",
            "Accept-Encoding": "gzip, deflate",
        }

    def fetch_insider_transactions(self, ticker: str, days_back: int = 30) -> List[Dict]:
        try:
            import sec_api
            api_key = CONFIG.collector.reddit_client_secret
            if api_key:
                return self._fetch_via_sec_api(ticker, days_back, api_key)
            else:
                return self._fetch_via_edgar(ticker, days_back)
        except ImportError:
            return self._fetch_via_edgar(ticker, days_back)
        except Exception as e:
            logger.warning(f"SEC insider fetch failed for {ticker}: {e}")
            return []

    def _fetch_via_sec_api(self, ticker: str, days_back: int, api_key: str) -> List[Dict]:
        try:
            from sec_api import InsiderTradingApi
            api = InsiderTradingApi(api_key)
            results = api.get_data({
                "ticker": ticker,
                "fromDate": (datetime.date.today() - datetime.timedelta(days=days_back)).isoformat(),
                "toDate": datetime.date.today().isoformat(),
            })
            transactions = []
            for t in results.get("transactions", []):
                transactions.append({
                    "filing_date": self._parse_date(t.get("filingDate")),
                    "transaction_date": self._parse_date(t.get("transactionDate")),
                    "insider_name": t.get("insiderName", {}).get("fullName", "Unknown"),
                    "title": t.get("insiderName", {}).get("officialTitle", ""),
                    "transaction_type": t.get("transactionType", ""),
                    "shares_traded": float(t.get("shares", 0)),
                    "price": float(t.get("price", 0)) if t.get("price") else None,
                    "shares_held": float(t.get("sharesHeld", 0)),
                    "trade_value": float(t.get("shares", 0)) * float(t.get("price", 1)),
                    "is_direct": t.get("direct", True),
                })
            return transactions
        except Exception as e:
            logger.warning(f"SEC API error for {ticker}: {e}")
            return []

    def _fetch_via_edgar(self, ticker: str, days_back: int) -> List[Dict]:
        try:
            import requests
            cik = self._lookup_cik(ticker)
            if not cik:
                return []
            filings_url = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4&dateb=&owner=only&start=0&count=20&output=atom")
            resp = requests.get(filings_url, headers=self.headers, timeout=30)
            if resp.status_code != 200:
                return []
            return self._parse_insider_filings(resp.text, ticker)
        except Exception as e:
            logger.warning(f"EDGAR fetch error for {ticker}: {e}")
            return []

    def _lookup_cik(self, ticker: str) -> Optional[str]:
        try:
            import requests
            url = "https://www.sec.gov/files/company_tickers.json"
            resp = requests.get(url, headers=self.headers, timeout=30)
            if resp.status_code != 200:
                return None
            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)
            return None
        except Exception as e:
            logger.warning(f"CIK lookup failed for {ticker}: {e}")
            return None

    def _parse_insider_filings(self, xml_text: str, ticker: str) -> List[Dict]:
        transactions = []
        try:
            root = ET.fromstring(xml_text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                filing_date = entry.find("atom:updated", ns)
                form_type = entry.find("atom:category", ns)
                if form_type is not None and "4" in (form_type.get("term", "")):
                    link = entry.find("atom:link", ns)
                    href = link.get("href") if link is not None else ""
                    transactions.append({
                        "filing_date": self._parse_date(filing_date.text if filing_date is not None else None),
                        "transaction_date": self._parse_date(filing_date.text if filing_date is not None else None),
                        "insider_name": ticker, "title": "", "transaction_type": "filing-4",
                        "shares_traded": 0, "price": None, "shares_held": 0, "trade_value": 0, "is_direct": True,
                    })
        except Exception as e:
            logger.warning(f"Parse insider filings error: {e}")
        return transactions

    def fetch_earnings_calendar(self, days_ahead: int = 14) -> List[Dict]:
        try:
            import requests
            from sec_api import QueryApi
            api_key = CONFIG.collector.reddit_client_secret
            if not api_key:
                return []
            queryApi = QueryApi(api_key=api_key)
            today = datetime.date.today()
            end = today + datetime.timedelta(days=days_ahead)
            query = f"SELECT * FROM earnings WHERE reportDate >= '{today.isoformat()}' AND reportDate <= '{end.isoformat()}'"
            results = queryApi.get_filings(query)
            earnings = []
            for filing in results.get("filings", []):
                earnings.append({
                    "ticker": filing.get("ticker", ""),
                    "report_date": self._parse_date(filing.get("reportDate")),
                    "fiscal_quarter": filing.get("fiscalQuarter"),
                    "fiscal_year": filing.get("fiscalYear"),
                    "eps_estimate": None, "eps_actual": None, "is_confirmed": True,
                })
            return earnings
        except Exception as e:
            logger.warning(f"SEC earnings calendar error: {e}")
            return []

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime.date]:
        if not date_str:
            return None
        if isinstance(date_str, datetime.date):
            return date_str
        try:
            if "T" in date_str:
                return datetime.datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            return datetime.datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None
