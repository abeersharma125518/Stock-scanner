"""
Rebuild universe CSVs from NASDAQ/NYSE symbol directories (no Wikipedia).
Filters out ETFs, preferred shares, warrants, rights, units, and test issues.
Run whenever you want to refresh the universe:
    python stock_intel/data/universe/rebuild_universe.py
"""
import csv
import logging
import os
import re
import sys
import urllib.request
from typing import List, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("rebuild_universe")

UNIVERSE_DIR = os.path.dirname(os.path.abspath(__file__))

# Patterns that indicate non-common-stock tickers
PREFERRED_RE = re.compile(
    r"(preferred|pfd\.|depositary|depository|% pfd|% cum|unit|equity warrant)",
    re.IGNORECASE,
)
# Ticker suffixes that indicate non-common-stock
BAD_SUFFIXES_RE = re.compile(
    r"\.PR|\.WT|\.WI|\.U$|\.RT$|\.WS$|[\^\-]",
    re.IGNORECASE,
)
BAD_PREFIXES_RE = re.compile(r"^[\^\.]", re.IGNORECASE)


def _download_ftp(url: str) -> str:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8-sig")


def _parse_nasdaq_listed(data: str) -> List[Tuple[str, str, bool]]:
    """
    Parse nasdaqlisted.txt.
    Returns list of (symbol, security_name, is_etf).
    """
    results = []
    for line in data.strip().split("\n"):
        if not line.strip() or line.startswith("Symbol|"):
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        symbol = parts[0].strip().upper()
        sec_name = parts[1].strip()
        is_etf = parts[6].strip().upper() == "Y"
        is_test = parts[3].strip().upper() == "Y"
        if is_test:
            continue
        results.append((symbol, sec_name, is_etf))
    return results


def _parse_other_listed(data: str) -> List[Tuple[str, str, bool]]:
    """
    Parse otherlisted.txt (NYSE/AMEX/ARCA).
    Returns list of (symbol, security_name, is_etf).
    """
    results = []
    for line in data.strip().split("\n"):
        if not line.strip() or line.startswith("ACT Symbol|"):
            continue
        parts = line.split("|")
        if len(parts) < 8:
            continue
        symbol = parts[0].strip().upper()
        sec_name = parts[1].strip()
        is_etf = parts[4].strip().upper() == "Y"
        is_test = parts[6].strip().upper() == "Y"
        if is_test:
            continue
        results.append((symbol, sec_name, is_etf))
    return results


def _is_valid_common_stock(symbol: str, sec_name: str, is_etf: bool) -> bool:
    """Check if this is a valid common stock ticker."""
    if is_etf:
        return False
    if len(symbol) > 5:
        return False
    if BAD_PREFIXES_RE.search(symbol):
        return False
    if BAD_SUFFIXES_RE.search(symbol):
        return False
    if PREFERRED_RE.search(sec_name):
        return False
    if "ETF" in sec_name.upper():
        return False
    # Skip obvious ETFs, notes, and funds
    skip_keywords = [
        "TRUST", "FUND", "ETF", "NOTE", "%", "PFD", "DEPOSITORY",
        "WARRANT", "RIGHT", "UNIT", "ACKNOWLEDGEMENT", "CERTIFICATE",
        "DTC", "LETTER", "CONTINGENT", "CONTINGENCY",
    ]
    for kw in skip_keywords:
        if kw in sec_name.upper():
            return False

    return True


def _classify_instrument(symbol: str, sec_name: str) -> str:
    """Classify a security for reporting purposes."""
    name_upper = sec_name.upper()
    if symbol.endswith("W") or "WARRANT" in name_upper:
        return "warrant"
    if " RIGHT" in name_upper or symbol.endswith("R"):
        return "right"
    if "ETF" in name_upper:
        return "etf"
    if "%" in symbol or "PFD" in name_upper or "PREFERRED" in name_upper or "DEPOSITARY" in name_upper:
        return "preferred"
    if "UNIT" in name_upper:
        return "unit"
    if len(symbol) > 5:
        return "long_ticker"
    if any(c in symbol for c in "-^."):
        return "malformed"
    if symbol.startswith(".") or symbol.startswith("^"):
        return "malformed"
    return "common_stock"


def main():
    logger.info("Downloading NASDAQ listed symbols...")
    nasdaq_raw = _download_ftp("ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt")
    nasdaq_parsed = _parse_nasdaq_listed(nasdaq_raw)
    logger.info(f"  Got {len(nasdaq_parsed)} symbols from NASDAQ")

    logger.info("Downloading NYSE/AMEX/ARCA listed symbols...")
    other_raw = _download_ftp("ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt")
    other_parsed = _parse_other_listed(other_raw)
    logger.info(f"  Got {len(other_parsed)} symbols from NYSE/AMEX/ARCA")

    all_symbols: List[Tuple[str, str, bool]] = nasdaq_parsed + other_parsed

    # Deduplicate by symbol (prefer NASDAQ version)
    seen: Set[str] = set()
    unique: List[Tuple[str, str, bool]] = []
    nasdaq_symbols = {s for s, _, _ in nasdaq_parsed}
    for symbol, sec_name, is_etf in all_symbols:
        if symbol in seen:
            continue
        # If a symbol is in both NASDAQ and other, and the other data says it's
        # an ETF but NASDAQ disagrees, trust NASDAQ (we already prefer NASDAQ by order)
        seen.add(symbol)
        unique.append((symbol, sec_name, is_etf))

    logger.info(f"Unique symbols after dedup: {len(unique)}")

    # Classify and filter
    common_stocks: List[Tuple[str, str]] = []
    removed_counts = {
        "etf": 0, "preferred": 0, "warrant": 0, "right": 0,
        "unit": 0, "long_ticker": 0, "malformed": 0,
    }
    for symbol, sec_name, is_etf in unique:
        cls = _classify_instrument(symbol, sec_name)
        if cls in removed_counts:
            removed_counts[cls] += 1
        if _is_valid_common_stock(symbol, sec_name, is_etf):
            common_stocks.append((symbol, sec_name))

    logger.info(f"")
    logger.info(f"{'Classification':<20} {'Count':>8}")
    logger.info(f"{'-'*20} {'-'*8}")
    for cls, count in sorted(removed_counts.items()):
        logger.info(f"{cls:<20} {count:>8}")
    logger.info(f"{'common_stock':<20} {len(common_stocks):>8}")

    # Save universe CSV
    symbols_only = sorted({s for s, _ in common_stocks})
    csv_path = os.path.join(UNIVERSE_DIR, "russell2000.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Symbol"])
        for s in symbols_only:
            writer.writerow([s])
    logger.info(f"\nSaved {len(symbols_only)} tickers to {csv_path}")

    # Also check for old sp500/nasdaq100 CSVs
    for fname in ("sp500.csv", "nasdaq100.csv"):
        path = os.path.join(UNIVERSE_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("Symbol")]
            logger.info(f"  {fname}: {len(lines)} tickers (retained from existing)")


if __name__ == "__main__":
    main()
