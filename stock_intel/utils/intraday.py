import datetime
import logging
from typing import List, Optional, Tuple
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

BUY_WINDOW_RULES = [
    ({"premarket_mover_down", "unusual_volume"}, (datetime.time(9, 45), datetime.time(10, 0))),
    ({"premarket_mover_up", "extreme_volume"}, (datetime.time(9, 30), datetime.time(9, 35))),
    ({"oversold", "technical_bullish"}, (datetime.time(9, 30), datetime.time(10, 0))),
    ({"premarket_mover_up"}, (datetime.time(9, 30), datetime.time(9, 45))),
]

SELL_WINDOW_DEFAULT = (datetime.time(15, 30), datetime.time(16, 0))

HOLDING_RULES = [
    ({"extreme_volume", "low_float"}, 1),
    ({"extreme_volume"}, 2),
    ({"low_float"}, 2),
]


class TradeSpec:
    def __init__(self, buy_start: datetime.time, buy_end: datetime.time,
                 sell_start: datetime.time, sell_end: datetime.time,
                 holding_days: int):
        self.buy_start = buy_start
        self.buy_end = buy_end
        self.sell_start = sell_start
        self.sell_end = sell_end
        self.holding_days = holding_days

    @staticmethod
    def from_signals(signals: Optional[List[str]]) -> "TradeSpec":
        sig = set(signals or [])
        buy_start, buy_end = datetime.time(9, 30), datetime.time(10, 0)
        for trigger_set, window in BUY_WINDOW_RULES:
            if trigger_set.issubset(sig):
                buy_start, buy_end = window
                break
        holding_days = 5
        for trigger_set, days in HOLDING_RULES:
            if trigger_set.issubset(sig):
                holding_days = days
                break
        return TradeSpec(
            buy_start=buy_start, buy_end=buy_end,
            sell_start=SELL_WINDOW_DEFAULT[0], sell_end=SELL_WINDOW_DEFAULT[1],
            holding_days=holding_days,
        )

    def buy_tip_text(self, date_str: str) -> str:
        return f"Buy at {date_str} {self.buy_start.strftime('%H:%M')}-{self.buy_end.strftime('%H:%M')}"

    def sell_tip_text(self, date_str: str) -> str:
        if self.holding_days == 1:
            return f"Sell at {date_str} {self.sell_start.strftime('%H:%M')}-{self.sell_end.strftime('%H:%M')}"
        return f"Sell at {date_str}+{self.holding_days} {self.sell_start.strftime('%H:%M')}-{self.sell_end.strftime('%H:%M')}"


def _get_intraday(ticker: str, date: datetime.date) -> Optional[pd.DataFrame]:
    for attempt in range(2):
        period = "5d" if attempt == 0 else "1mo"
        interval = "5m" if attempt == 0 else "15m"
        hist = yf.download(ticker, period=period, interval=interval, progress=False)
        if hist.empty:
            continue
        hist.index = pd.to_datetime(hist.index)
        if isinstance(hist.columns, pd.MultiIndex):
            close_series = hist["Close"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series[ticker]
        else:
            close_series = hist["Close"]
        target_ny = date.strftime("%Y-%m-%d")
        day_data = close_series[close_series.index.strftime("%Y-%m-%d") == target_ny]
        if not day_data.empty:
            return day_data.to_frame(name="close")
    return None


def _window_midpoint(day_data: pd.DataFrame, window_start: datetime.time,
                     window_end: datetime.time) -> Optional[float]:
    if day_data is None or day_data.empty:
        return None
    mask = (day_data.index.time >= window_start) & (day_data.index.time <= window_end)
    window_bars = day_data[mask]
    if window_bars.empty:
        return None
    return round(float(window_bars["close"].mean()), 4)


def calc_entry_price(ticker: str, entry_date: datetime.date,
                     trade_spec: TradeSpec) -> Tuple[Optional[float], Optional[float]]:
    day_data = _get_intraday(ticker, entry_date)
    midpoint = _window_midpoint(day_data, trade_spec.buy_start, trade_spec.buy_end)
    if midpoint is not None:
        return midpoint, midpoint
    daily = yf.download(ticker, start=entry_date, end=entry_date + datetime.timedelta(days=1), progress=False)
    if not daily.empty:
        close_val = daily["Close"].iloc[-1]
        if isinstance(close_val, pd.Series):
            close_val = close_val.iloc[0]
        return None, round(float(close_val), 4)
    return None, None


def calc_exit_price(ticker: str, exit_date: datetime.date,
                    trade_spec: TradeSpec) -> Optional[float]:
    day_data = _get_intraday(ticker, exit_date)
    midpoint = _window_midpoint(day_data, trade_spec.sell_start, trade_spec.sell_end)
    if midpoint is not None:
        return midpoint
    daily = yf.download(ticker, start=exit_date, end=exit_date + datetime.timedelta(days=1), progress=False)
    if not daily.empty:
        close_val = daily["Close"].iloc[-1]
        if isinstance(close_val, pd.Series):
            close_val = close_val.iloc[0]
        return round(float(close_val), 4)
    return None


def calc_current_price(ticker: str) -> Optional[float]:
    day_data = _get_intraday(ticker, datetime.date.today())
    if day_data is not None and not day_data.empty:
        return round(float(day_data["close"].iloc[-1]), 4)
    daily = yf.download(ticker, period="1d", progress=False)
    if not daily.empty:
        close_val = daily["Close"].iloc[-1]
        if isinstance(close_val, pd.Series):
            close_val = close_val.iloc[0]
        return round(float(close_val), 4)
    return None


def next_market_date(from_date: Optional[datetime.date] = None) -> datetime.date:
    today = from_date or datetime.date.today()
    offset = 1 if from_date is None else 0
    while True:
        d = today + datetime.timedelta(days=offset)
        if d.weekday() < 5:
            return d
        offset += 1


def planned_exit(entry_date: datetime.date, holding_days: int) -> datetime.date:
    d = entry_date
    count = 0
    while count < holding_days:
        d += datetime.timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d
