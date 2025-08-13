from dataclasses import dataclass
from datetime import datetime

@dataclass
class StockData:
    """Data structure for stock price data"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float