from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class CrossSymbolFeatures:
    """Data structure representing cross-symbol features for a specific symbol"""
    timestamp: datetime
    symbol: str
    sector: Optional[str] = None

    # Comparison against sector-wide data
    sector_sentiment_mean: Optional[float] = None
    sector_sentiment_skew: Optional[float] = None
    sector_sentiment_std: Optional[float] = None
    sector_news_volume: Optional[int] = None
    relative_sentiment_ratio: Optional[float] = None
    sector_sentiment_correlation: Optional[float] = None
    sector_sentiment_divergence: Optional[float] = None

    # Comparison against market-wide data
    market_sentiment_correlation: Optional[float] = None
    market_sentiment_divergence: Optional[float] = None
