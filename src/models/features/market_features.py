from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class MarketFeatures:
    """Data structure representing market features over a 24-hour sliding window"""
    timestamp: datetime
    # Market-based sentiment features
    market_sentiment_mean: Optional[float] = None
    market_sentiment_skew: Optional[float] = None
    market_sentiment_std: Optional[float] = None
    market_sentiment_momentum: Optional[float] = None
    market_news_volume: Optional[int] = None
    market_source_credibility: Optional[float] = None
    market_source_diversity: Optional[float] = None
    market_sentiment_regime: Optional[float] = None
    market_hours_sentiment: Optional[float] = None
    pre_market_sentiment: Optional[float] = None
    after_market_sentiment: Optional[float] = None