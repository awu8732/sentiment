from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class SentimentFeatures:
    """Data structure for sentiment-derived features"""
    timestamp: datetime
    symbol: str
    sentiment_score: float
    sentiment_momentum: float
    news_volume: int
    source_diversity: float
    
    # Cross-symbol sentiment features
    sector_sentiment_mean: Optional[float] = None
    market_sentiment_mean: Optional[float] = None
    sentiment_sector_correlation: Optional[float] = None
    sentiment_market_correlation: Optional[float] = None
    relative_sentiment_strength: Optional[float] = None
    sector_news_volume: Optional[int] = None
    market_news_volume: Optional[int] = None
    sentiment_divergence: Optional[float] = None
    sector_sentiment_volatility: Optional[float] = None
    market_sentiment_volatility: Optional[float] = None