from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class SentimentFeatures:
    """Data structure for sentiment-derived features"""
    timestamp: datetime
    symbol: str
    sentiment_score: float
    sentiment_skew: Optional[float] = None
    sentiment_std: Optional[float] = None
    sentiment_momentum: Optional[float] = None
    extreme_sentiment_ratio: Optional[float] = None
    sentiment_persistence: Optional[float] = None
    news_flow_intensity: Optional[float] = None
    news_volume: Optional[int] = None
    source_diversity: Optional[float] = None
