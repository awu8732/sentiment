from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class NewsArticle:
    """Data structure for news articles"""
    timestamp: datetime
    title: str
    description: str
    source: str
    url: str
    symbol: str
    sentiment_score: Optional[float] = None
    
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
    
@dataclass
class SentimentFeatures:
    """Data structure for sentiment-derived features"""
    timestamp: datetime
    symbol: str
    sentiment_score: float
    sentiment_momentum: float
    news_volume: int
    source_diversity: float