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