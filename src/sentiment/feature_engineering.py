import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .analyzers import EnsembleSentimentAnalyzer
from .utils import StatisticalUtils, TimeUtils, SentimentUtils, NewsUtils, CrossSymbolUtils
from ..models import NewsArticle, SentimentFeatures

logger = logging.getLogger(__name__)

class SentimentFeatureEngineer:
    """Implements feature engineering for financial news analysis.
    Creates features that capture sentiment patterns, momentum, and market dynamics.
    """

    def __init__(self):
        self.sentiment_analyzer = EnsembleSentimentAnalyzer()
        self.stats_utils = StatisticalUtils()
        self.time_utils = TimeUtils()
        self.sentiment_utils = SentimentUtils()
        self.news_utils = NewsUtils()
        self.cross_symbol_utils = CrossSymbolUtils()
        
    def analyze_article_batch_sentiment(self, articles_df: pd.DataFrame) -> List[dict]:
        """Analyze sentiment for a batch of articles"""
        results = []
        for _, row in articles_df.iterrows():
            try:
                article = NewsArticle(
                    timestamp=pd.to_datetime(row['timestamp']),
                    title=row['title'] or '',
                    description=row['description'] or '',
                    source=row['source'] or '',
                    url=row['url'] or '',
                    symbol=row['symbol']
                )

                sentiment = self.analyze_article_sentiment(article)
                results.append({
                    'id': row['id'],
                    'sentiment_score': sentiment['compound'],
                    'sentiment_positive': sentiment['positive'],
                    'sentiment_neutral': sentiment['neutral'],
                    'sentiment_negative': sentiment['negative']
                })
                
            except Exception as e:
                self.logger.error(f"Error analyzing article {row['id']}: {e}")
                results.append({
                    'id': row['id'],
                    'sentiment_score': 0.0,
                    'sentiment_positive': 0.0,
                    'sentiment_neutral': 1.0,
                    'sentiment_negative': 0.0
                })
        return results

    def analyze_article_sentiment(self, article: NewsArticle) -> Dict[str, float]:
        """Analyze sentiment of a single article using multiple methods"""
        text = f"{article.title} {article.description}".strip()
        return self.sentiment_analyzer.analyze_text(text)
