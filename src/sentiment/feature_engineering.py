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

    def create_time_based_features(self, articles_df: pd.DataFrame, 
                                 symbol: str, 
                                 time_windows: List[str] = ['1h', '4h', '24h']) -> pd.DataFrame:
        """Create time-based features with differing aggregation windows"""
        if articles_df.empty:
            return pd.DataFrame()
        
        # Ensure timestamp is datetime format
        articles_df['timestamp'] = pd.to_datetime(articles_df['timestamp'], errors='coerce')
        articles_df = articles_df.dropna(subset=['timestamp'])
        articles_df = articles_df.sort_values('timestamp')

        # Create a complete time range (rounded to the nearest hour)
        start_time = articles_df['timestamp'].min().floor('h')
        end_time = articles_df['timestamp'].max().ceil('h')
        time_range = pd.date_range(start=start_time, end=end_time, freq='h')

        features_list = []
        for timestamp in time_range:
            feature_row = {
                'timestamp': timestamp,
                'symbol': symbol
            }

            # Create features for each time window
            for window in time_windows:
                window_start = timestamp - pd.Timedelta(window)
                window_articles = articles_df[(articles_df['timestamp'] >= window_start) & (articles_df['timestamp'] < timestamp)]

                if len(window_articles) == 0:
                    # No articles in this window, set default values
                    feature_row.update(self._create_empty_window_features(window))
                    continue

                # Calculate sentiment for articles in window
                sentiments = self._get_sentiment_scores(window_articles, symbol)
                feature_row.update(self._calculate_window_features(window_articles, sentiments, window))

            features_list.append(feature_row)
        return pd.DataFrame(features_list)

    def create_advanced_features(self, articles_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create advanced sentiment features including:
            - Sentiment regime detection
            - News flow patterns
            - Source credibility weights
            - Market hours vs after-hours analysis
        """
        if articles_df.empty:
            return pd.DataFrame()
        
        articles_df['timestamp'] = pd.to_datetime(articles_df['timestamp'], errors='coerce')
        articles_df = articles_df.sort_values(by='timestamp').dropna(subset=['timestamp'])

        sentiment_data = self._prepare_sentiment_data(articles_df, symbol)
        sentiment_df = pd.DataFrame(sentiment_data)

        # Create hourly features
        hourly_features = []
        start_time = sentiment_df['timestamp'].min().floor('h')
        end_time = sentiment_df['timestamp'].max().ceil('h')
        time_range = pd.date_range(start=start_time, end=end_time, freq='h')

        for timestamp in time_range:
            # Get articles in the last 24 hours
            window_start = timestamp - pd.Timedelta('24h')
            window_data = sentiment_df[(sentiment_df['timestamp'] >= window_start) & (sentiment_df['timestamp'] < timestamp)]

            if len(window_data) == 0:
                continue

            features = self._calculate_advanced_features(timestamp, symbol, window_data)
            hourly_features.append(features)

        return pd.DataFrame(hourly_features)
    
    def create_cross_symbol_features(self, 
                                     all_articles_df: pd.DataFrame,
                                     target_symbol: str,
                                     sector_symbols: List[str]) -> pd.DataFrame:
        """Create features based on sentiment spillover effects from related symbols"""
        if all_articles_df.empty:
            return pd.DataFrame()
        
        all_articles_df['timestamp'] = pd.to_datetime(all_articles_df['timestamp'], errors='coerce')

        # Get sentiment for target symbol
        target_articles = all_articles_df[all_articles_df['symbol'] == target_symbol]
        if target_articles.empty:
            return pd.DataFrame()
        
        # Get sentiment for sector symbols
        sector_articles = all_articles_df[
            all_articles_df['symbol'].isin(sector_symbols) &
            (all_articles_df['symbol'] != target_symbol)
        ].copy()
            
        features_list = []
        start_time = target_articles['timestamp'].min().floor('h')
        end_time = target_articles['timestamp'].max().ceil('h')
        time_range = pd.date_range(start=start_time, end=end_time, freq='h')

        for timestamp in time_range:
            window_start = timestamp - pd.Timedelta('24h')

            # Target symbol & sector sentiment
            target_window = target_articles[
                (target_articles['timestamp'] >= window_start) &
                (target_articles['timestamp'] <= timestamp)
            ]
            sector_window = sector_articles[
                (sector_articles['timestamp'] >= window_start) &
                (sector_articles['timestamp'] <= timestamp)
            ]

            if len(target_window) == 0 or len(sector_window) == 0:
                continue

            target_sentiments = self._get_sentiment_scores(target_window, target_symbol)
            sector_sentiments = self._get_sentiment_scores(sector_window, None) if len(sector_window) > 0 else []
            features = self._calculate_cross_symbol_features(
                timestamp, target_symbol, target_sentiments, sector_sentiments, len(sector_window)
            )
            features_list.append(features)
            
        return pd.DataFrame(features_list)
    
    # Private helper methods that use the utility classes
    def _create_empty_window_features(self, window: str) -> Dict[str, float]:
        """Create empty features for window with no articles"""
        return {
            f'sentiment_mean_{window}': 0.0,
            f'sentiment_std_{window}': 0.0,
            f'sentiment_skew_{window}': 0.0,
            f'news_volume_{window}': 0,
            f'source_diversity_{window}': 0.0,
            f'sentiment_momentum_{window}': 0.0
        }
    
    def _get_sentiment_scores(self, articles_df: pd.DataFrame, symbol: Optional[str]) -> List[float]:
        """Extract sentiment scores from articles"""
        sentiments = []
        for _, article in articles_df.iterrows():
            article_obj = NewsArticle(
                timestamp=article['timestamp'],
                title=article.get('title', ''),
                description=article.get('description', ''),
                source=article.get('source', ''),
                url=article.get('url', ''),
                symbol=symbol or article.get('symbol', '')
            )
            sentiment = self.analyze_article_sentiment(article_obj)
            sentiments.append(sentiment['compound'])
        return sentiments
    
    def _calculate_window_features(self, window_articles: pd.DataFrame, 
                                 sentiments: List[float], window: str) -> Dict[str, float]:
        """Calculate features for a time window using utility classes"""
        sentiments_array = np.array(sentiments)
        
        return {
            f'sentiment_mean_{window}': np.mean(sentiments_array),
            f'sentiment_std_{window}': self.stats_utils.safe_std(sentiments_array),
            f'sentiment_skew_{window}': self.stats_utils.safe_skew(sentiments_array),
            f'news_volume_{window}': len(window_articles),
            f'source_diversity_{window}': self.news_utils.calculate_source_diversity(window_articles),
            f'sentiment_momentum_{window}': self.sentiment_utils.calculate_sentiment_momentum(sentiments_array)
        }
    
    
    def _prepare_sentiment_data(self, articles_df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Prepare sentiment data for advanced features"""
        sentiment_data = []
        for _, article in articles_df.iterrows():
            article_obj = NewsArticle(
                timestamp=article['timestamp'],
                title=article.get('title', ''),
                description=article.get('description', ''),
                source=article.get('source', ''),
                url=article.get('url', ''),
                symbol=symbol
            )
            sentiment = self.analyze_article_sentiment(article_obj)
            sentiment_data.append({
                'timestamp': article['timestamp'],
                'sentiment': sentiment['compound'],
                'source': article.get('source', ''),
                'is_market_hours': self.time_utils.is_market_hours(article['timestamp'])
            })
        return sentiment_data
    
    def _calculate_advanced_features(self, timestamp: datetime, symbol: str, 
                                   window_data: pd.DataFrame) -> Dict:
        """Calculate advanced features using utility classes"""
        market_hours_data = window_data[window_data['is_market_hours'] == True]
        after_hours_data = window_data[window_data['is_market_hours'] == False]
        
        return {
            'timestamp': timestamp,
            'symbol': symbol,
            'sentiment_regime': self.sentiment_utils.detect_sentiment_regime(window_data['sentiment']),
            'market_hours_sentiment': market_hours_data['sentiment'].mean() if len(market_hours_data) > 0 else 0.0,
            'after_hours_sentiment': after_hours_data['sentiment'].mean() if len(after_hours_data) > 0 else 0.0,
            'sentiment_volatility': self.stats_utils.safe_std(window_data['sentiment'].values),
            'news_flow_intensity': self.news_utils.calculate_news_flow_intensity(window_data),
            'source_credibility_score': self.news_utils.calculate_source_credibility(window_data),
            'sentiment_persistence': self.sentiment_utils.calculate_sentiment_persistence(window_data['sentiment']),
            'extreme_sentiment_ratio': self.sentiment_utils.calculate_extreme_sentiment_ratio(window_data['sentiment'])
        }
    
    def _calculate_cross_symbol_features(self, timestamp: datetime, target_symbol: str,
                                       target_sentiments: List[float], sector_sentiments: List[float],
                                       sector_news_volume: int) -> Dict:
        """Calculate cross-symbol features using utility classes"""
        return {
            'timestamp': timestamp,
            'symbol': target_symbol,
            'sector_sentiment_mean': np.mean(sector_sentiments) if sector_sentiments else 0.0,
            'sentiment_sector_correlation': self.cross_symbol_utils.calculate_sentiment_correlation(
                target_sentiments, sector_sentiments
            ),
            'relative_sentiment_strength': self.cross_symbol_utils.calculate_relative_sentiment(
                target_sentiments, sector_sentiments
            ),
            'sector_news_volume': sector_news_volume,
            'sentiment_divergence': self.cross_symbol_utils.calculate_sentiment_divergence(
                target_sentiments, sector_sentiments
            )
        }
    

    def update_sentiment_scores(self, results: List[dict]) -> int:
        """Update sentiment scores in database"""
        import sqlite3
        conn = sqlite3.connect(self.config.DATABASE_PATH)
        cursor = conn.cursor()
        
        updated_count = 0
        for result in results:
            try:
                cursor.execute("""
                    UPDATE news 
                    SET sentiment_score = ?,
                        created_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (result['sentiment_score'], result['id']))
                
                if cursor.rowcount > 0:
                    updated_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error updating article {result['id']}: {e}")
        
        conn.commit()
        conn.close()
        return updated_count

