import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..data.models import NewsArticle, SentimentFeatures

logger = logging.getLogger(__name__)

class SentimentFeatureEngineer:
    """Implements feature engineering for financial news analysis.
    Creates features that capture sentiment patterns, momentum, and market dynamics.
    """

    def __init__(self, use_finbert: bool = False):
        self.use_finbert = use_finbert
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.finbert_model = None

        if use_finbert:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                logger.info("FinBERT model loaded successfully.")
            except ImportError:
                logger.error("Transformers library is not installed. Falling back to VADER and TextBlob")
                self.use_finbert = False
            except Exception as e:
                logger.error(f"Error loading FinBERT model: {e}")
                self.use_finbert = False
        
    def analyze_article_sentiment(self, article: NewsArticle) -> Dict[str, float]:
        """Analyze sentiment of a single article using multiple methods"""
        text = f"{article.title} {article.description}".strip()
        if not text:
            return {'compound': 0.0, 'textblob': 0.0, 'finbert': 0.0}
        
        sentiment_scores = {}

        # VADER sentiment (specialized in social media & informal text)
        vader_scores = self.vader_analyzer.polarity_scores(text)
        sentiment_scores['compound'] = vader_scores['compound']
        sentiment_scores['positive'] = vader_scores['pos']
        sentiment_scores['negative'] = vader_scores['neg']
        sentiment_scores['neutral'] = vader_scores['neu']

        # Textblob sentiment (general-purpose)
        try: 
            blob = TextBlob(text)
            sentiment_scores['textblob'] = blob.sentiment.polarity
            sentiment_scores['subjectivity'] = blob.sentiment.subjectivity
        except:
            sentiment_scores['textblob'] = 0.0
            sentiment_scores['subjectivity'] = 0.0

        # FinBERT sentiment ( practiced for financial text)
        if self.use_finbert and self.finbert_model:
            try:
                inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                outputs = self.finbert_model(**inputs)
                predictions = outputs.logits.softmax(dim=1)

                # FinBERT outputs: positive, neutraul, negative
                finbert_score = predictions[0][2].item() - predictions[0][0].item()  # positive - negative
                sentiment_scores['finbert'] = finbert_score
            
            except:
                sentiment_scores['finbert'] = 0.0
        else:
            sentiment_scores['finbert'] = 0.0

        return sentiment_scores

    def create_time_based_features(self, articles_df: pd.DataFrame, 
                                 symbol: str, 
                                 time_windows: List[str] = ['1H', '4H', '1D']) -> pd.DataFrame:
        """Create time-based features with differing aggregation windows"""
        if articles_df.empty:
            return pd.DataFrame()
        
        # Ensure timestamp is datetime format
        articles_df['timestamp'] = pd.to_datetime(articles_df['timestamp'], errors='coerce')
        articles_df = articles_df.dropna(subset=['timestamp'])

        # Create a complete time range (rounded to the nearest hour)
        start_time = articles_df['timestamp'].min().floor('H')
        end_time = articles_df['timestamp'].max().ceil('H')
        time_range = pd.date_range(start=start_time, end=end_time, freq='H')

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
                    feature_row.update({
                        f'sentiment_mean_{window}': 0.0,
                        f'sentiment_std_{window}': 0.0,
                        f'sentiment_skew_{window}': 0.0,
                        f'news_volume_{window}': 0,
                        f'source_diversity_{window}': 0.0,
                        f'sentiment_momentum_{window}': 0.0
                    })
                    continue

                # Calculate sentiment for articles in window
                sentiments = []
                for _, article in window_articles.iterrows():
                    article_obj = NewsArticle(
                        timestamp=article['timestamp'],
                        title=article.get('title', ''),
                        description=article.get('description', ''),
                        source=article.get('source', ''),
                        url=article.get('url', ''),
                        symbol=symbol
                    )

                    sentiment = self.analyze_article_sentiment(article_obj)
                    sentiments.append(sentiment['compound'])

                sentiments = np.array(sentiments)
                feature_row.update({
                    f'sentiment_mean_{window}': np.mean(sentiments),
                    f'sentiment_std_{window}': np.std(sentiments) if len(sentiments) > 1 else 0.0,
                    f'sentiment_skew_{window}': self._safe_skew(sentiments),
                    f'news_volume_{window}': len(window_articles),
                    f'source_diversity_{window}': self._calculate_source_diversity(window_articles),
                    f'sentiment_momentum_{window}': self._calculate_sentiment_momentum(window_articles, sentiments)
                })

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

        # Analyze all articles first
        sentiment_data = []
        for __, article in articles_df.iterrows():
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
                'sentiment': article.get('compound', ''),
                'source': article.get('source', ''),
                'is_market_hours': self._is_market_hours(article['timestamp'])
            })
        sentiment_df = pd.DataFrame(sentiment_data)

        # Create hourly features
        hourly_features = []
        start_time = sentiment_df['timestamp'].min().floor('H')
        end_time = sentiment_df['timestamp'].max().ceil('H')
        time_range = pd.date_range(start=start_time, end=end_time, freq='H')

        for timestamp in time_range:
            # Get articles in the last 24 hours
            window_start = timestamp - pd.Timedelta('24H')
            window_data = sentiment_df[(sentiment_df['timestamp'] >= window_start) & (sentiment_df['timestamp'] < timestamp)]

            if len(window_data) == 0:
                continue

            features = {
                'timestamp': timestamp,
                'symbol': symbol
                #MORE LATER: Add more features here
            }
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
        start_time = target_articles['timestamp'].min().floor('H')
        end_time = target_articles['timestamp'].max().ceil('H')
        time_range = pd.date_range(start=start_time, end=end_time, freq='H')

        for timestamp in time_range:
            window_start = timestamp - pd.Timedelta('24H')

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

            target_sentiments = [
                self.analyze_article_sentiment(
                    NewsArticle(
                        timestamp=row['timestamp'],
                        title=row.get('title', ''),
                        description=row.get('description', ''),
                        source=row.get('source', ''),
                        url=row.get('url', ''),
                        symbol=target_symbol
                    )
                )['compound'] for _, row in target_window.iterrows()
            ]
            sector_sentiments = []
            if len(sector_window) > 0:
                for _, row in sector_window.iterrows():
                    sentiment = self.analyze_article_sentiment(
                        NewsArticle(
                            timestamp=row['timestamp'],
                            title=row.get('title', ''),
                            description=row.get('description', ''),
                            source=row.get('source', ''),
                            url=row.get('url', ''),
                            symbol=row['symbol']
                        )
                    )['compound']
                    sector_sentiments.append(sentiment)

            features = {
                'timestamp': timestamp,
                'symbol': target_symbol

                # MORE LATER: Add more features here
            }
            features_list.append(features)
            
        return pd.DataFrame(features_list)


