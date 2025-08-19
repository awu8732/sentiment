import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from config.symbols import get_symbols_by_sector, get_symbol_sector, get_all_symbols
from src.data.database import DatabaseManager
from src.models.features.market_features import MarketFeatures
from src.sentiment.feature_engineering import SentimentFeatureEngineer
from src.sentiment.utils import NewsUtils, TimeUtils, StatisticalUtils, SentimentUtils

class MarketSentimentManager:
    """Handles market-wide sentiment analysis for cross-table reference"""

    def __init__(self, config: Config, 
                 logger: logging,
                 db_manager: DatabaseManager,
                 feature_engineer: SentimentFeatureEngineer):
        self.config = config
        self.logger = logger
        self.db_manager = db_manager
        self.feature_engineer = feature_engineer
        self.news_utils = NewsUtils()
        self.time_utils = TimeUtils()
        self.stats_utils = StatisticalUtils()
        self.sentiment_utils = SentimentUtils()

        # Check if analyzer initialized successfully
        if not self.feature_engineer.sentiment_analyzer.is_initialized:
            self.logger.error("Sentiment analyzer failed to initialize")
            raise RuntimeError("Cannot proceed without sentiment analyzer")
        
    def create_market_features(self,
                               lookback_date: Optional[datetime] = None, 
                               lookback_hours: Optional[int] = None,
                               window_size: int = 24):
        """Outward facing method to create market-features"""
        market_df = self._get_market_data(lookback_date, lookback_hours)
        self._create_market_features(market_df, window_size)

    def _create_market_features(self, 
                            all_news_df: pd.DataFrame,
                            window_size: int = 24) -> Dict[str, int]:
        """Create market sentiment features for given dataframe and hourly sliding window"""
        
        if all_news_df.empty:
            self.logger.warning("No analyzed articles found")
            return {'features_created': 0, 'features_updated': 0}

        self.logger.info(f"Creating market_features with {len(all_news_df)} articles...")

        # Convert naive timestamps to UTC
        all_news_df['timestamp'] = pd.to_datetime(
            all_news_df['timestamp'], 
            errors='coerce', 
            utc=True
        )
        # Set timestamp as index for easier slicing
        all_news_df = all_news_df.set_index('timestamp')

        # Generate hourly time range based on available data extrema
        start_time = all_news_df.index.min().floor('h')
        end_time = all_news_df.index.max().ceil('h')
        time_range = pd.date_range(start=start_time, end=end_time, freq='1h')

        features_list = []

        for timestamp in time_range:
            window_start = timestamp - pd.Timedelta(hours=window_size)
            window_articles = all_news_df[(all_news_df.index >= window_start) & (all_news_df.index < timestamp)]
            window_sentiments = window_articles['sentiment_score'].dropna().values

            market_feature = MarketFeatures(timestamp=timestamp)

            if len(window_articles) == 0 or len(window_sentiments) == 0:
                features_list.append(market_feature)
                continue

            market_feature.market_sentiment_mean = self.stats_utils.weighted_average(window_sentiments)
            market_feature.market_sentiment_std = self.stats_utils.safe_std(window_sentiments)
            market_feature.market_sentiment_skew = self.stats_utils.safe_skew(window_sentiments)
            market_feature.market_sentiment_momentum = self.sentiment_utils.calculate_sentiment_momentum(window_sentiments)
            market_feature.market_news_volume = len(window_articles)
            market_feature.market_source_credibility = self.news_utils.calculate_source_credibility(window_articles)
            market_feature.market_source_diversity = self.news_utils.calculate_source_diversity(window_articles)
            market_feature.market_sentiment_regime = self.sentiment_utils.detect_sentiment_regime(pd.Series(window_sentiments))
            
            # Calculate for market/after-market hours sentiment
            timestamps_eastern = window_articles.index.tz_convert('US/Eastern')
            window_articles = window_articles.assign(
                day_type = np.where(
                    timestamps_eastern.time < TimeUtils.MARKET_OPEN, "pre_market",
                    np.where(
                        timestamps_eastern.time < TimeUtils.MARKET_CLOSE, "market_hours", "after_hours"
                    )
                )
            )
            market_feature.market_hours_sentiment = window_articles.loc[
                window_articles['day_type'] == "market_hours", "sentiment_score"
            ].mean()
            market_feature.pre_market_sentiment = window_articles.loc[
                window_articles['day_type'] == "pre_market", "sentiment_score"
            ].mean()
            market_feature.after_market_sentiment = window_articles.loc[
                window_articles['day_type'] == "after_hours", "sentiment_score"
            ].mean()
            features_list.append(market_feature)

        # Attempt to insert new batch into db
        features_created = 0
        try:
            if features_list:
                inserted = self.db_manager.insert_market_features_batch(features_list)
                features_created += inserted
                self.logger.info(f"Created {inserted} new market sentiment features")
        except Exception as e:
            self.logger.error(f"Error inserting market features: {e}")

        return {
            'features_created': features_created,
            'features_updated': 0
        }

    def _get_market_data(self, 
                         lookback_date: Optional[datetime] = None, 
                         window_hours: Optional[int] = None) -> pd.DataFrame:
        """Get market news data needed, given a designated windwo"""
        now = datetime.now(timezone.utc)

        # Resolve input cases (no args => full history)
        if lookback_date is None and window_hours is None:
            start_time, end_time = None, None
        elif lookback_date is None and window_hours is not None:
            start_time = now - timedelta(hours=window_hours)
            end_time = now
        elif lookback_date is not None and window_hours is None:
            start_time = lookback_date
            end_time = now
        else:
            start_time = lookback_date
            end_time = lookback_date + timedelta(hours=window_hours)

        # Query db
        all_news_df = self.db_manager.get_news_data(
            start_date=start_time,
            end_date=end_time
        )

        if all_news_df is None or all_news_df.empty:
            return pd.DataFrame()
        return all_news_df[all_news_df['sentiment_score'].notna()].copy()
