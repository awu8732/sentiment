import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from config.symbols import get_symbols_by_sector, get_symbol_sector
from src.data.database import DatabaseManager
from src.models.features.cross_symbol_features import CrossSymbolFeatures
from src.sentiment.feature_engineering import SentimentFeatureEngineer
from src.sentiment.utils import CrossSymbolUtils, StatisticalUtils, SentimentUtils

class CrossSymbolSentimentManager:
    """Handles cross-symbol sentiment analyssi against market & sector-wide data"""

    def __init__(self, config: Config, 
                 logger: logging,
                 db_manager: DatabaseManager,
                 feature_engineer: SentimentFeatureEngineer):
        self.config = config
        self.logger = logger
        self.db_manager = db_manager
        self.feature_engineer = feature_engineer
        self.cross_utils = CrossSymbolUtils()
        self.stats_utils = StatisticalUtils()
        self.sentiment_utils = SentimentUtils()

        # Check if analyzer initialized successfully
        if not self.feature_engineer.sentiment_analyzer.is_initialized:
            self.logger.error("Sentiment analyzer failed to initialize")
            raise RuntimeError("Cannot proceed without sentiment analyzer")
    
    def _create_cross_sector_features(self, 
                                     symbol: str,
                                     symbol_df: pd.DataFrame,
                                     sector_df: pd.DataFrame,
                                     window_size: int = 24) -> Dict[str, int]:
        """Create cross-sector sentiment features over sliding hourly window for all available articles."""
        if symbol_df.empty or sector_df.empty:
            self.logger.warning("Either passed symbol or sector sentiment features are empty")
            return {'features_created': 0, 'features_updated': 0}
        
        self.logger.info(f"Creating cross-sector features with {len(sector_df)} sector articles and {len(symbol_df)} articles for {symbol}")
        # Convert naive timestamps to UTC
        symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'], errors='coerce', utc=True)
        symbol_df = symbol_df.set_index('timestamp')
        sector_df['timestamp'] = pd.to_datetime(sector_df['timestamp'], errors='coerce', utc=True)
        sector_df = sector_df.set_index('timestamp')

        # Generate hourly time range based on available data extrema
        start_time = max(symbol_df.index.min().floor('h'), sector_df.index.min().floor('h'))
        end_time = min(symbol_df.index.max().floor('h'), sector_df.index.max().floor('h'))
        time_range = pd.date_range(start=start_time, end=end_time, freq='1h')

        features_list = []
        
        for timestamp in time_range:
            window_start = timestamp - pd.Timedelta(hours=window_size)
            window_symbol_articles = symbol_df[(symbol_df.index >= window_start) & (symbol_df.index < timestamp)]
            window_sector_articles = sector_df[(sector_df.index >= window_start) & (sector_df.index < timestamp)]
            window_symbol_sentiments = window_symbol_articles['sentiment_score'].dropna().tolist()
            window_sector_sentiments = window_sector_articles['sentiment_score'].dropna().tolist()

            cross_symbol_feature = CrossSymbolFeatures(
                timestamp=timestamp, 
                symbol=symbol,
                sector=get_symbol_sector(symbol))

            if len(window_sector_sentiments) == 0 or len(window_symbol_sentiments) == 0 :
                features_list.append(cross_symbol_feature)
                continue

            cross_symbol_feature.sector_sentiment_mean = self.stats_utils.weighted_average(window_sector_sentiments)
            cross_symbol_feature.sector_sentiment_std = self.stats_utils.safe_std(window_sector_sentiments)
            cross_symbol_feature.sector_sentiment_skew = self.stats_utils.safe_skew(window_sector_sentiments)
            cross_symbol_feature.sector_news_volume = len(window_sector_articles)
            cross_symbol_feature.relative_sentiment_ratio = self.cross_utils.calculate_relative_sentiment(window_symbol_sentiments, window_sector_sentiments)
            cross_symbol_feature.sector_sentiment_correlation = self.cross_utils.calculate_sentiment_correlation(window_symbol_sentiments, window_sector_sentiments)
            cross_symbol_feature.sector_sentiment_divergence = self.cross_utils.calculate_sentiment_divergence(window_symbol_sentiments, window_sector_sentiments)
            # find a place to insert market feature comparison

            features_list.append(cross_symbol_feature)

        # Attempt to insert new batch into db
        features_created = 0
        try:
            if features_list:
                inserted = self.db_manager.insert_cross_symbol_features_batch(features_list)
                features_created += inserted
                self.logger.info(f"Created {inserted} new cross-symbol sentiment features")
        except Exception as e:
            self.logger.error(f"Error inserting market features: {e}")

        return {
            'features_created': features_created,
            'features_updated': 0
        }
    
    def _get_symbol_sentiment_data(self, symbol: str,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Queue existing sentiment data for a specific article"""
        symbol_sentiment_df = self.db_manager.get_news_data(symbol, start_date, end_date)
        if symbol_sentiment_df.empty:
            return pd.DataFrame
        else:
            return symbol_sentiment_df[symbol_sentiment_df['sentiment_score'].notna()].copy()

    def _get_sector_data(self, 
                        symbol: str,
                        lookback_date: Optional[datetime] = None, 
                        window_hours: Optional[int] = None
                        ) -> pd.DataFrame:
        """Get sector data needed for cross-symbol analysis, given a symbol and designated window"""
        now = datetime.now(timezone.utc)

        # Get sector and associated symbols
        sector = get_symbol_sector(symbol)
        sector_symbols = get_symbols_by_sector(sector)
        if not sector_symbols:
            return pd.DataFrame()

        # Resolve time window (no args => full history)
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

        # Accumulate news for each symbol in the sector
        sector_dfs = []
        for symbol in sector_symbols:
            df = self.db_manager.get_news_data(
                symbol=symbol,
                start_date=start_time,
                end_date=end_time,
            )
            if df is not None and not df.empty:
                sector_dfs.append(df[df['sentiment_score'].notna()].copy())

        if not sector_dfs:
            return pd.DataFrame()

        # Stitch together into one DataFrame
        sector_df = pd.concat(sector_dfs, ignore_index=True)
        return sector_df
