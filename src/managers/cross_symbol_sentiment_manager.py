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
from src.models.features.cross_symbol_features import CrossSymbolFeatures
from src.sentiment.feature_engineering import SentimentFeatureEngineer
from src.sentiment.utils import NewsUtils, TimeUtils, StatisticalUtils, SentimentUtils

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
        self.news_utils = NewsUtils()
        self.time_utils = TimeUtils()
        self.stats_utils = StatisticalUtils()
        self.sentiment_utils = SentimentUtils()

        # Check if analyzer initialized successfully
        if not self.feature_engineer.sentiment_analyzer.is_initialized:
            self.logger.error("Sentiment analyzer failed to initialize")
            raise RuntimeError("Cannot proceed without sentiment analyzer")
    
    def create_cross_symbol_features(self, 
                                     symbol: str,
                                     all_news_df: pd.DataFrame,
                                     window_hours: int = 24) -> Dict[str, int]:
        """Create market sentiment features over sliding hourly window for all available articles."""
        

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
