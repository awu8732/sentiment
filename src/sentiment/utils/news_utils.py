import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("error")

class NewsUtils:
    """Utility functions for analyzing news article data"""

    @staticmethod
    def calculate_source_diversity(articles_df: pd.DataFrame, source_col: str = 'source') -> float:
        """Calculate the diversity of news sources using Shannon entropy.
        (Returns 0.0 if data is empty or only one source exists)"""
        if articles_df.empty or source_col not in articles_df.columns:
            return 0.0
        
        try:
            source_col = articles_df[source_col].value_counts(normalize=True)
            entropy = -np.sum([p * np.log2(p) for p in source_col if p > 0])
            return entropy
        except Exception as e:
            logger.error(f"Error calculating source diversity: {e}")
            return 0.0
        
    @staticmethod
    def calculate_news_flow_intensity(window_data: pd.DataFrame, time_col: str = 'timestamp') -> float:
        """Compute news flow intensity (articles per hour) over given time window.
        (Use minimum duration of 1 hour to avoid div by zero)"""
        if window_data.empty or time_col not in window_data.columns:
            return 0.0
        
        try:
            time_span_sec = (window_data[time_col].max() - window_data[time_col].min()).total_seconds()
            time_span_hours = max(time_span_sec / 3600, 1e-6)
            return len(window_data) / time_span_hours
        except Exception as e:
            logger.error(f"Error calculating news flow intensity: {e}")
            return 0.0
        
    @staticmethod
    def calculate_source_credibility(articles_df: pd.DataFrame, 
                                     source_col: str = 'source', 
                                     credibility_weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted source credibility score across articles
        (Unknown sources default to a weight 0.4)"""
        if articles_df.empty or source_col not in articles_df.columns:
            return 0.0
        
        # Default credibility weights
        default_weights = {
            'Reuters': 1.0,
            'Bloomberg': 1.0,
            'Wall Street Journal': 0.95,
            'Financial Times': 0.95,
            'Associated Press': 0.9,
            'CNBC': 0.8,
            'MarketWatch': 0.7,
            'Yahoo Finance': 0.6,
            'Seeking Alpha': 0.5
        }
        weights = credibility_weights or default_weights
        default_weights = 0.4

        try:
            source_counts = articles_df[source_col].value_counts(normalize=True)
            total_articles = source_counts.sum()

            weighted_score = 0.0
            max_score = 0.0
            for source, count in source_counts.items():
                weight = weights.get(source, default_weights)
                weighted_score += weight * count
                max_score += 1.0 * count
            
            return weighted_score / max_score if max_score > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating source credibility: {e}")
            return 0.0

