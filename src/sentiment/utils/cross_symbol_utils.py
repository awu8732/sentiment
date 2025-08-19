import numpy as np
from typing import List
import logging
from .statistical_utils import StatisticalUtils

logger = logging.getLogger(__name__)

class CrossSymbolUtils:
    """Utilities for analyzing sentiment relationships between a target symbol 
    and its accompanying sector or peer group"""

    @staticmethod
    def calculate_relative_sentiment(target_sentiments: List[float],
                                     sector_sentiments: List[float]) -> float:
        """Calculate the relative sentiment of target symbol vs sector/peers, returning
        the ratio between the mean target sentiment and mean sector sentiment"""

        if not target_sentiments or not sector_sentiments:
            return 0.0
        
        target_mean = np.mean(target_sentiments)
        sector_mean = np.mean(sector_sentiments)
        return target_mean / sector_mean if sector_mean != 0 else 0.0
    
    @staticmethod
    def calculate_sentiment_divergence(target_sentiments: List[float],
                                         sector_sentiments: List[float]) -> float:
        """Measure how differently the target sentiment varies compared to its sector, 
        returning absolute difference in standard deviations"""
        if not target_sentiments or not sector_sentiments:
            return 0.0
        
        target_std = np.std(target_sentiments, ddof=0)
        sector_std = np.std(sector_sentiments, ddof=0)
        return float(abs(target_std - sector_std))
    
    @staticmethod
    def calculate_sentiment_correlation(target_sentiments: List[float],
                                        sector_sentiments: List[float]) -> float:
        """Calculate the Pearson correlation coefficient between target and sector
        sentiments, using truncated matching based on shortest sequence length"""
        min_length = min(len(target_sentiments), len(sector_sentiments))
        if min_length < 2:
            return 0.0
        
        target_trimmed = target_sentiments[:min_length]
        sector_trimmed = sector_sentiments[:min_length]
        return float(StatisticalUtils.safe_correlation(target_trimmed, sector_trimmed))
        