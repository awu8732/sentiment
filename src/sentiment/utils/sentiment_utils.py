import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import List, Dict, Optional, Union, Literal
import pytz
import logging
import warnings
from scipy.stats import skew as scipy_skew

logger = logging.getLogger(__name__)
warnings.filterwarnings("error")

class SentimentUtils:
    """Utility functions specifically for sentiment analysis"""

    @staticmethod
    def detect_sentiment_regime(sentiments: pd.Series, threshold_factor: float = 0.5) -> str:
        """Detect the sentiment regime: bullish, bearish, or neutral"""
        sentiments = sentiments.dropna()
        if sentiments.empty:
            return 'neutral'
        
        mean = sentiments.mean()
        std = sentiments.std(ddof=0)
        threshold = threshold_factor * std

        if mean > threshold:
            return 'bullish'
        elif mean < -threshold:
            return 'bearish'
        else:
            return 'neutral'
        
    @staticmethod
    def calculate_sentiment_momentum(sentiments: np.ndarray, decay_factor: float = 1.0) -> float:
        """Compute exponentially-weighted sentiment momentum"""
        if not isinstance(sentiments, np.ndarray):
            sentiments = np.array(sentiments)
        if len(sentiments) == 0:
            return np.nan
        
        weights = np.exp(np.linspace(-decay_factor, 0, num=len(sentiments)))
        if weights.sum() == 0:
            return np.nan
        weights /= weights.sum()
        return np.dot(weights, sentiments)
    
    @staticmethod
    def calculate_extreme_sentiment_ratio(sentiments: pd.Series, threshold: float = 0.5) -> float:
        """Compute the proportion of articles with oddly-strong sentiment"""
        sentiments = sentiments.dropna()
        if sentiments.empty:
            return np.nan
        
        extreme_count = (sentiments.abs() > threshold).sum()
        return extreme_count / len(sentiments)

    @staticmethod
    def calculate_sentiment_persistence(sentiments: pd.Series, lag: int = 5) -> float:
        """Estimate sentiment autocorrelation to capture persistence"""
        sentiments = sentiments.dropna()
        if len(sentiments) <= lag or sentiments.var() == 0:
            return np.nan

        try:
            return sentiments.autocorr(lag=lag)
        except Exception as e:
            logger.warning(f"Autocorrelation calculation failed: {e}")
            return np.nan