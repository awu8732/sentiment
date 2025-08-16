import pandas as pd
import numpy as np
from typing import List, Optional, Union
import pytz
import logging
from scipy.stats import skew as scipy_skew

logger = logging.getLogger(__name__)

class StatisticalUtils:
    """Utility class for statistical calculation"""

    @staticmethod
    def safe_skew(values: np.ndarray) -> float:
        """Calculate skewness safely, handling NaN values"""
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if len(values) == 0 or np.all(np.isnan(values)):
            return np.nan
        
        try:
            return scipy_skew(values, nan_policy='omit')
        except Exception as e:
            logger.error(f"Error calculating skewness: {e}")
            return np.nan
        
    @staticmethod
    def safe_correlation(x: Union[List[float], np.ndarray],
                          y: Union[List[float], np.ndarray]) -> float:
        """Calculate correlation safely, handling NaN values and shape mismatches"""
        try:
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)

            if x.shape != y.shape or x.size < 2:
                return np.nan
            
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            if np.sum(valid_mask) < 2:
                return np.nan
            return np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return np.nan
        
    @staticmethod
    def safe_std(values: Union[List[float], np.ndarray]) -> float:
        """Safely calculate standard deviation, ignoring NaNs"""
        try:
            values = np.asarray(values, dtype=np.float64)
            if values.size <= 1 or np.all(np.isnan(values)):
                return np.nan
            return np.nanstd(values)
        except Exception as e:
            logger.error(f"Error calculating standard deviation: {e}")
            return np.nan
        
    @staticmethod
    def safe_cast(value, dtype=float):
        """
        Safely cast a value to float or int.
        Returns np.nan if the value is None or cannot be converted.
        """
        if value is None:
            return np.nan
        try:
            return dtype(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to cast value {value} to {dtype}: {e}")
            return np.nan

    @staticmethod
    def safe_float(value):
        """Convenience wrapper for float"""
        return StatisticalUtils.safe_cast(value, float)

    @staticmethod
    def safe_int(value):
        """Convenience wrapper for int"""
        return StatisticalUtils.safe_cast(value, int)
        
    @staticmethod
    def weighted_average(values: Union[List[float], np.ndarray],
                        weights: Optional[Union[List[float], np.ndarray]] = None) -> float:
        """Safely calculate weighted average, handling NaNs and shape mismatches"""
        try:
            values = np.asarray(values, dtype=np.float64)
            if values.size == 0 or np.all(np.isnan(values)):
                return np.nan
            
            if weights is None:
                return np.nanmean(values)
            weights = np.asarray(weights, dtype=np.float64)

            if weights.shape != values.shape:
                logger.warning("Weights and values length mismatch; falling back to unweighted average.")
                return np.nanmean(values)

            mask = ~np.isnan(values) & ~np.isnan(weights)
            if np.sum(mask) == 0:
                return np.nan

            return np.average(values[mask], weights=weights[mask])
        except Exception as e:
            logger.error(f"Error calculating weighted average: {e}")
            return np.nan

