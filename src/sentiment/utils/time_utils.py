import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Literal
import pytz
import logging
from scipy.stats import skew as scipy_skew

logger = logging.getLogger(__name__)

class TimeUtils:
    """Time-related utility functions for U.S. market hours (Eastern Time)"""

    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    EASTERN = pytz.timezone("US/Eastern")

    @staticmethod
    def to_eastern(timestamp: datetime) -> datetime:
        """Convert any timestamp to US Eastern Time (aware)"""
        if timestamp.tzinfo is None:
            logger.warning("Naive datetime passed to to_eastern(); assuming UTC.")
            timestamp = timestamp.replace(tzinfo=pytz.utc)
        return timestamp.astimezone(TimeUtils.EASTERN)

    @staticmethod
    def is_weekend(timestamp: datetime) -> bool:
        """Check if the given timestamp falls on a weekend (Saturday or Sunday)"""
        return TimeUtils.to_eastern(timestamp).weekday() >= 5

    @staticmethod
    def is_market_hours(timestamp: datetime) -> bool:
        """Check if timestamp is during regular U.S. market hours (Mon–Fri, 9:30 AM–4:00 PM ET)"""
        if TimeUtils.is_weekend(timestamp):
            return False
        t = TimeUtils.to_eastern(timestamp).time()
        return TimeUtils.MARKET_OPEN <= t < TimeUtils.MARKET_CLOSE

    @staticmethod
    def is_after_hours(timestamp: datetime) -> bool:
        """Check if timestamp is after market close or before market open on a weekday"""
        if TimeUtils.is_weekend(timestamp):
            return True
        t = TimeUtils.to_eastern(timestamp).time()
        return t < TimeUtils.MARKET_OPEN or t >= TimeUtils.MARKET_CLOSE

    @staticmethod
    def get_trading_day_type(timestamp: datetime) -> Literal['weekend', 'pre_market', 'market_hours', 'after_hours']:
        """Categorize timestamp as one of: weekend, pre_market, market_hours, after_hours"""
        # if TimeUtils.is_weekend(timestamp):
        #     return 'weekend'
        
        t = TimeUtils.to_eastern(timestamp).time()
        if t < TimeUtils.MARKET_OPEN:
            return 'pre_market'
        elif t < TimeUtils.MARKET_CLOSE:
            return 'market_hours'
        else:
            return 'after_hours'

