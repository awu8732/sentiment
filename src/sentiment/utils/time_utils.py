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
    def is_market_hours(timestamp: datetime) -> bool:
        """Check if timestamp is during regular U.S. market hours (Mon–Fri, 9:30 AM–4:00 PM ET)"""
        ts_eastern = TimeUtils.to_eastern(timestamp)
        if ts_eastern.weekday() >= 5:
            return False
        return TimeUtils.MARKET_OPEN <= ts_eastern.time() < TimeUtils.MARKET_CLOSE

    @staticmethod
    def is_after_hours(timestamp: datetime) -> bool:
        """Check if timestamp is after regular market hours (including pre-market)"""
        ts_eastern = TimeUtils.to_eastern(timestamp)
        if ts_eastern.weekday() >= 5:
            return True
        t = ts_eastern.time()
        return t < TimeUtils.MARKET_OPEN or t >= TimeUtils.MARKET_CLOSE

    @staticmethod
    def get_trading_day_type(timestamp: datetime) -> Literal['weekend', 'pre_market', 'market_hours', 'after_hours']:
        """Categorize timestamp as one of: weekend, pre_market, market_hours, after_hours"""
        ts_eastern = TimeUtils.to_eastern(timestamp)
        t = ts_eastern.time()

        if ts_eastern.weekday() >= 5:
            return 'weekend'
        elif t < TimeUtils.MARKET_OPEN:
            return 'pre_market'
        elif TimeUtils.MARKET_OPEN <= t < TimeUtils.MARKET_CLOSE:
            return 'market_hours'
        else:
            return 'after_hours'
