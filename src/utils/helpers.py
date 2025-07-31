import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_returns(prices: pd.Series, periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
    """Calculate returns for different periods"""
    returns_df = pd.DataFrame(index=prices.index)
    
    for period in periods:
        returns_df[f'return_{period}d'] = prices.pct_change(periods=period)
        returns_df[f'return_{period}d_forward'] = prices.pct_change(periods=-period)
    
    return returns_df

def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling volatility"""
    returns = prices.pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    return volatility

def align_timestamps(df1: pd.DataFrame, df2: pd.DataFrame, 
                    time_col1: str = 'timestamp', time_col2: str = 'timestamp',
                    tolerance: str = '1H') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align two dataframes by timestamp with tolerance"""
    
    # Convert to datetime if needed
    df1[time_col1] = pd.to_datetime(df1[time_col1])
    df2[time_col2] = pd.to_datetime(df2[time_col2])
    
    # Set as index for merging
    df1_indexed = df1.set_index(time_col1)
    df2_indexed = df2.set_index(time_col2)
    
    # Merge with tolerance
    merged = pd.merge_asof(
        df1_indexed.sort_index(),
        df2_indexed.sort_index(),
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta(tolerance),
        direction='nearest'
    )
    
    return merged.reset_index(), df2_indexed.reset_index()

def filter_business_hours(df: pd.DataFrame, time_col: str = 'timestamp',
                         start_hour: int = 9, end_hour: int = 16) -> pd.DataFrame:
    """Filter data to business hours only"""
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Filter by hour and weekday
    mask = (
        (df[time_col].dt.hour >= start_hour) & 
        (df[time_col].dt.hour < end_hour) &
        (df[time_col].dt.weekday < 5)  # Monday=0, Friday=4
    )
    
    return df[mask]

def detect_outliers(series: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """Detect outliers in a series"""
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

def create_lagged_features(df: pd.DataFrame, columns: List[str], 
                          lags: List[int]) -> pd.DataFrame:
    """Create lagged features for time series"""
    result_df = df.copy()
    
    for col in columns:
        for lag in lags:
            result_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return result_df

def validate_data_quality(df: pd.DataFrame, required_columns: List[str]) -> Dict:
    """Validate data quality and return report"""
    report = {
        'total_rows': len(df),
        'missing_data': {},
        'duplicate_rows': df.duplicated().sum(),
        'date_range': {},
        'issues': []
    }
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        report['issues'].append(f"Missing required columns: {missing_cols}")
    
    # Check missing data
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        report['missing_data'][col] = {
            'count': missing_count,
            'percentage': round(missing_pct, 2)
        }
        
        if missing_pct > 50:
            report['issues'].append(f"Column '{col}' has {missing_pct:.1f}% missing data")
    
    # Check date range if timestamp column exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        report['date_range'] = {
            'start': df['timestamp'].min().isoformat(),
            'end': df['timestamp'].max().isoformat(),
            'days': (df['timestamp'].max() - df['timestamp'].min()).days
        }
    
    return report
