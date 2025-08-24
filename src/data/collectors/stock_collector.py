import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

from src.models.stock import StockData
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

class AlphaVantageStockCollector(BaseCollector):
    """Collects stock price data from Alpha Vantage"""
    
    def __init__(self, api_key: str, rate_limit: float = 12.0):
        super().__init__("AlphaVantage", rate_limit)
        self.api_key = api_key
        self.base_url = 'https://www.alphavantage.co/query'
    
    def collect_data(self, symbols: List[str], interval: str = "60min", 
                    period: str = "1mo") -> Dict[str, List[StockData]]:
        """
        Fetch intraday stock data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            interval: Data interval ('15min', '30min', '60min')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y')
        """
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided")
            return {}
        
        # Validate interval
        if interval not in ['15min', '30min', '60min']:
            logger.error(f"Invalid interval: {interval}. Must be '15min', '30min', or '60min'")
            return {}
        
        # Parse period to get date range
        start_date, end_date = self._parse_period(period)
        
        stock_data = {}
        
        for symbol in symbols:
            self._rate_limit_wait()
            
            try:
                data_points = self._fetch_symbol_data(symbol, interval, start_date, end_date)
                stock_data[symbol] = data_points
                logger.info(f"Collected {len(data_points)} price points for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching stock data for {symbol}: {e}")
                stock_data[symbol] = []
        
        return stock_data
    
    def _fetch_symbol_data(self, symbol: str, interval: str, 
                          start_date: datetime, end_date: datetime) -> List[StockData]:
        """Fetch data for a single symbol using monthly API calls"""
        # Generate list of months to fetch
        months_to_fetch = self._get_months_in_range(start_date, end_date)
        
        all_data_points = []
        
        for month_str in months_to_fetch:
            self._rate_limit_wait()
            
            try:
                month_data = self._fetch_month_data(symbol, interval, month_str)
                all_data_points.extend(month_data)
                logger.info(f"API call for {symbol} month {month_str}: {len(month_data)} data points")
                
            except Exception as e:
                logger.error(f"Error fetching {symbol} data for month {month_str}: {e}")
                continue
        
        # Filter by actual date range and sort
        filtered_data = []
        for data_point in all_data_points:
            if start_date <= data_point.timestamp <= end_date:
                filtered_data.append(data_point)
        
        filtered_data.sort(key=lambda x: x.timestamp)
        logger.info(f"Total API calls for {symbol}: {len(months_to_fetch)}, "
                   f"filtered to {len(filtered_data)} data points")
        
        return filtered_data
    
    def _get_months_in_range(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Generate list of YYYY-MM strings for months in date range"""
        months = []
        current = start_date.replace(day=1)  # Start at beginning of month
        
        while current <= end_date:
            months.append(current.strftime('%Y-%m'))
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return months
    
    def _fetch_month_data(self, symbol: str, interval: str, month: str) -> List[StockData]:
        """Fetch data for a single symbol and month"""
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'month': month,
            'apikey': self.api_key,
            'outputsize': 'full',
            'datatype': 'json',
            'extended_hours': 'true'
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            raise Exception(f"API Error: {data['Error Message']}")
        
        if 'Note' in data:
            raise Exception(f"API Rate Limit: {data['Note']}")
        
        # Extract time series data
        time_series_key = f'Time Series ({interval})'
        if time_series_key not in data:
            logger.warning(f"No time series data found for {symbol} in {month}")
            return []
        
        time_series = data[time_series_key]
        data_points = []
        
        for timestamp_str, values in time_series.items():
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            try:
                data_point = StockData(
                    timestamp=timestamp,
                    symbol=symbol,
                    open=float(values['1. open']),
                    high=float(values['2. high']),
                    low=float(values['3. low']),
                    close=float(values['4. close']),
                    volume=int(values['5. volume']),
                    adj_close=float(values['4. close'])  # Alpha Vantage doesn't provide adj_close for intraday
                )
                data_points.append(data_point)
                
            except (KeyError, ValueError) as e:
                logger.warning(f"Error parsing data point for {symbol} at {timestamp_str}: {e}")
                continue
        
        return data_points
    
    def get_latest_price(self, symbol: str) -> Dict:
        """Get latest price information for a symbol"""
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided")
            return {}
        
        self._rate_limit_wait()
        
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                raise Exception(f"API Error: {data['Error Message']}")
            
            if 'Global Quote' not in data:
                logger.warning(f"No quote data found for {symbol}")
                return {}
            
            quote = data['Global Quote']
            
            return {
                'symbol': symbol,
                'current_price': float(quote.get('05. price', 0)),
                'previous_close': float(quote.get('08. previous close', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': quote.get('10. change percent', '0%'),
                'volume': int(quote.get('06. volume', 0)),
                'latest_trading_day': quote.get('07. latest trading day', '')
            }
            
        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            return {}
    
    def get_intraday_data(self, symbol: str, interval: str = "60min", 
                         period: str = "1d") -> List[StockData]:
        """
        Get recent intraday data for a symbol
        
        Args:
            symbol: Stock symbol
            interval: Data interval ('15min', '30min', '60min')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y')
        """
        data = self.collect_data([symbol], interval, period)
        return data.get(symbol, [])
    
    def _parse_period(self, period: str) -> tuple[datetime, datetime]:
        """Parse period string to start and end dates"""
        end_date = datetime.now()
        
        period_mapping = {
            '1d': 1,
            '2d': 2,
            '5d': 5,
            '10d': 10,
            '15d': 15,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825
        }
        
        if period not in period_mapping:
            logger.warning(f"Invalid period: {period}. Using default '1mo'")
            period = '1mo'
        
        days_back = period_mapping[period]
        start_date = end_date - timedelta(days=days_back)
        
        return start_date, end_date

class YahooFinanceCollector(BaseCollector):
    """Collects stock price data from Yahoo Finance"""
    
    def __init__(self, rate_limit: float = 0.1):
        super().__init__("YahooFinance", rate_limit)
    
    def collect_data(self, symbols: List[str], period: str = "6mo") -> Dict[str, List[StockData]]:
        """Fetch stock data for multiple symbols"""
        stock_data = {}
        
        for symbol in symbols:
            self._rate_limit_wait()
            
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if hist.empty:
                    logger.warning(f"No data returned for {symbol}")
                    stock_data[symbol] = []
                    continue
                
                data_points = []
                for date, row in hist.iterrows():
                    data_points.append(StockData(
                        timestamp=date.to_pydatetime().replace(tzinfo=None),
                        symbol=symbol,
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=int(row['Volume']),
                        adj_close=float(row['Close'])  # yfinance auto-adjusts
                    ))
                
                stock_data[symbol] = data_points
                logger.info(f"Collected {len(data_points)} price points for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching stock data for {symbol}: {e}")
                stock_data[symbol] = []
        
        return stock_data
    
    def get_latest_price(self, symbol: str) -> Dict:
        """Get latest price information for a symbol"""
        self._rate_limit_wait()
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'market_cap': info.get('marketCap', 0),
                'volume': info.get('volume', 0),
                'pe_ratio': info.get('trailingPE', 0)
            }
            
        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            return {}
    
    def get_intraday_data(self, symbol: str, interval: str = "1m", period: str = "1d") -> List[StockData]:
        """Get intraday data for a symbol"""
        self._rate_limit_wait()
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            data_points = []
            for date, row in hist.iterrows():
                data_points.append(StockData(
                    timestamp=date.to_pydatetime().replace(tzinfo=None),
                    symbol=symbol,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                    adj_close=float(row['Close'])
                ))
            
            logger.info(f"Collected {len(data_points)} intraday points for {symbol}")
            return data_points
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return []