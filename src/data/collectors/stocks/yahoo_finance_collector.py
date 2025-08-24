import yfinance as yf
from typing import List, Dict
import logging

from src.models.stock import StockData
from ..base_collector import BaseCollector

logger = logging.getLogger(__name__)

class YahooFinanceCollector(BaseCollector):
    """Collects stock price data from Yahoo Finance"""
    
    def __init__(self, rate_limit: float = 0.1):
        super().__init__("YahooFinance", rate_limit)
    
    def collect_data(self, symbols: List[str], interval: str = "1h", period: str = "6mo") -> Dict[str, List[StockData]]:
        """Fetch stock data for multiple symbols"""
        stock_data = {}
        
        for symbol in symbols:
            self._rate_limit_wait()
            
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval, prepost=True)
                
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