import requests
from datetime import datetime, timedelta
from typing import List
import logging

from src.models.article import NewsArticle
from ..base_collector import BaseCollector

logger = logging.getLogger(__name__)

class AlphaVantageNewsCollector(BaseCollector):
    """Collects news from Alpha Vantage"""
    
    def __init__(self, api_key: str, rate_limit: float = 12.0):
        super().__init__("AlphaVantage", rate_limit)
        self.api_key = api_key
        self.base_url = 'https://www.alphavantage.co/query'
    
    def collect_data(self, symbol: str, days_back: int = 7) -> List[NewsArticle]:
        """Fetch news from Alpha Vantage"""
        if not self.api_key:
            logger.warning("Alpha Vantage key not provided")
            return []
        
        self._rate_limit_wait()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'time_from': start_date.strftime('%Y%m%dT%H%M'),
            'time_to': end_date.strftime('%Y%m%dT%H%M'),
            'apikey': self.api_key,
            'limit': 100
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage API limit reached: {data['Note']}")
                return []
            
            articles = []
            for item in data.get('feed', []):
                if item.get('title') and item.get('time_published'):
                    # Parse Alpha Vantage timestamp format
                    timestamp = datetime.strptime(item['time_published'], '%Y%m%dT%H%M%S')
                    
                    articles.append(NewsArticle(
                        timestamp=timestamp,
                        title=item['title'],
                        description=item.get('summary', ''),
                        source=item.get('source', 'Alpha Vantage'),
                        url=item.get('url', ''),
                        symbol=symbol
                    ))
            
            logger.info(f"Collected {len(articles)} articles for {symbol} from Alpha Vantage")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news for {symbol}: {e}")
            return []
