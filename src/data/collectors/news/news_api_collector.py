import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

from src.models.article import NewsArticle
from ..base_collector import BaseCollector
from config.symbols import get_company_name

logger = logging.getLogger(__name__)

class NewsAPICollector(BaseCollector):
    """Collects news from NewsAPI"""
    
    def __init__(self, api_key: str, rate_limit: float = 1.0):
        super().__init__("NewsAPI", rate_limit)
        self.api_key = api_key
        self.base_url = 'https://newsapi.org/v2/everything'
    
    def collect_data(self, symbol: str, days_back: int = 7) -> List[NewsArticle]:
        """Fetch articles from NewsAPI"""
        if not self.api_key:
            logger.warning("NewsAPI key not provided")
            return []
        
        self._rate_limit_wait()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        company_name = get_company_name(symbol)
        query = f'{symbol} OR "{company_name}"'
        
        params = {
            'q': query,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'apiKey': self.api_key,
            'language': 'en',
            'pageSize': 100
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                if article['title'] and article['publishedAt']:
                    # Filter out articles that don't actually mention the company
                    text_content = f"{article['title']} {article.get('description', '')}"
                    if symbol.lower() in text_content.lower() or company_name.lower() in text_content.lower():
                        articles.append(NewsArticle(
                            timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                            title=article['title'],
                            description=article.get('description', ''),
                            source=article['source']['name'],
                            url=article['url'],
                            symbol=symbol
                        ))
            
            logger.info(f"Collected {len(articles)} articles for {symbol} from NewsAPI")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI data for {symbol}: {e}")
            return []
