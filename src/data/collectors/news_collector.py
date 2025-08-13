import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

from src.models.article import NewsArticle
from .base_collector import BaseCollector
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

class RedditCollector(BaseCollector):
    """Collects sentiment from Reddit"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str, rate_limit: float = 1.0):
        super().__init__("Reddit", rate_limit)
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.reddit = None
        self._initialize_reddit()
    
    def _initialize_reddit(self):
        """Initialize Reddit API connection"""
        try:
            import praw
            if self.client_id and self.client_secret:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                logger.info("Reddit API initialized successfully")
        except ImportError:
            logger.warning("praw not installed. Install with: pip install praw")
        except Exception as e:
            logger.error(f"Error initializing Reddit API: {e}")
    
    def collect_data(self, symbol: str, days_back: int = 7, limit: int = 100) -> List[NewsArticle]:
        """Collect posts from Reddit about a symbol"""
        if not self.reddit:
            logger.warning("Reddit API not available")
            return []
        
        self._rate_limit_wait()
        
        try:
            articles = []
            subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'StockMarket', 'wallstreetbets']
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    company_name = get_company_name(symbol)
                    
                    # Search for posts mentioning the symbol or company
                    for submission in subreddit.search(f'{symbol} OR {company_name}', 
                                                     sort='new', 
                                                     time_filter='week', 
                                                     limit=limit//len(subreddits)):
                        
                        # Filter by date
                        post_date = datetime.fromtimestamp(submission.created_utc)
                        if (datetime.now() - post_date).days <= days_back:
                            articles.append(NewsArticle(
                                timestamp=post_date,
                                title=submission.title,
                                description=submission.selftext[:500] if submission.selftext else '',
                                source=f'Reddit r/{subreddit_name}',
                                url=f'https://reddit.com{submission.permalink}',
                                symbol=symbol
                            ))
                
                except Exception as e:
                    logger.error(f"Error collecting from r/{subreddit_name}: {e}")
                    continue
            
            logger.info(f"Collected {len(articles)} Reddit posts for {symbol}")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting Reddit data for {symbol}: {e}")
            return []