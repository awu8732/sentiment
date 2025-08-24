from datetime import datetime
from typing import List
import logging

from src.models.article import NewsArticle
from ..base_collector import BaseCollector
from config.symbols import get_company_name

logger = logging.getLogger(__name__)

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