import time
from typing import List, Dict
import logging
from datetime import datetime

from .database import DatabaseManager
from .collectors import YahooFinanceCollector, AlphaVantageStockCollector, NewsAPICollector, AlphaVantageNewsCollector, RedditCollector
from config.config import Config

logger = logging.getLogger(__name__)

class DataPipeline:
    """Main orchestrator for data collection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = DatabaseManager(config.DATABASE_PATH)
        
        # Initialize collectors
        self.news_collectors = []
        if config.API_KEYS.get('newsapi'):
            self.news_collectors.append(
                NewsAPICollector(config.API_KEYS['newsapi'], config.RATE_LIMITS['newsapi'])
            )
        
        if config.API_KEYS.get('alpha_vantage'):
            self.news_collectors.append(
                AlphaVantageNewsCollector(config.API_KEYS['alpha_vantage'], config.RATE_LIMITS['alpha_vantage'])
            )
        
        if all(config.API_KEYS.get(key) for key in ['reddit_client_id', 'reddit_client_secret']):
            self.news_collectors.append(
                RedditCollector(
                    config.API_KEYS['reddit_client_id'],
                    config.API_KEYS['reddit_client_secret'],
                    config.API_KEYS['reddit_user_agent'],
                    config.RATE_LIMITS['reddit']
                )
            )
        
        self.stock_collector = YahooFinanceCollector(config.RATE_LIMITS['yfinance'])
    
    def collect_all_data(self, symbols: List[str], days_back: int = None):
        """Collect both news and stock data for given symbols"""
        days_back = days_back or self.config.DEFAULT_LOOKBACK_DAYS
        logger.info(f"Starting data collection for {len(symbols)} symbols")
        
        # Collect stock data first
        logger.info(f"Collecting stock price data from a period of {self.config.STOCK_PERIOD}...")
        stock_data = self.stock_collector.collect_data(symbols, self.config.STOCK_INTERVAL, self.config.STOCK_PERIOD)
        
        # Insert stock data
        total_stock_records = 0
        for symbol, data_points in stock_data.items():
            if data_points:
                inserted = self.db_manager.insert_stock_data_batch(data_points)
                total_stock_records += inserted
        
        logger.info(f"Inserted {total_stock_records} total stock price records")
        
        # Collect news data
        logger.info(f"Collecting news data from {days_back} days back...")
        total_articles = 0
        
        for symbol in symbols:
            symbol_articles = []

            # Collect from all available news sources
            for collector in self.news_collectors:
                try:
                    articles = collector.collect_data(symbol, days_back)
                    symbol_articles.extend(articles)
                    logger.info(f"Collected {len(articles)} articles for {symbol} from {collector.name}")
                except Exception as e:
                    logger.error(f"Error collecting from {collector.name} for {symbol}: {e}")
            
            # Insert articles for this symbol
            if symbol_articles:
                inserted = self.db_manager.insert_news_batch(symbol_articles)
                total_articles += inserted
        
        logger.info(f"Data collection completed. Total articles: {total_articles}, Total stock records: {total_stock_records}")
        
        return {
            'articles_collected': total_articles,
            'stock_records_collected': total_stock_records,
            'symbols_processed': len(symbols)
        }
    
    def collect_incremental_data(self, symbols: List[str], hours_back: int = 1):
        """Collect only recent data for updates"""
        logger.info(f"Collecting incremental data for last {hours_back} hours")
        total_articles = 0
        
        for symbol in symbols:
            symbol_articles = []
            
            # Only collect from fast-updating sources for incremental
            for collector in self.news_collectors:
                if collector.name in ['NewsAPI', 'Reddit']:  # Skip Alpha Vantage for incremental due to rate limits
                    try:
                        articles = collector.collect_data(symbol, days_back=max(1, hours_back // 24))
                        recent_articles = []
                        for article in articles:
                            try:
                                # Handle both offset-aware and offset-naive timestamps
                                if article.timestamp.tzinfo is not None:
                                    current_time = datetime.now(article.timestamp.tzinfo)
                                else:
                                    current_time = datetime.now()
                                
                                if (current_time - article.timestamp).total_seconds() < hours_back * 3600:
                                    recent_articles.append(article)
                            except (AttributeError, TypeError) as e:
                                logger.warning(f"Error processing timestamp for article: {e}")
                                continue
                        
                        symbol_articles.extend(recent_articles)
                    except Exception as e:
                        logger.error(f"Error in incremental collection from {collector.name}: {e}")
            
            if symbol_articles:
                inserted = self.db_manager.insert_news_batch(symbol_articles)
                total_articles += inserted
        
        logger.info(f"Incremental collection completed. New articles: {total_articles}")
        return total_articles
    
    def get_pipeline_status(self) -> Dict:
        """Get status and summary of data pipeline"""
        summary = self.db_manager.get_data_summary()
        
        status = {
            'database_path': self.config.DATABASE_PATH,
            'active_collectors': [collector.name for collector in self.news_collectors],
            'data_summary': summary,
            'last_update': datetime.now().isoformat()
        }
        
        return status