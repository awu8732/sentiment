import sqlite3
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import logging

from .models import NewsArticle, StockData, SentimentFeatures

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create news table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                title TEXT,
                description TEXT,
                source TEXT,
                url TEXT UNIQUE,
                symbol TEXT,
                sentiment_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create stock price table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp, symbol)
            )
        ''')
        
        # Create sentiment features table with cross-symbol features
        # Enhanced sentiment features table with cross-symbol features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                sentiment_score REAL,
                sentiment_momentum REAL,
                news_volume INTEGER,
                source_diversity REAL,
                
                -- Cross-symbol sentiment features
                sector_sentiment_mean REAL,
                market_sentiment_mean REAL,
                sentiment_sector_correlation REAL,
                sentiment_market_correlation REAL,
                relative_sentiment_strength REAL,
                sector_news_volume INTEGER,
                market_news_volume INTEGER,
                sentiment_divergence REAL,
                sector_sentiment_volatility REAL,
                market_sentiment_volatility REAL,
                
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp, symbol)
            )
        ''')
        
        # Create cross-symbol analysis cache table for performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_symbol_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                analysis_type TEXT, -- 'sector' or 'market'
                reference_group TEXT, -- sector name or 'market'
                sentiment_mean REAL,
                sentiment_volatility REAL,
                news_volume INTEGER,
                symbols_count INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp, analysis_type, reference_group)
            )
        ''')

        # Create indices for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_symbol_timestamp ON news(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_symbol_timestamp ON stock_prices(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_timestamp ON sentiment_features(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cross_symbol_cache ON cross_symbol_cache(timestamp, analysis_type, reference_group)')

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def insert_news_batch(self, articles: List[NewsArticle]):
        """Insert multiple news articles"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted = 0
        for article in articles:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO news 
                    (timestamp, title, description, source, url, symbol, sentiment_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (article.timestamp, article.title, article.description, 
                     article.source, article.url, article.symbol, article.sentiment_score))
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                logger.error(f"Error inserting article: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Inserted {inserted} new articles")
        return inserted
    
    def insert_stock_data_batch(self, stock_data: List[StockData]):
        """Insert multiple stock price records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted = 0
        for data in stock_data:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO stock_prices 
                    (timestamp, symbol, open, high, low, close, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data.timestamp, data.symbol, data.open, data.high, 
                     data.low, data.close, data.volume, data.adj_close))
                inserted += 1
            except Exception as e:
                logger.error(f"Error inserting stock data: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Inserted {inserted} stock price records")
        return inserted
    
    def insert_sentiment_features_batch(self, features: List[SentimentFeatures]):
        """Insert sentiment features iwth cross-symbol data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted = 0
        for feature in features:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO sentiment_features 
                    (timestamp, symbol, sentiment_score, sentiment_momentum, news_volume, source_diversity,
                     sector_sentiment_mean, market_sentiment_mean, sentiment_sector_correlation, 
                     sentiment_market_correlation, relative_sentiment_strength, sector_news_volume,
                     market_news_volume, sentiment_divergence, sector_sentiment_volatility,
                     market_sentiment_volatility)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    feature.timestamp, feature.symbol, feature.sentiment_score, 
                    feature.sentiment_momentum, feature.news_volume, feature.source_diversity,
                    getattr(feature, 'sector_sentiment_mean', None),
                    getattr(feature, 'market_sentiment_mean', None),
                    getattr(feature, 'sentiment_sector_correlation', None),
                    getattr(feature, 'sentiment_market_correlation', None),
                    getattr(feature, 'relative_sentiment_strength', None),
                    getattr(feature, 'sector_news_volume', None),
                    getattr(feature, 'market_news_volume', None),
                    getattr(feature, 'sentiment_divergence', None),
                    getattr(feature, 'sector_sentiment_volatility', None),
                    getattr(feature, 'market_sentiment_volatility', None)
                ))
                inserted += 1
            except Exception as e:
                logger.error(f"Error inserting sentiment features: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Inserted {inserted} sentiment feature records")
        return inserted
    
    def cache_cross_symbol_analysis(self, timestamp: datetime, analysis_type: str, 
                                    reference_group: str, sentiment_mean: float,
                                    sentiment_volatility: float, news_volume: int,
                                    symbols_count = int):
        """Cache cross-symbol analysis results for performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO cross_symbol_cache
                (timestamp, analysis_type, reference_group, sentiment_mean, 
                 sentiment_volatility, news_volume, symbols_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, analysis_type, reference_group, sentiment_mean,
                  sentiment_volatility, news_volume, symbols_count))
            conn.commit()
        except Exception as e:
            logger.error(f"Error caching cross-symbol analysis object: {e}")
        finally:
            conn.close()

    def get_cross_symbol_cache(self, timestamp: datetime, analysis_type: str,
                               reference_group: str) -> Optional[Dict]:
        """Retrieve cached cross-symbol analysis metrics"""
        conn = sqlite3.connect(self.db_path)

        try:
            query = '''
                SELECT sentiment_mean, sentiment_volatility, news_volume, symbols_count
                FROM cross_symbol_cache
                WHERE timestamp = ? AND analysis_type = ? AND reference_group = ?
            '''
            result = pd.read_sql_query(query, conn, params=[timestamp, analysis_type, reference_group])
            
            if not result.empty:
                return {
                    'sentiment_mean': result.iloc[0]['sentiment_mean'],
                    'sentiment_volatility': result.iloc[0]['sentiment_volatility'],
                    'news_volume': result.iloc[0]['news_volume'],
                    'symbols_count': result.iloc[0]['symbols_count']
                }
        except Exception as e:
            logger.error(f"Error retrieving cross-symbol cache data: {e}")
        finally:
            conn.close()

        return None

    def get_news_data(self, symbol: Optional[str] = None, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve news data as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM news WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    
    def get_stock_data(self, symbol: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve stock data as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM stock_prices WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    
    def get_data_summary(self) -> Dict:
        """Get summary of collected data"""
        conn = sqlite3.connect(self.db_path)
        
        # News summary
        news_summary = pd.read_sql_query('''
            SELECT symbol, COUNT(*) as article_count, 
                   MIN(timestamp) as earliest_article,
                   MAX(timestamp) as latest_article,
                   AVG(sentiment_score) as avg_sentiment
            FROM news 
            GROUP BY symbol
            ORDER BY article_count DESC
        ''', conn)
        
        # Stock data summary
        stock_summary = pd.read_sql_query('''
            SELECT symbol, COUNT(*) as price_points,
                   MIN(timestamp) as earliest_date,
                   MAX(timestamp) as latest_date,
                   AVG(close) as avg_price
            FROM stock_prices 
            GROUP BY symbol
            ORDER BY price_points DESC
        ''', conn)
        
        conn.close()
        
        return {
            'news_summary': news_summary,
            'stock_summary': stock_summary
        }
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Remove data older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
        
        cursor.execute('DELETE FROM news WHERE timestamp < ?', (cutoff_date,))
        news_deleted = cursor.rowcount
        
        cursor.execute('DELETE FROM sentiment_features WHERE timestamp < ?', (cutoff_date,))
        features_deleted = cursor.rowcount

        cursor.execute('DELETE FROM cross_symbol_cache WHERE timestamp < ?', (cutoff_date,))
        cache_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {news_deleted} old news records, {features_deleted} old feature records, and {cache_deleted} cache entries")
        return news_deleted, features_deleted