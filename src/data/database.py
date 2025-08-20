import sqlite3
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import logging

from ..models import NewsArticle, StockData, SentimentFeatures, MarketFeatures, CrossSymbolFeatures
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
        
        # Create sentiment features table matching the SentimentFeatures dataclass
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                sentiment_score REAL,
                sentiment_skew REAL,
                sentiment_std REAL,
                sentiment_momentum REAL,
                extreme_sentiment_ratio REAL,
                sentiment_persistence REAL,
                news_flow_intensity REAL,
                news_volume INTEGER,
                source_diversity REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp, symbol)
            )
        ''')
        
        # Create market features table for market-wide features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                market_sentiment_mean REAL,
                market_sentiment_skew REAL,
                market_sentiment_std REAL,
                market_sentiment_momentum REAL,
                market_news_volume INTEGER,
                market_source_credibility REAL,
                market_source_diversity REAL,
                market_sentiment_regime REAL,
                market_hours_sentiment REAL,
                pre_market_sentiment REAL,
                after_market_sentiment REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp)
            )
        ''')
        
        # Create cross-symbol features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_symbol_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                sector TEXT,
                
                -- Comparison against sector-wide data
                sector_sentiment_mean REAL,
                sector_sentiment_skew REAL,
                sector_sentiment_std REAL,
                sector_news_volume INTEGER,
                relative_sentiment_ratio REAL,
                sector_sentiment_correlation REAL,
                sector_sentiment_divergence REAL,
                
                -- Comparison against market-wide data
                market_sentiment_correlation REAL,
                market_sentiment_divergence REAL,
                
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp, symbol)
            )
        ''')

        # Create indices for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_symbol_timestamp ON news(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_symbol_timestamp ON stock_prices(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_timestamp ON sentiment_features(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_features_timestamp ON market_features(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cross_symbol_features_timestamp_symbol ON cross_symbol_features(timestamp, symbol)')

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
                ''', (article.timestamp.isoformat() if article.timestamp else None,
                      article.title, article.description, 
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
                ''', (data.timestamp.isoformat() if data.timestamp else None,
                      data.symbol, data.open, data.high, 
                     data.low, data.close, data.volume, data.adj_close))
                inserted += 1
            except Exception as e:
                logger.error(f"Error inserting stock data: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Inserted {inserted} stock price records")
        return inserted
    
    def insert_sentiment_features_batch(self, features: List[SentimentFeatures]):
        """Insert sentiment features matching the SentimentFeatures dataclass"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted = 0
        for feature in features:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO sentiment_features 
                    (timestamp, symbol, sentiment_score, sentiment_skew, sentiment_std, 
                     sentiment_momentum, extreme_sentiment_ratio, sentiment_persistence, 
                     news_flow_intensity, news_volume, source_diversity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    feature.timestamp.isoformat() if feature.timestamp else None,
                    feature.symbol, feature.sentiment_score, 
                    feature.sentiment_skew, feature.sentiment_std, feature.sentiment_momentum,
                    feature.extreme_sentiment_ratio, feature.sentiment_persistence,
                    feature.news_flow_intensity, feature.news_volume, feature.source_diversity
                ))
                inserted += 1
            except Exception as e:
                logger.error(f"Error inserting sentiment features: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Inserted {inserted} sentiment feature records")
        return inserted
    
    def insert_market_features_batch(self, features: List[MarketFeatures]):
        """Insert market-wide features"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted = 0
        for feature in features:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO market_features 
                    (timestamp, market_sentiment_mean, market_sentiment_skew, market_sentiment_std,
                     market_sentiment_momentum, market_news_volume, market_source_credibility,
                     market_source_diversity, market_sentiment_regime, pre_market_sentiment,
                    market_hours_sentiment, after_market_sentiment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    feature.timestamp.isoformat() if feature.timestamp else None,
                    feature.market_sentiment_mean, feature.market_sentiment_skew,
                    feature.market_sentiment_std, feature.market_sentiment_momentum,
                    feature.market_news_volume, feature.market_source_credibility,
                    feature.market_source_diversity, feature.market_sentiment_regime,
                    feature.market_hours_sentiment, feature.pre_market_sentiment,
                    feature.after_market_sentiment
                ))
                inserted += 1
            except Exception as e:
                logger.error(f"Error inserting market features: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Inserted {inserted} market feature records")
        return inserted
    
    def insert_cross_symbol_features_batch(self, features: List[CrossSymbolFeatures]):
        """Insert cross-symbol features"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted = 0
        for feature in features:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO cross_symbol_features 
                    (timestamp, symbol, sector, sector_sentiment_mean, sector_sentiment_skew,
                     sector_sentiment_std, sector_news_volume, relative_sentiment_ratio,
                     sector_sentiment_correlation, sector_sentiment_divergence,
                     market_sentiment_correlation, market_sentiment_divergence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    feature.timestamp.isoformat() if feature.timestamp else None,
                    feature.symbol, feature.sector,feature.sector_sentiment_mean, 
                    feature.sector_sentiment_skew, feature.sector_sentiment_std, feature.sector_news_volume,
                    feature.relative_sentiment_ratio, feature.sector_sentiment_correlation,
                    feature.sector_sentiment_divergence, feature.market_sentiment_correlation,
                    feature.market_sentiment_divergence
                ))
                inserted += 1
            except Exception as e:
                logger.error(f"Error inserting cross-symbol features: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Inserted {inserted} cross-symbol feature records")
        return inserted

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

    def get_sentiment_features_data(self, symbol: Optional[str] = None,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve sentiment features data as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM sentiment_features WHERE 1=1"
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
    
    def get_market_features_data(self, start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve market features data as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM market_features WHERE 1=1"
        params = []
        
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
    
    def get_cross_symbol_features_data(self, symbol: Optional[str] = None,
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve cross-symbol features data as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM cross_symbol_features WHERE 1=1"
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
        
        # Market features summary
        market_features_summary = pd.read_sql_query('''
            SELECT COUNT(*) as feature_count,
                   MIN(timestamp) as earliest_feature,
                   MAX(timestamp) as latest_feature,
                   AVG(market_sentiment_mean) as avg_market_sentiment,
                   AVG(market_news_volume) as avg_news_volume
            FROM market_features
        ''', conn)
        
        # Sentiment features summary
        sentiment_features_summary = pd.read_sql_query('''
            SELECT symbol, COUNT(*) as feature_count,
                   MIN(timestamp) as earliest_feature,
                   MAX(timestamp) as latest_feature,
                   AVG(sentiment_score) as avg_sentiment,
                   AVG(sentiment_momentum) as avg_sentiment_momentum,
                   AVG(news_volume) as avg_news_volume
            FROM sentiment_features
            GROUP BY symbol
            ORDER BY feature_count DESC
        ''', conn)
        
        # Cross-symbol features summary
        cross_symbol_features_summary = pd.read_sql_query('''
            SELECT symbol, sector, COUNT(*) as feature_count,
                   MIN(timestamp) as earliest_feature,
                   MAX(timestamp) as latest_feature,
                   AVG(sector_sentiment_mean) as avg_sector_sentiment,
                   AVG(relative_sentiment_ratio) as avg_relative_sentiment
            FROM cross_symbol_features
            GROUP BY symbol, sector
            ORDER BY feature_count DESC
        ''', conn)
        
        conn.close()
        
        return {
            'news_summary': news_summary,
            'stock_summary': stock_summary,
            'market_features_summary': market_features_summary,
            'sentiment_features_summary': sentiment_features_summary,
            'cross_symbol_features_summary': cross_symbol_features_summary
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

        cursor.execute('DELETE FROM market_features WHERE timestamp < ?', (cutoff_date,))
        market_features_deleted = cursor.rowcount

        cursor.execute('DELETE FROM cross_symbol_features WHERE timestamp < ?', (cutoff_date,))
        cross_symbol_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {news_deleted} old news records, {features_deleted} old sentiment feature records, {market_features_deleted} old market feature records, and {cross_symbol_deleted} cross-symbol feature records")
        return news_deleted, features_deleted, market_features_deleted, cross_symbol_deleted