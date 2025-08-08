import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from src.data.database import DatabaseManager
from src.data.models import NewsArticle
from src.sentiment.feature_engineering import SentimentFeatureEngineer
from src.sentiment.analyzers import EnsembleSentimentAnalyzer

class SentimentAnalysisRunner:
    """Handles sentiment analysis for news articles"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager(config.DATABASE_PATH)
        self.sentiment_analyzer = EnsembleSentimentAnalyzer()
        self.feature_engineer = SentimentFeatureEngineer()
        
        # Check if analyzer initialized successfully
        if not self.sentiment_analyzer.is_initialized:
            self.logger.error("Sentiment analyzer failed to initialize")
            raise RuntimeError("Cannot proceed without sentiment analyzer")
    
    def analyze_articles(self, 
                        symbols: Optional[List[str]] = None,
                        article_ids: Optional[List[int]] = None,
                        since_date: Optional[datetime] = None,
                        batch_size: int = 100,
                        force_reanalyze: bool = False) -> dict:
        """Analyze sentiment for articles matching the criteria"""
        # Build query conditions
        conditions = []
        params = []
        
        if symbols:
            placeholders = ','.join(['?' for _ in symbols])
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)
        if article_ids:
            placeholders = ','.join(['?' for _ in article_ids])
            conditions.append(f"id IN ({placeholders})")
            params.extend(article_ids)
        if since_date:
            conditions.append("timestamp >= ?")
            params.append(since_date)
        if not force_reanalyze:
            conditions.append("sentiment_score IS NULL")
        
        # Build and execute query
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT id, timestamp, title, description, source, url, symbol, sentiment_score
            FROM news 
            WHERE {where_clause}
            ORDER BY timestamp DESC
        """
        
        self.logger.info(f"Fetching articles with query: {query}")
        self.logger.info(f"Query parameters: {params}")
        
        # Get articles from database
        import sqlite3
        conn = sqlite3.connect(self.config.DATABASE_PATH)
        articles_df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if articles_df.empty:
            self.logger.warning("No articles found matching the criteria")
            return {
                'articles_processed': 0,
                'articles_updated': 0,
                'symbols_processed': [],
                'processing_time': 0.0
            }
        self.logger.info(f"Found {len(articles_df)} articles to analyze")

        # Process in batches
        start_time = datetime.now()
        updated_count = 0
        processed_symbols = set()
        
        for i in range(0, len(articles_df), batch_size):
            batch_df = articles_df.iloc[i:i+batch_size].copy()
            self.logger.info(f"Processing BATCH {i//batch_size + 1} ({len(batch_df)} articles)")
            
            # Analyze sentiment for batch
            batch_results = self._analyze_batch(batch_df)
            # Update database with results
            updated = self._update_sentiment_scores(batch_results)
            updated_count += updated
            # Track processed symbols
            processed_symbols.update(batch_df['symbol'].unique())
            self.logger.info(f"Batch completed: {updated} articles updated")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        results = {
            'articles_processed': len(articles_df),
            'articles_updated': updated_count,
            'symbols_processed': list(processed_symbols),
            'processing_time': processing_time
        }
        self.logger.info(f"Sentiment analysis completed: {results}")
        return results
    
    def _analyze_batch(self, articles_df: pd.DataFrame) -> List[dict]:
        """Analyze sentiment for a batch of articles"""
        results = []
        
        for _, row in articles_df.iterrows():
            try:
                article = NewsArticle(
                    timestamp=pd.to_datetime(row['timestamp']),
                    title=row['title'] or '',
                    description=row['description'] or '',
                    source=row['source'] or '',
                    url=row['url'] or '',
                    symbol=row['symbol']
                )

                sentiment = self.feature_engineer.analyze_article_sentiment(article)
                results.append({
                    'id': row['id'],
                    'sentiment_score': sentiment['compound'],
                    'sentiment_positive': sentiment['positive'],
                    'sentiment_neutral': sentiment['neutral'],
                    'sentiment_negative': sentiment['negative']
                })
                
            except Exception as e:
                self.logger.error(f"Error analyzing article {row['id']}: {e}")
                results.append({
                    'id': row['id'],
                    'sentiment_score': 0.0,
                    'sentiment_positive': 0.0,
                    'sentiment_neutral': 1.0,
                    'sentiment_negative': 0.0
                })
        
        return results
    
    def _update_sentiment_scores(self, results: List[dict]) -> int:
        """Update sentiment scores in database"""
        import sqlite3
        conn = sqlite3.connect(self.config.DATABASE_PATH)
        cursor = conn.cursor()
        
        updated_count = 0
        for result in results:
            try:
                cursor.execute("""
                    UPDATE news 
                    SET sentiment_score = ?,
                        created_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (result['sentiment_score'], result['id']))
                
                if cursor.rowcount > 0:
                    updated_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error updating article {result['id']}: {e}")
        
        conn.commit()
        conn.close()
        return updated_count
    
    def get_sentiment_summary(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get sentiment analysis summary"""
        conditions = []
        params = []

        if symbols:
            placeholders = ','.join(['?' for _ in symbols])
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT 
                symbol,
                COUNT(*) as total_articles,
                COUNT(CASE WHEN sentiment_score IS NOT NULL THEN 1 END) as analyzed_articles,
                AVG(sentiment_score) as avg_sentiment,
                MIN(sentiment_score) as min_sentiment,
                MAX(sentiment_score) as max_sentiment,
                COUNT(CASE WHEN sentiment_score > 0.1 THEN 1 END) as positive_articles,
                COUNT(CASE WHEN sentiment_score < -0.1 THEN 1 END) as negative_articles,
                COUNT(CASE WHEN sentiment_score BETWEEN -0.1 AND 0.1 THEN 1 END) as neutral_articles,
                MIN(timestamp) as earliest_article,
                MAX(timestamp) as latest_article
            FROM news
            WHERE {where_clause}
            GROUP BY symbol
            ORDER BY analyzed_articles DESC
        """
        
        import sqlite3
        conn = sqlite3.connect(self.config.DATABASE_PATH)
        summary_df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Calculate analysis progress percentage
        summary_df['analysis_progress'] = (
            summary_df['analyzed_articles'] / summary_df['total_articles'] * 100
        ).round(2)
        
        return summary_df
    
    def analyze_by_time_period(self, 
                              symbols: List[str],
                              start_date: datetime,
                              end_date: datetime) -> pd.DataFrame:
        """Analyze sentiment trends over time period"""
        
        placeholders = ','.join(['?' for _ in symbols])
        params = symbols + [start_date, end_date]
        
        query = f"""
            SELECT 
                symbol,
                DATE(timestamp) as date,
                COUNT(*) as article_count,
                AVG(sentiment_score) as avg_sentiment,
                COUNT(CASE WHEN sentiment_score > 0.1 THEN 1 END) as positive_count,
                COUNT(CASE WHEN sentiment_score < -0.1 THEN 1 END) as negative_count
            FROM news
            WHERE symbol IN ({placeholders})
                AND timestamp BETWEEN ? AND ?
                AND sentiment_score IS NOT NULL
            GROUP BY symbol, DATE(timestamp)
            ORDER BY symbol, date
        """
        
        import sqlite3
        conn = sqlite3.connect(self.config.DATABASE_PATH)
        trends_df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return trends_df
