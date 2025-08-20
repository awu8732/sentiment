import sys
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from src.data.database import DatabaseManager
from src.models.features.sentiment_features import SentimentFeatures
from src.managers import CrossSymbolSentimentManager, MarketSentimentManager
from src.sentiment.feature_engineering import SentimentFeatureEngineer
from src.sentiment.utils import NewsUtils, StatisticalUtils, SentimentUtils

class SentimentFeatureManager:
    """Implements combined script-facing functions for CrossSymbolSentimentManager, 
    MarketSentimentManager, and SentimentFeatureEngineer"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager(config.DATABASE_PATH)
        self.feature_engineer = SentimentFeatureEngineer()
        self.news_utils = NewsUtils()
        self.stats_utils = StatisticalUtils()
        self.sentiment_utils = SentimentUtils()
        self.market_manager = MarketSentimentManager(config, 
                                                     self.logger, 
                                                     self.db_manager, 
                                                     self.feature_engineer)
        self.cross_symbol_manager = CrossSymbolSentimentManager(config,
                                                                self.logger,
                                                                self.db_manager,
                                                                self.feature_engineer)
    
    def analyze_articles(self, 
                        symbols: Optional[List[str]]= None,
                        article_ids: Optional[List[int]] = None,
                        start_date:  Optional[datetime] = None,
                        end_date:  Optional[datetime] = None,
                        batch_size: int = 100,
                        force_reanalyze: bool = False,
                        create_features: bool = True,
                        enable_cross_symbol: bool = False,
                        window_size: int = 24) -> Dict:
        """Analyze sentiment for articles matching passed critera and optionally create features"""
        # Analyze articles for sentiment-only
        analysis_results = self._analyze_article_sentiment(
            symbols, article_ids, start_date, end_date, batch_size, force_reanalyze
        )
        if analysis_results['articles_updated'] == 0:
            return analysis_results
        
        # Create and populate sentiment features if requested
        if create_features:
            self.logger.info("Creating sentiment features... ")
            start_time = datetime.now()

            for symbol in analysis_results['symbols_processed']:
                features_results = self._create_symbol_features(symbol, start_date, end_date, enable_cross_symbol, window_size)

                if enable_cross_symbol:
                    self.cross_symbol_manager.create_cross_sector_features(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        window_size=window_size
                    )
                analysis_results['features_created'] = analysis_results.get('features_created', 0) + features_results['features_created']
                analysis_results['features_updated'] = analysis_results.get('features_updated', 0) + features_results['features_updated']
            
            processing_time = (datetime.now() - start_time).total_seconds()
            analysis_results['feature_processing_time'] = processing_time
        
        self.logger.info(f"Article analysis {'and feature creation' if create_features else ''} complete")
        return analysis_results
    
    def create_market_features(self,
                               start_date: Optional[datetime] = None, 
                               end_date: Optional[datetime] = None, 
                               window_size: int = 24) -> Dict:
        """Create market-wdie features from MarketSentimentManager"""
        market_df = self.market_manager._get_market_data(start_date, end_date)
        analysis_results = self.market_manager._create_market_features(market_df, window_size)
        return analysis_results
    
    # Summary methods
    def get_sentiment_summary(self, symbols: Optional[List[str]] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get sentiment analysis summary from passed symbols, with optional date filters"""
        conditions = []
        params = []

        if symbols:
            placeholders = ','.join(['?' for _ in symbols])
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat() if isinstance(start_date, datetime) else start_date)

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat() if isinstance(end_date, datetime) else end_date)

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
            
    def get_features_summary(self, symbols: Optional[List[str]] = None,
                            include_cross_symbol: bool = True,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get sentiment features summary from passed symbols, with optional date filters"""
        conditions = []
        params = []

        if symbols:
            placeholders = ','.join(['?' for _ in symbols])
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat() if isinstance(start_date, datetime) else start_date)

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat() if isinstance(end_date, datetime) else end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT 
                symbol,
                COUNT(*) as total_features,
                AVG(sentiment_score) as avg_sentiment,
                AVG(sentiment_std) as avg_std,
                AVG(sentiment_skew) as avg_skew,
                AVG(sentiment_momentum) as avg_momentum,
                AVG(sentiment_persistence) as avg_persistence,
                AVG(news_volume) as avg_news_volume,
                MIN(timestamp) as earliest_feature,
                MAX(timestamp) as latest_feature
            FROM sentiment_features
            WHERE {where_clause}
            GROUP BY symbol
            ORDER BY total_features DESC
        """
        
        import sqlite3
        conn = sqlite3.connect(self.config.DATABASE_PATH)
        features_df = pd.read_sql_query(query, conn, params=params)

        # Add cross-symbol summary if requested
        if include_cross_symbol and not features_df.empty:
            cross_query = f"""
                SELECT 
                    symbol,
                    COUNT(*) as cross_symbol_records,
                    AVG(sector_sentiment_mean) as avg_sector_sentiment,
                    AVG(sector_sentiment_std) as avg_sector_std,
                    AVG(sector_sentiment_skew) as avg_sector_skew,
                    AVG(relative_sentiment_ratio) as avg_relative_ratio,
                    AVG(sentiment_sector_correlation) as avg_sector_correlation,
                    AVG(sentiment_sector_divergence) as avg_sector_divergence,
                    AVG(sector_news_volume) as avg_sector_news_volume
                FROM cross_symbol_features
                WHERE {where_clause}
                    AND sector_sentiment_mean IS NOT NULL
                GROUP BY symbol
            """
            
            cross_df = pd.read_sql_query(cross_query, conn, params=params)
            if not cross_df.empty:
                features_df = features_df.merge(cross_df, on='symbol', how='left')
        
        conn.close()
        return features_df

    def get_market_features_summary(self, start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get market-wide features summary"""
        params = []
        query = """
            SELECT 
                symbol,
                COUNT(*) as total_features,
                AVG(market_sentiment_mean) as avg_sentiment,
                AVG(market_sentiment_std) as avg_std,
                AVG(market_sentiment_skew) as avg_skew,
                AVG(market_sentiment_momentum) as avg_momentum,
                AVG(market_news_volume) as avg_news_volume,
                AVG(market_hours_sentiment) as avg_market_hours_sentiment,
                AVG(pre_market_sentiment) as avg_pre_market_sentiment,
                AVG(after_hours_sentiment) as avg_after_hours_sentiment,
                MIN(timestamp) as earliest_feature,
                MAX(timestamp) as latest_feature
            FROM market_features
            WHERE 1=1
        """

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat() if isinstance(start_date, datetime) else start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat() if isinstance(end_date, datetime) else end_date)

        query += """
            GROUP BY symbol
            ORDER BY total_features DESC
        """

        import sqlite3
        conn = sqlite3.connect(self.config.DATABASE_PATH)
        features_df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return features_df

    # Interal processing methods
    def _analyze_article_sentiment(self,
                                   symbols: Optional[List[str]] = None,
                                   article_ids: Optional[List[int]] = None,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None,
                                   batch_size: int = 100,
                                   force_reanalyze: bool = False) -> dict:
        """Specifically analyze sentiment for articles, possibly of varying symbols"""
        self.logger.info( f"Starting article sentiment analysis | "
                          f"symbols={symbols}, article_ids={article_ids}, " 
                          f"start_date={start_date}, end_date={end_date}, " 
                          f"batch_size={batch_size}, force_reanalyze={force_reanalyze}" )
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
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.isoformat()
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.isoformat()
            conditions.append("timestamp <= ?")
            params.append(end_date)
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
                'sentiment_processing_time': 0.0,
                'feature_processing_time': 0.0
            }
        self.logger.info(f"Found {len(articles_df)} articles to analyze")

        # Process articles_df in batches
        start_time = datetime.now()
        updated_count = 0
        processed_symbols = set()
        
        for i in range(0, len(articles_df), batch_size):
            batch_df = articles_df.iloc[i:i+batch_size].copy()
            self.logger.info(f"Processing BATCH {i//batch_size + 1} ({len(batch_df)} articles)")
            
            # Analyze sentiment for batch
            batch_results = self.feature_engineer.analyze_article_batch_sentiment(batch_df)
            updated = self._update_sentiment_scores(batch_results)
            updated_count += updated
            # Track processed symbols
            processed_symbols.update(batch_df['symbol'].unique())
            self.logger.info(f"Batch completed: {updated} articles updated")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'articles_processed': len(articles_df),
            'articles_updated': updated_count,
            'symbols_processed': list(processed_symbols),
            'sentiment_processing_time': processing_time,
            'feature_processing_time': 0.0
        }
    
    def _create_symbol_features(self, 
                                symbol: str,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                enable_cross_symbol: bool = False,
                                window_size: int = 24) -> Dict:
        """Create sentiment features for a specific symbol, with optional cross-symbol analysis"""
        if not symbol:
            return {'features_created': 0, 'features_updated': 0}
        
        # Query db_manager for symbol data
        symbol_df = self.db_manager.get_news_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        if symbol_df is not None and not symbol_df.empty:
            self.logger.info(f"Creating features for {symbol} ({len(symbol_df)})")
        else:
            self.logger.warning(f"No articles found for {symbol}")
            return {'features_created': 0, 'features_updated': 0}
        
        # Convert naive timestamps to UTC
        symbol_df['timestamp'] = pd.to_datetime(
            symbol_df['timestamp'], 
            errors='coerce', 
            utc=True
        )
        # Set timestamp as index for easier slicing
        symbol_df = symbol_df.set_index('timestamp')

        # Generate hourly time range based on available data extrema
        start_time = symbol_df.index.min().floor('h')
        end_time = symbol_df.index.max().ceil('h')
        time_range = pd.date_range(start=start_time, end=end_time, freq='1h')

        features_list = []
        for timestamp in time_range:
            window_start = timestamp - pd.Timedelta(hours=window_size)
            window_articles = symbol_df[(symbol_df.index >= window_start) & (symbol_df.index < timestamp)].reset_index()
            window_sentiments = window_articles['sentiment_score'].dropna()
            sentiment_feature = SentimentFeatures(
                timestamp=timestamp,
                symbol=symbol,
                sentiment_score = self.stats_utils.weighted_average(window_sentiments)
            )

            if len(window_articles) == 0 or len(window_sentiments) == 0:
                features_list.append(sentiment_feature)
                continue

            sentiment_feature.sentiment_skew = self.stats_utils.safe_skew(window_sentiments)
            sentiment_feature.sentiment_std = self.stats_utils.safe_std(window_sentiments)
            sentiment_feature.sentiment_momentum = self.sentiment_utils.calculate_sentiment_momentum(window_sentiments)
            sentiment_feature.extreme_sentiment_ratio = self.sentiment_utils.calculate_extreme_sentiment_ratio(window_sentiments)
            sentiment_feature.sentiment_persistence = self.sentiment_utils.calculate_sentiment_persistence(window_sentiments)
            sentiment_feature.news_flow_intensity = self.news_utils.calculate_news_flow_intensity(window_articles)
            sentiment_feature.news_volume = len(window_articles)
            sentiment_feature.source_diversity = self.news_utils.calculate_source_diversity(window_articles)
            features_list.append(sentiment_feature)
    
        # Attempt to insert new batch into db
        features_created = 0
        try:
            if features_list:
                inserted = self.db_manager.insert_sentiment_features_batch(features_list)
                features_created += inserted
                self.logger.info(f"Created {inserted} new sentiment features for symbol {symbol}")
        except Exception as e:
            self.logger.error(f"Error inserting sentiment features for {symbol}: {e}")

        return {
            'features_created': features_created,
            'features_updated': 0
        }
    
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