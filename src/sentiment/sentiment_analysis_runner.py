import sys
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from config.symbols import get_symbols_by_sector, get_symbol_sector, get_all_symbols
from src.data.database import DatabaseManager
from src.models import NewsArticle, SentimentFeatures
from src.sentiment.feature_engineering import SentimentFeatureEngineer
from src.sentiment.analyzers import EnsembleSentimentAnalyzer
from .utils import StatisticalUtils

class SentimentAnalysisRunner:
    """Handles sentiment analysis for news articles"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager(config.DATABASE_PATH)
        self.sentiment_analyzer = EnsembleSentimentAnalyzer()
        self.feature_engineer = SentimentFeatureEngineer()
        self.stats_utils = StatisticalUtils()
        
        # Check if analyzer initialized successfully
        if not self.sentiment_analyzer.is_initialized:
            self.logger.error("Sentiment analyzer failed to initialize")
            raise RuntimeError("Cannot proceed without sentiment analyzer")
        
    def analyze_articles(self, 
                         symbols: Optional[List[str]]= None,
                         article_ids: Optional[List[int]] = None,
                         since_date:  Optional[datetime] = None,
                         batch_size: int = 100,
                         force_reanalyze: bool = False,
                         create_features: bool = True,
                         enable_cross_symbol: bool = False,
                         cross_symbol_window: int = 24) -> Dict:
        """Analyze sentiment for articles matching passed criteria and optionally create features"""
        # STEP 1: Analyze individual articles
        analysis_results = self._analyze_individual_articles(
            symbols, article_ids, since_date, batch_size, force_reanalyze
        )

        if analysis_results['articles_updated'] == 0:
            return analysis_results
        
        # STEP 2: Create and populate sentiment features if requested
        if create_features:
            self.logger.info("Creating sentiment features... ")
            features_results = self._create_sentiment_features(
                analysis_results['symbols_processed'],
                since_date,
                enable_cross_symbol,
                cross_symbol_window
            )
            analysis_results['features_created'] = features_results['features_created']
            analysis_results['features_updated'] = features_results['features_updated']
        else:
            analysis_results['features_created'] = 0
            analysis_results['features_updated'] = 0

        return analysis_results
    
    def _analyze_individual_articles(self,
                                   symbols: Optional[List[str]] = None,
                                   article_ids: Optional[List[int]] = None,
                                   since_date: Optional[datetime] = None,
                                   batch_size: int = 100,
                                   force_reanalyze: bool = False) -> dict:
        """Analyze sentiment for individual articles"""
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
        
        return {
            'articles_processed': len(articles_df),
            'articles_updated': updated_count,
            'symbols_processed': list(processed_symbols),
            'processing_time': processing_time
        }
    
    def _create_sentiment_features(self, symbols: List[str], 
                                   since_date: Optional[datetime] = None,
                                   enable_cross_symbol: bool = False,
                                   cross_symbol_window: int = 24) -> Dict[str, int]:
        """Create sentiment features for the analyzed symbols with optional cross-symbol """
        if not symbols:
            return {'features_created': 0, 'features_updated': 0}
        
        features_created = 0
        features_updated = 0
        # Default lookback for feature creation
        lookback_date = since_date or (datetime.now() - timedelta(days=self.config.DEFAULT_LOOKBACK_DAYS))

        all_symbols = get_all_symbols() if enable_cross_symbol else symbols
        all_news_df = None

        if enable_cross_symbol:
            self.logger.info("Loading all relevant news data for cross-symbol analysis...")
            all_news_df = self._get_all_news_for_cross_symbol(lookback_date, cross_symbol_window)

        for symbol in symbols:
            try:
                if enable_cross_symbol and all_news_df is not None:
                    symbol_features = self._create_symbol_features_with_cross_symbol(
                        symbol, since_date, all_news_df, all_symbols
                    )
                else:
                    symbol_features = self._create_symbol_features(symbol, lookback_date)

                if symbol_features:
                    inserted = self.db_manager.insert_sentiment_features_batch(symbol_features)
                    features_created += inserted
                    self.logger.info(f"Created {inserted} sentiment features for {symbol}")
            except Exception as e:
                self.logger.error(f"Error creating features for {symbol}: {e}")

        return {
            'features_created': features_created,
            'features_updated': features_updated
        }
    
    def _get_all_news_for_cross_symbol(self, lookback_date: datetime,
                                        window_hours: int) -> pd.DataFrame:
        """Get all news data needed for cross-symbol analysis"""
        end_time = datetime.now()
        # Extra buffer for lookback calculations
        start_time = lookback_date - timedelta(hours=window_hours)

        all_news_df = self.db_manager.get_news_data(
            start_date=start_time,
            end_date=end_time
        )

        # Filter only articles with sentiment scores
        return all_news_df[all_news_df['sentiment_score'].notna()].copy()
    
    def _create_symbol_features(self, symbol: str, since_date: datetime) -> List[SentimentFeatures]:
        """Create sentiment featurse for a single symbol"""
        # Get already-analyzed articles for this symbol
        articles_df = self.db_manager.get_news_data(
            symbol = symbol,
            start_date = since_date
        ).dropna(subset=['sentiment_score'])

        if articles_df.empty:
            self.logger.warning(f"No analyzed articles found for {symbol}")
            return []
        self.logger.info(f"Creating features for {symbol} with {len(articles_df)} articles...")
        # Create time-based features
        time_features_df = self.feature_engineer.create_time_based_features(
            articles_df, symbol, time_windows = ['1h', '4h', '24h']
        )

        if time_features_df.empty:
            return []
        
        # Convert to SentimentFeatures objects
        features_list = []
        for _ , row in time_features_df.iterrows():
            try:
                # Extract key metrics from the feature row
                sentiment_score = row.get('sentiment_mean_24h', 0.0)
                sentiment_momentum = row.get('sentiment_momentum_24h', 0.0)
                news_volume = int(row.get('news_volume_24h', 0))
                source_diversity = row.get('source_diversity_24h', 0.0)

                timestamp = row['timestamp']
                if hasattr(timestamp, 'to_pydatetime'):
                    timestamp = timestamp.to_pydatetime() 

                feature = SentimentFeatures(
                    timestamp=timestamp,
                    symbol=symbol,
                    sentiment_score=float(sentiment_score),
                    sentiment_momentum=float(sentiment_momentum),
                    news_volume=news_volume,
                    source_diversity=float(source_diversity)
                )
                features_list.append(feature)                

            except Exception as e:
                self.logger.error(f"Error creating feature object: {e}")
                continue
        return features_list
    
    def _create_symbol_features_with_cross_symbol(self, symbol: str,
                                                  since_date: datetime,
                                                  all_news_df: pd.DataFrame,
                                                  all_symbols: List[str]) -> List[SentimentFeatures]:
        """Create sentiment features with cross-symbol for a passed symbol"""
        # Get already-analyzed articles for this symbol
        symbol_articles_df = self.db_manager.get_news_data(
            symbol = symbol,
            start_date = since_date
        ).dropna(subset=['sentiment_score'])

        if symbol_articles_df.empty:
            self.logger.warning(f"No analyzed articles found for {symbol}. Returning empty list..")
            return []

        self.logger.info(f"Creating enhanced features for {symbol} with {len(symbol_articles_df)} articles..")

        # Get sector information
        sector = get_symbol_sector(symbol)
        sector_symbols = get_symbols_by_sector(sector) if sector else []

        # Create time-based features
        time_features_df = self.feature_engineer.create_time_based_features(
            symbol_articles_df, symbol, time_windows=['1h', '4h', '24h']
        )
        if time_features_df.empty:
            return []
        
        # Create cross-symbol features & merge
        cross_features_df = self.feature_engineer.create_cross_symbol_features(
            all_news_df, symbol, sector_symbols
        )
        if not cross_features_df.empty:
            enhanced_features_df = pd.merge(
                time_features_df, 
                cross_features_df,
                on='timestamp',
                how='left'
            )
        else:
            enhanced_features_df = time_features_df

        # Convert to SentimentFeatures objects
        features_list = []
        for _, row in enhanced_features_df.iterrows():
            try:
                timestamp = row['timestamp']
                if hasattr(timestamp, 'to_pydatetime'):
                    timestamp = timestamp.to_pydatetime().astimezone(timezone.utc)
                elif isinstance(timestamp, pd.Timestamp):
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.tz_localize("UTC")
                    else:
                        timestamp = timestamp.tz_convert("UTC")
                
                market_features = self._get_market_features(
                    symbol, timestamp, all_news_df, all_symbols
                )

                feature = SentimentFeatures(
                    timestamp=timestamp,
                    symbol=symbol,
                    # Basic Features
                    sentiment_score=float(row.get('sentiment_mean_24h', 0.0)),
                    sentiment_momentum=float(row.get('sentiment_momentum_24h', 0.0)),
                    news_volume=int(row.get('news_volume_24h', 0)),
                    source_diversity=float(row.get('source_diversity_24h', 0.0)),
                    
                    # Cross-symbol features
                    sector_sentiment_mean=self.stats_utils.safe_float(row.get('sector_sentiment_mean')),
                    sentiment_sector_correlation=self.stats_utils.safe_float(row.get('sentiment_sector_correlation')),
                    relative_sentiment_strength=self.stats_utils.safe_float(row.get('relative_sentiment_strength')),
                    sector_news_volume=self.stats_utils.safe_int(row.get('sector_news_volume')),
                    sentiment_divergence=self.stats_utils.safe_float(row.get('sentiment_divergence')),
                    
                    # Market-wide features
                    market_sentiment_mean=self.stats_utils.safe_float(market_features.get('market_sentiment_mean')),
                    sentiment_market_correlation=self.stats_utils.safe_float(market_features.get('sentiment_market_correlation')),
                    market_news_volume=self.stats_utils.safe_int(market_features.get('market_news_volume')),
                    sector_sentiment_volatility=self.stats_utils.safe_float(market_features.get('sector_sentiment_volatility')),
                    market_sentiment_volatility=self.stats_utils.safe_float(market_features.get('market_sentiment_volatility'))
                )
                features_list.append(feature)
            
            except Exception as e:
                self.logger.error(f"Error creating enhanced feature object: {e}")
                continue
        
        return features_list

    def _get_market_features(self, symbol: str,
                             timestamp: datetime, 
                             all_news_df: pd.DataFrame, 
                             all_symbols: List[str]) -> Dict:
        """Calculate market-wide features in last 24 hours using existing patterns"""
        window_start = timestamp - timedelta(hours=24)
        sector = get_symbol_sector(symbol)
        sector_symbols = get_symbols_by_sector(sector)

        market_data = all_news_df[
            (all_news_df['timestamp'] >= window_start) & 
            (all_news_df['timestamp'] <= timestamp)
        ]
        if market_data.empty:
            return {}
        
        sector_data = all_news_df[
            (all_news_df['timestamp'] >= window_start) &
            (all_news_df['timestamp'] <= timestamp) &
            (all_news_df['symbol'].isin(sector_symbols))
        ]

        return {
            'market_sentiment_mean': float(self.stats_utils.weighted_average(market_data['sentiment_score'].values)),
            'market_news_volume': len(market_data),
            'market_volatility': float(self.stats_utils.safe_std(market_data['sentiment_score'].values)),
            'market_correlation': 0.0, # would need target symbol sentiments for correlation
            'sector_sentiment_volatility': float(self.stats_utils.safe_std(sector_data['sentiment_score'].values))
        }

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
        """Get sentiment analysis summary from passed symbols"""
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
    
    def get_features_summary(self, symbols: Optional[List[str]] = None,
                             include_cross_symbol: bool = False) -> pd.DataFrame:
        """Get sentiment features summary from passed symbols"""
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
                COUNT(*) as total_features,
                AVG(sentiment_score) as avg_sentiment,
                AVG(sentiment_momentum) as avg_momentum,
                AVG(news_volume) as avg_news_volume,
                AVG(source_diversity) as avg_source_diversity,
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
                    AVG(market_sentiment_mean) as avg_market_sentiment,
                    AVG(sentiment_sector_correlation) as avg_sector_correlation,
                    AVG(relative_sentiment_strength) as avg_relative_strength,
                    AVG(sector_news_volume) as avg_sector_news_volume,
                    AVG(market_news_volume) as avg_market_news_volume
                FROM sentiment_features
                WHERE {where_clause}
                    AND sector_sentiment_mean IS NOT NULL
                GROUP BY symbol
            """
            
            cross_df = pd.read_sql_query(cross_query, conn, params=params)
            if not cross_df.empty:
                features_df = features_df.merge(cross_df, on='symbol', how='left')
        
        conn.close()
        return features_df

    def analyze_by_time_period(self, 
                              symbols: List[str],
                              start_date: datetime,
                              end_date: datetime,
                              include_cross_symbol: bool = False) -> pd.DataFrame:
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

        # Add cross-symbol trends if requested
        if include_cross_symbol and not trends_df.empty:
            cross_query = f"""
                SELECT 
                    symbol,
                    DATE(timestamp) as date,
                    AVG(sector_sentiment_mean) as avg_sector_sentiment,
                    AVG(market_sentiment_mean) as avg_market_sentiment,
                    AVG(sentiment_sector_correlation) as avg_sector_correlation,
                    AVG(relative_sentiment_strength) as avg_relative_strength
                FROM sentiment_features
                WHERE symbol IN ({placeholders})
                    AND timestamp BETWEEN ? AND ?
                    AND sector_sentiment_mean IS NOT NULL
                GROUP BY symbol, DATE(timestamp)
            """
            
            cross_trends_df = pd.read_sql_query(cross_query, conn, params=params)
            if not cross_trends_df.empty:
                trends_df = trends_df.merge(cross_trends_df, on=['symbol', 'date'], how='left')
        
        conn.close()
        return trends_df

    def get_cross_symbol_insights(self, symbols: List[str], days_back: int=7) -> pd.DataFrame:
        """Get cross-symbol sentiment insights for specified symbols"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        placeholders = ','.join(['?' for _ in symbols])
        params = symbols + [start_date, end_date]

        query = f"""
            SELECT 
                symbol,
                AVG(sector_sentiment_mean) as avg_sector_sentiment,
                AVG(market_sentiment_mean) as avg_market_sentiment,
                AVG(sentiment_sector_correlation) as sector_correlation,
                AVG(sentiment_market_correlation) as market_correlation,
                AVG(relative_sentiment_strength) as relative_strength,
                AVG(sentiment_divergence) as sentiment_divergence,
                COUNT(*) as observation_count,
                MIN(timestamp) as period_start,
                MAX(timestamp) as period_end
            FROM sentiment_features
            WHERE symbol IN ({placeholders})
                AND timestamp BETWEEN ? AND ?
                AND sector_sentiment_mean IS NOT NULL
            GROUP BY symbol
            ORDER BY relative_strength DESC
        """
        try:
            import sqlite3
            conn = sqlite3.connect(self.config.DATABASE_PATH)
            insights_df = pd.read_sql_query(query, conn, params=params)
            return insights_df
        except Exception as e:
            logging.error(f"Error gathering cross-symbol data for {symbols}: {e}")
        finally:
            conn.close()