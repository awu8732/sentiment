#!/usr/bin/env python3
"""
Analyze sentiment for collected news articles and updates the database.

Usage:
    python scripts/analyze_sentiment.py --all                    # Analyze all articles
    python scripts/analyze_sentiment.py --symbols AAPL GOOGL    # Analyze specific symbols
    python scripts/analyze_sentiment.py --sector technology      # Analyze by sector
    python scripts/analyze_sentiment.py --article-ids 1 2 3     # Analyze specific articles
    python scripts/analyze_sentiment.py --since "2024-01-01"    # Analyze articles since date
    python scripts/analyze_sentiment.py --batch-size 50         # Custom batch size

    # Cross-symbol analysis options
    python scripts/analyze_sentiment.py --all --cross-symbol                    # Enable cross-symbol features
    python scripts/analyze_sentiment.py --symbols AAPL MSFT --cross-symbol     # With specific symbols
    python scripts/analyze_sentiment.py --cross-symbol --cross-window-hours 48 # Custom time window
    
    # Cross-symbol insights and reporting
    python scripts/analyze_sentiment.py --cross-insights --symbols AAPL GOOGL  # Show cross-symbol insights
    python scripts/analyze_sentiment.py --cross-insights --days-back 14        # Custom time period for insights
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from config.symbols import get_all_symbols, get_symbols_by_sector
from src.utils.logger import setup_logging
from src.managers.sentiment_feature_manager import SentimentFeatureManager

def main():
    parser = argparse.ArgumentParser(
        description='Analyze sentiment for collected news articles',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Article selection options
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument('--all', action='store_true', 
                               help='Analyze all articles without sentiment scores')
    selection_group.add_argument('--symbols', nargs='+', 
                               help='Analyze articles for specific symbols')
    selection_group.add_argument('--sector', 
                               choices=['technology', 'finance', 'healthcare', 'energy'],
                               help='Analyze articles for specific sector')
    selection_group.add_argument('--article-ids', nargs='+', type=int,
                               help='Analyze specific article IDs')
    selection_group.add_argument('--market-features', action='store_true',
                               help='Generate market-wide features (ignores symbols/sectors)')
    
    # Additional filters
    parser.add_argument('--since', type=str,
                       help='Analyze articles since date (YYYY-MM-DD or "X days ago")')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of articles to process at once (default: 100)')
    parser.add_argument('--force', action='store_true',
                       help='Re-analyze articles that already have sentiment scores')
    
    # Cross-symbol analysis options
    parser.add_argument('--cross-symbol', action='store_true',
                        help='Enable cross-symbol sentiment analysi')
    parser.add_argument('--cross-window-hours', type=int, default=24,
                        help='Time window in hours for cross-symbol analysis (default: 24)')
    
    args = parser.parse_args()
    config = Config()
    logger = setup_logging(config)
    try:
        manager = SentimentFeatureManager(config)
        # Parse since date
        since_date = None
        if args.since:
            if args.since.endswith('days ago'):
                days = int(args.since.split()[0])
                since_date = datetime.now() - timedelta(days=days)
            else:
                since_date = datetime.strptime(args.since, '%Y-%m-%d')
        
        # Determine symbols to analyze
        symbols = None
        if args.market_features:    
            # Create market features and return
            results = manager.market_manager.create_market_features(
                start_date=since_date,
                window_size=args.cross_window_hours
            )
            if results:
                print(f"Articles processed: {results['articles_processed']}")
                print(f"Features created: {results['features_created']}")
                print(f"Features updated: {results['features_updated']}")
                print(f"Processing time: {results['feature_processing_time']}")
            else:
                print("No results returned from market feature creation.")
            return 0
        
        elif args.all:
            symbols = None  # Analyze all symbols
        elif args.symbols:
            symbols = args.symbols
        elif args.sector:
            symbols = get_symbols_by_sector(args.sector)
            logger.info(f"Analyzing sector '{args.sector}': {symbols}")
        
        # Run sentiment analysis
        logger.info("Starting sentiment analysis...")
        
        if args.article_ids:
            results = manager.analyze_articles(
                article_ids=args.article_ids,
                start_date=since_date,
                batch_size=args.batch_size,
                force_reanalyze=args.force,
                enable_cross_symbol=args.cross_symbol,
                window_size=args.cross_window_hours
            )
        else:
            results = manager.analyze_articles(
                symbols=symbols,
                start_date=since_date,
                batch_size=args.batch_size,
                force_reanalyze=args.force,
                enable_cross_symbol=args.cross_symbol,
                window_size=args.cross_window_hours
            )
        print_analysis_results(results)
        return 0
    
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return 1

def print_analysis_results(results):
    """Print sentiment analysis results"""
    print("\n=== SENTIMENT ANALYSIS RESULTS ===")
    print(f"Articles processed: {results['articles_processed']}")
    print(f"Articles updated: {results['articles_updated']}")
    print(f"Symbols processed: {len(results['symbols_processed'])}")
    print(f"Sentiment processing time: {results['sentiment_processing_time']:.2f} seconds")
    print(f"Feature processing time: {results['feature_processing_time']:.2f} seconds")
    
    if results.get('cross_symbol_features'):
        print("Cross-symbol features: ENABLED")
    
    if results['symbols_processed']:
        print(f"Symbols: {', '.join(results['symbols_processed'])}")

if __name__ == "__main__":
    exit(main())