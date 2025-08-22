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
    selection_group.add_argument('--summary', action='store_true',
                               help='Show sentiment analysis summary only')
    selection_group.add_argument('--cross-insights', action='store_true',
                               help='Show cross-symbol sentiment insights')
    
    # Additional filters
    parser.add_argument('--since', type=str,
                       help='Analyze articles since date (YYYY-MM-DD or "X days ago")')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of articles to process at once (default: 100)')
    parser.add_argument('--force', action='store_true',
                       help='Re-analyze articles that already have sentiment scores')

    # Analysis options
    parser.add_argument('--trends', action='store_true',
                       help='Show sentiment trends over time')
    parser.add_argument('--days-back', type=int, default=30,
                       help='Days back for trend analysis (default: 30)')
    
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
        if args.symbols:
            symbols = args.symbols
        elif args.sector:
            symbols = get_symbols_by_sector(args.sector)
            logger.info(f"Analyzing sector '{args.sector}': {symbols}")
        elif args.all:
            symbols = None  # Analyze all symbols
        elif args.cross_insights:
            symbols = args.symbols if args.symbols else get_all_symbols()[:10]  # Default to top 10
        
        # Run sentiment analysis
        logger.info("Starting sentiment analysis...")
        
        if args.article_ids:
            results = manager.analyze_articles(
                article_ids=args.article_ids,
                batch_size=args.batch_size,
                force_reanalyze=args.force,
                enable_cross_symbol=args.cross_symbol,
                cross_symbol_window=args.cross_window_hours
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
    
def show_summary(runner, symbols, include_cross_symbol):
    """Show sentiment analysis summary"""
    print("\n=== SENTIMENT ANALYSIS SUMMARY ===")
    summary_df = runner.get_sentiment_summary(symbols, include_cross_symbol=include_cross_symbol)
    
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
        
        # Overall statistics
        total_articles = summary_df['total_articles'].sum()
        analyzed_articles = summary_df['analyzed_articles'].sum()
        avg_progress = summary_df['analysis_progress'].mean()
        
        print(f"\nOverall Statistics:")
        print(f"Total articles: {total_articles}")
        print(f"Analyzed articles: {analyzed_articles}")
        print(f"Average analysis progress: {avg_progress:.1f}%")
        
        if include_cross_symbol and 'cross_symbol_records' in summary_df.columns:
            cross_symbol_records = summary_df['cross_symbol_records'].sum()
            print(f"Cross-symbol feature records: {cross_symbol_records}")
    else:
        print("No articles found matching criteria")
    
    return 0

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

def show_sentiment_trends(runner, symbols, days_back, include_cross_symbol):
    """Show sentiment trends over time"""
    print("\n=== SENTIMENT TRENDS ===")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    trends_df = runner.analyze_by_time_period(
        symbols, start_date, end_date, include_cross_symbol=include_cross_symbol
    )
    
    if not trends_df.empty:
        # Show recent trends (last 7 days per symbol)
        recent_trends = trends_df.groupby('symbol').tail(7)
        
        display_columns = ['symbol', 'date', 'article_count', 'avg_sentiment', 
                          'positive_count', 'negative_count']
        
        if include_cross_symbol:
            cross_columns = ['avg_sector_sentiment', 'avg_relative_strength']
            display_columns.extend([col for col in cross_columns if col in recent_trends.columns])
        
        print(recent_trends[display_columns].to_string(index=False))
    else:
        print("No trend data available for the specified period")

if __name__ == "__main__":
    exit(main())