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
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from config.symbols import get_all_symbols, get_symbols_by_sector
from src.utils.logger import setup_logging
from src.sentiment.sentiment_analysis_runner import SentimentAnalysisRunner

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
    
    args = parser.parse_args()
    config = Config()
    logger = setup_logging(config)
    try:
        runner = SentimentAnalysisRunner(config)
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
        
        # Show summary if requested (No analysis executed)
        if args.summary:
            print("\n=== SENTIMENT ANALYSIS SUMMARY ===")
            summary_df = runner.get_sentiment_summary(symbols)
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
            else:
                print("No articles found matching criteria")
            return 0
        
        # Run sentiment analysis
        logger.info("Starting sentiment analysis...")
        
        if args.article_ids:
            results = runner.analyze_articles(
                article_ids=args.article_ids,
                batch_size=args.batch_size,
                force_reanalyze=args.force
            )
        else:
            results = runner.analyze_articles(
                symbols=symbols,
                since_date=since_date,
                batch_size=args.batch_size,
                force_reanalyze=args.force
            )
        
        # Print results
        print("\n=== SENTIMENT ANALYSIS RESULTS ===")
        print(f"Articles processed: {results['articles_processed']}")
        print(f"Articles updated: {results['articles_updated']}")
        print(f"Symbols processed: {len(results['symbols_processed'])}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        
        if results['symbols_processed']:
            print(f"Symbols: {', '.join(results['symbols_processed'])}")
        
        # Show updated summary
        if results['articles_updated'] > 0:
            print("\n=== UPDATED SENTIMENT SUMMARY ===")
            summary_df = runner.get_sentiment_summary(results['symbols_processed'])
            if not summary_df.empty:
                print(summary_df[['symbol', 'analyzed_articles', 'avg_sentiment', 
                                'positive_articles', 'negative_articles', 'analysis_progress']].to_string(index=False))
        
        # Show trends if requested
        if args.trends and symbols:
            print("\n=== SENTIMENT TRENDS ===")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days_back)
            
            trends_df = runner.analyze_by_time_period(symbols, start_date, end_date)
            if not trends_df.empty:
                recent_trends = trends_df.groupby('symbol').tail(7)  # Last 7 days per symbol
                print(recent_trends.to_string(index=False))
            else:
                print("No trend data available for the specified period")
        
        logger.info("Sentiment analysis completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())