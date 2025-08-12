#!/usr/bin/env python3
"""
Data collection script
Usage: python scripts/collect_data.py --symbols AAPL GOOGL --days 30
"""

import sys
import os
import argparse
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from config.symbols import get_all_symbols, get_symbols_by_sector
from src.data.pipeline import DataPipeline
from src.utils.logger import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Collect financial data')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to collect')
    parser.add_argument('--sector', help='Sector to collect (technology, finance, healthcare, energy)')
    parser.add_argument('--all', action='store_true', help='Collect all symbols')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data')
    parser.add_argument('--incremental', action='store_true', help='Incremental update (last few hours)')
    
    args = parser.parse_args()
    
    # Setup
    config = Config()
    logger = setup_logging(config)
    
    # Determine symbols to collect
    if args.all:
        symbols = get_all_symbols()
    elif args.sector:
        symbols = get_symbols_by_sector(args.sector)
    elif args.symbols:
        symbols = args.symbols
    else:
        logger.error("Must specify --symbols, --sector, or --all")
        return 1
    
    logger.info(f"Starting data collection for {len(symbols)} symbols: {symbols}")
    
    # Initialize pipeline
    pipeline = DataPipeline(config)
    
    try:
        if args.incremental:
            result = pipeline.collect_incremental_data(symbols, hours_back=2)
            logger.info(f"Incremental collection completed: {result} new articles")
        else:
            result = pipeline.collect_all_data(symbols, args.days)
            logger.info(f"Full collection completed: {result}")
        
        # Print summary
        status = pipeline.get_pipeline_status()
        print("\n=== COLLECTION SUMMARY ===")
        print(f"Symbols processed: {len(symbols)}")
        print(f"Active collectors: {status['active_collectors']}")
        
        print("\nData summary:")
        news_df = status['data_summary']['news_summary'][['symbol', 'article_count']]
        stock_df = status['data_summary']['stock_summary'][['symbol', 'price_points']]
        
        # Sanity check for symbol alignment, null points, and duplicates
        news_symbols = set(news_df['symbol'])
        stock_symbols = set(stock_df['symbol'])
        intersection = news_symbols & stock_symbols
        if len(intersection) < len(news_symbols):
            missing = news_symbols - stock_symbols
            logger.warning(
                f"{len(missing)} symbol(s) in news_summary are missing from stock_summary: {sorted(missing)}"
            )
        null_price_points = stock_df['price_points'].isna().sum()
        if null_price_points > 0:
            logger.warning(f"stock_summary has {null_price_points} rows with missing price_points.")

        if news_df['symbol'].duplicated().any():
            logger.warning(f"news_summary has duplicate symbols: {news_df[news_df['symbol'].duplicated()]['symbol'].tolist()}")
        if stock_df['symbol'].duplicated().any():
            logger.warning(f"stock_summary has duplicate symbols: {stock_df[stock_df['symbol'].duplicated()]['symbol'].tolist()}")

        # Merge after checks
        merged_summary = news_df.merge(stock_df, on='symbol', how='left').head(10)
        print(merged_summary)
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())