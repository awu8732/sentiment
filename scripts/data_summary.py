#!/usr/bin/env python3
"""
Data summary and validation script
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from src.data.database import DatabaseManager
from src.utils.helpers import validate_data_quality
from src.utils.logger import setup_logging

def main():
    config = Config()
    logger = setup_logging(config)
    
    db_manager = DatabaseManager(config.DATABASE_PATH)
    
    print("=== FINANCIAL DATA SUMMARY ===\n")
    
    # Get overall summary
    summary = db_manager.get_data_summary()
    
    print("NEWS DATA SUMMARY:")
    print(summary['news_summary'])
    print(f"\nTotal articles: {summary['news_summary']['article_count'].sum()}")
    
    print("\n" + "="*50 + "\n")
    
    print("STOCK DATA SUMMARY:")
    print(summary['stock_summary'])
    print(f"\nTotal price points: {summary['stock_summary']['price_points'].sum()}")
    
    # Data quality validation
    print("\n" + "="*50 + "\n")
    print("DATA QUALITY VALIDATION:")
    
    # Check news data quality
    news_df = db_manager.get_news_data()
    if not news_df.empty:
        news_quality = validate_data_quality(news_df, ['timestamp', 'title', 'symbol'])
        print(f"\nNews data quality:")
        print(f"- Total records: {news_quality['total_rows']}")
        print(f"- Duplicate records: {news_quality['duplicate_rows']}")
        print(f"- Date range: {news_quality['date_range'].get('start', 'N/A')} to {news_quality['date_range'].get('end', 'N/A')}")
        if news_quality['issues']:
            print(f"- Issues: {news_quality['issues']}")
    
    # Check stock data quality
    stock_df = db_manager.get_stock_data()
    if not stock_df.empty:
        stock_quality = validate_data_quality(stock_df, ['timestamp', 'symbol', 'close'])
        print(f"\nStock data quality:")
        print(f"- Total records: {stock_quality['total_rows']}")
        print(f"- Duplicate records: {stock_quality['duplicate_rows']}")
        print(f"- Date range: {stock_quality['date_range'].get('start', 'N/A')} to {stock_quality['date_range'].get('end', 'N/A')}")
        if stock_quality['issues']:
            print(f"- Issues: {stock_quality['issues']}")
    
    # Recent activity
    print("\n" + "="*50 + "\n")
    print("RECENT ACTIVITY (Last 24 hours):")
    
    from datetime import datetime, timedelta
    yesterday = datetime.now() - timedelta(days=1)
    
    recent_news = db_manager.get_news_data(start_date=yesterday)
    recent_articles_count = len(recent_news)
    
    print(f"- New articles collected: {recent_articles_count}")
    
    if recent_articles_count > 0:
        print("- Top sources:")
        source_counts = recent_news['source'].value_counts().head(5)
        for source, count in source_counts.items():
            print(f"  * {source}: {count} articles")

if __name__ == "__main__":
    main()
