#!/usr/bin/env python3
"""
Data Summary CLI Script
Provides command-line access to comprehensive data summary and trend analysis.

Date format examples:
- YYYY-MM-DD: 2024-01-15
- YYYY-MM-DD HH:MM: 2024-01-15 14:30
- YYYY-MM-DD HH:MM:SS: 2024-01-15 14:30:45
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional
import io
import contextlib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from src.data.database import DatabaseManager
from src.managers.data_summary_manager import DataSummaryManager
from src.utils.logger import setup_logging

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Data Summary and Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Date Format Examples:
            YYYY-MM-DD          2024-01-15
            YYYY-MM-DD HH:MM    2024-01-15 14:30  
            YYYY-MM-DD HH:MM:SS 2024-01-15 14:30:45

            Examples:
            %(prog)s news --symbols AAPL,GOOGL --start-date 2024-01-01 --end-date 2024-12-31
            %(prog)s stock --symbols AAPL --formatted --output-file stock_summary.txt
            %(prog)s trends --symbols AAPL,GOOGL,MSFT --start-date 2024-06-01
        """
    )
    
    # Create subparsers for different analysis types
    subparsers = parser.add_subparsers(dest='command', help='Analysis type')
    subparsers.required = True
    
    # Common arguments for all subcommands
    def add_common_args(subparser):
        subparser.add_argument(
            '--symbols', '-s',
            type=str,
            default='',
            help='Comma-separated list of stock symbols (e.g., AAPL,GOOGL,MSFT). Default: all symbols'
        )
        subparser.add_argument(
            '--start-date',
            type=str,
            help='Start date (YYYY-MM-DD, YYYY-MM-DD HH:MM, or YYYY-MM-DD HH:MM:SS). Default: earliest available'
        )
        subparser.add_argument(
            '--end-date',
            type=str,
            help='End date (YYYY-MM-DD, YYYY-MM-DD HH:MM, or YYYY-MM-DD HH:MM:SS). Default: latest available'
        )
        subparser.add_argument(
            '--formatted', '-f',
            action='store_true',
            help='Use formatted table output instead of simple text'
        )
        subparser.add_argument(
            '--output-file', '-o',
            type=str,
            help='Write output to file instead of printing to console'
        )
    
    # News summary subcommand
    news_parser = subparsers.add_parser('news', help='Generate news data summary')
    add_common_args(news_parser)
    news_parser.set_defaults(func=run_news_summary)
    
    # Stock summary subcommand  
    stock_parser = subparsers.add_parser('stock', help='Generate stock data summary')
    add_common_args(stock_parser)
    stock_parser.set_defaults(func=run_stock_summary)
    
    # Sentiment summary subcommand
    sentiment_parser = subparsers.add_parser('sentiment', help='Generate sentiment features summary')
    add_common_args(sentiment_parser)
    sentiment_parser.set_defaults(func=run_sentiment_summary)
    
    # Cross-symbol summary subcommand
    cross_parser = subparsers.add_parser('cross-symbol', help='Generate cross-symbol features summary')
    add_common_args(cross_parser)
    cross_parser.set_defaults(func=run_cross_symbol_summary)
    
    # Trends analysis subcommand
    trends_parser = subparsers.add_parser('trends', help='Generate comprehensive trends analysis')
    add_common_args(trends_parser)
    trends_parser.set_defaults(func=run_trends_analysis)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize components
    try:
        print("🔄 Initializing components...")
        config = Config()
        logger = setup_logging(config)
        db_manager = DatabaseManager(config.DATABASE_PATH)
        summary_manager = DataSummaryManager(config, logger, db_manager)
        print("✅ Components initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        sys.exit(1)
    
    # Run the selected command
    try:
        args.func(args, summary_manager)
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def parse_datetime(date_string: str) -> datetime:
    """Parse datetime string in various formats"""
    if not date_string:
        return None
    
    formats = [
        '%Y-%m-%d',           # 2024-01-15
        '%Y-%m-%d %H:%M',     # 2024-01-15 14:30
        '%Y-%m-%d %H:%M:%S'   # 2024-01-15 14:30:45
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Invalid date format: {date_string}. Use YYYY-MM-DD, YYYY-MM-DD HH:MM, or YYYY-MM-DD HH:MM:SS")


def parse_symbols(symbols_string: str) -> List[str]:
    """Parse comma-separated symbols string"""
    if not symbols_string:
        return []
    return [symbol.strip().upper() for symbol in symbols_string.split(',') if symbol.strip()]


def capture_output(func, *args, **kwargs):
    """Capture function output to string"""
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        func(*args, **kwargs)
    return output_buffer.getvalue()


def write_to_file(content: str, filepath: str):
    """Write content to file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Output written to: {filepath}")
    except Exception as e:
        print(f"❌ Error writing to file {filepath}: {e}")


def run_news_summary(args, summary_manager):
    """Run news summary analysis"""
    try:
        start_date = parse_datetime(args.start_date) if args.start_date else datetime.min
        end_date = parse_datetime(args.end_date) if args.end_date else datetime.max
        symbols = parse_symbols(args.symbols)
        
        if args.output_file:
            content = capture_output(
                summary_manager.print_news_summary,
                symbols, start_date, end_date, args.formatted
            )
            write_to_file(content, args.output_file)
        else:
            summary_manager.print_news_summary(symbols, start_date, end_date, args.formatted)
            
    except Exception as e:
        print(f"❌ Error in news summary: {e}")


def run_stock_summary(args, summary_manager):
    """Run stock summary analysis"""
    try:
        start_date = parse_datetime(args.start_date) if args.start_date else datetime.min
        end_date = parse_datetime(args.end_date) if args.end_date else datetime.max
        symbols = parse_symbols(args.symbols)
        
        if args.output_file:
            content = capture_output(
                summary_manager.print_stock_summary,
                symbols, start_date, end_date, args.formatted
            )
            write_to_file(content, args.output_file)
        else:
            summary_manager.print_stock_summary(symbols, start_date, end_date, args.formatted)
            
    except Exception as e:
        print(f"❌ Error in stock summary: {e}")


def run_sentiment_summary(args, summary_manager):
    """Run sentiment summary analysis"""
    try:
        start_date = parse_datetime(args.start_date) if args.start_date else datetime.min
        end_date = parse_datetime(args.end_date) if args.end_date else datetime.max
        symbols = parse_symbols(args.symbols)
        
        if args.output_file:
            content = capture_output(
                summary_manager.print_sentiment_summary,
                symbols, start_date, end_date, args.formatted
            )
            write_to_file(content, args.output_file)
        else:
            summary_manager.print_sentiment_summary(symbols, start_date, end_date, args.formatted)
            
    except Exception as e:
        print(f"❌ Error in sentiment summary: {e}")


def run_cross_symbol_summary(args, summary_manager):
    """Run cross-symbol summary analysis"""
    try:
        start_date = parse_datetime(args.start_date) if args.start_date else datetime.min
        end_date = parse_datetime(args.end_date) if args.end_date else datetime.max
        symbols = parse_symbols(args.symbols)
        
        if args.output_file:
            content = capture_output(
                summary_manager.print_cross_symbol_summary,
                symbols, start_date, end_date, args.formatted
            )
            write_to_file(content, args.output_file)
        else:
            summary_manager.print_cross_symbol_summary(symbols, start_date, end_date, args.formatted)
            
    except Exception as e:
        print(f"❌ Error in cross-symbol summary: {e}")


def run_trends_analysis(args, summary_manager):
    """Run trends analysis"""
    try:
        start_date = parse_datetime(args.start_date) if args.start_date else datetime.min
        end_date = parse_datetime(args.end_date) if args.end_date else datetime.max
        symbols = parse_symbols(args.symbols)
        
        if args.output_file:
            content = capture_output(
                summary_manager.print_trends_analysis,
                symbols, start_date, end_date, args.formatted
            )
            write_to_file(content, args.output_file)
        else:
            summary_manager.print_trends_analysis(symbols, start_date, end_date, args.formatted)
            
    except Exception as e:
        print(f"❌ Error in trends analysis: {e}")


if __name__ == '__main__':
    main()