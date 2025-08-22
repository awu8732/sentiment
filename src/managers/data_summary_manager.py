import sys
import os
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
import logging
from tabulate import tabulate

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.database import DatabaseManager
from config.config import Config

class DataSummaryManager:
    """Handles comprehensive data summary output for news, stock, sentiment, and cross-symbol data"""
    
    def __init__(self, config: Config, logger: logging, db_manager: DatabaseManager):
        self.config = config
        self.logger = logger
        self.db_manager = db_manager
    
    def print_news_summary(self, symbols: List[str], start_date: datetime, end_date: datetime, formatted: bool = False):
        """Print comprehensive news data summary"""
        print("=" * 80)
        print("NEWS DATA SUMMARY")
        print("=" * 80)
        print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Target Symbols: {', '.join(symbols) if symbols else 'All'}")
        print("-" * 80)
        
        try:
            # Get data for specified symbols
            symbol_data = []
            for symbol in symbols:
                df = self.db_manager.get_news_data(symbol=symbol, start_date=start_date, end_date=end_date)
                if not df.empty:
                    symbol_data.append(df)
            
            # Get overall data for comparison
            overall_df = self.db_manager.get_news_data(start_date=start_date, end_date=end_date)
            
            if overall_df.empty:
                print("❌ No news data found for the specified date range.")
                return
            
            # Symbol-specific analysis
            if symbol_data:
                combined_symbol_df = pd.concat(symbol_data, ignore_index=True)
                print(f"\n📊 SYMBOL-SPECIFIC ANALYSIS ({len(symbols)} symbols)")
                self._print_news_statistics(combined_symbol_df, "Selected Symbols", formatted)
                
                # Individual symbol breakdown
                for symbol in symbols:
                    symbol_df = self.db_manager.get_news_data(symbol=symbol, start_date=start_date, end_date=end_date)
                    if not symbol_df.empty:
                        print(f"\n📈 {symbol} Breakdown:")
                        self._print_news_statistics(symbol_df, symbol, formatted, detailed=False)
                    else:
                        print(f"\n❌ No data found for {symbol}")
            else:
                print("❌ No data found for any of the specified symbols.")
            
            # Overall market analysis
            print(f"\n🌍 OVERALL MARKET ANALYSIS")
            self._print_news_statistics(overall_df, "Market-Wide", formatted)
            
            # Comparative analysis if we have symbol data
            if symbol_data:
                self._print_news_comparison(combined_symbol_df, overall_df, formatted)
                
        except Exception as e:
            self.logger.error(f"Error in news summary: {e}")
            print(f"❌ Error generating news summary: {e}")
    
    def print_stock_summary(self, symbols: List[str], start_date: datetime, end_date: datetime, formatted: bool = False):
        """Print comprehensive stock data summary"""
        print("=" * 80)
        print("STOCK DATA SUMMARY")
        print("=" * 80)
        print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Target Symbols: {', '.join(symbols) if symbols else 'All'}")
        print("-" * 80)
        
        try:
            # Get data for specified symbols
            symbol_data = []
            for symbol in symbols:
                df = self.db_manager.get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
                if not df.empty:
                    symbol_data.append(df)
            
            # Get overall data for comparison
            overall_df = self.db_manager.get_stock_data(start_date=start_date, end_date=end_date)
            
            if overall_df.empty:
                print("❌ No stock data found for the specified date range.")
                return
            
            # Symbol-specific analysis
            if symbol_data:
                combined_symbol_df = pd.concat(symbol_data, ignore_index=True)
                print(f"\n📊 SYMBOL-SPECIFIC ANALYSIS ({len(symbols)} symbols)")
                self._print_stock_statistics(combined_symbol_df, "Selected Symbols", formatted)
                
                # Individual symbol breakdown
                for symbol in symbols:
                    symbol_df = self.db_manager.get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
                    if not symbol_df.empty:
                        print(f"\n📈 {symbol} Breakdown:")
                        self._print_stock_statistics(symbol_df, symbol, formatted, detailed=False)
                    else:
                        print(f"\n❌ No data found for {symbol}")
            else:
                print("❌ No data found for any of the specified symbols.")
            
            # Overall market analysis
            print(f"\n🌍 OVERALL MARKET ANALYSIS")
            self._print_stock_statistics(overall_df, "Market-Wide", formatted)
            
            # Comparative analysis if we have symbol data
            if symbol_data:
                self._print_stock_comparison(combined_symbol_df, overall_df, formatted)
                
        except Exception as e:
            self.logger.error(f"Error in stock summary: {e}")
            print(f"❌ Error generating stock summary: {e}")
    
    def print_sentiment_summary(self, symbols: List[str], start_date: datetime, end_date: datetime, formatted: bool = False):
        """Print comprehensive sentiment features summary"""
        print("=" * 80)
        print("SENTIMENT FEATURES SUMMARY")
        print("=" * 80)
        print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Target Symbols: {', '.join(symbols) if symbols else 'All'}")
        print("-" * 80)
        
        try:
            # Get data for specified symbols
            symbol_data = []
            for symbol in symbols:
                df = self.db_manager.get_sentiment_features_data(symbol=symbol, start_date=start_date, end_date=end_date)
                if not df.empty:
                    symbol_data.append(df)
            
            # Get overall data for comparison
            overall_df = self.db_manager.get_sentiment_features_data(start_date=start_date, end_date=end_date)
            
            if overall_df.empty:
                print("❌ No sentiment features data found for the specified date range.")
                return
            
            # Symbol-specific analysis
            if symbol_data:
                combined_symbol_df = pd.concat(symbol_data, ignore_index=True)
                print(f"\n📊 SYMBOL-SPECIFIC ANALYSIS ({len(symbols)} symbols)")
                self._print_sentiment_statistics(combined_symbol_df, "Selected Symbols", formatted)
                
                # Individual symbol breakdown
                for symbol in symbols:
                    symbol_df = self.db_manager.get_sentiment_features_data(symbol=symbol, start_date=start_date, end_date=end_date)
                    if not symbol_df.empty:
                        print(f"\n📈 {symbol} Breakdown:")
                        self._print_sentiment_statistics(symbol_df, symbol, formatted, detailed=False)
                    else:
                        print(f"\n❌ No data found for {symbol}")
            else:
                print("❌ No data found for any of the specified symbols.")
            
            # Overall market analysis
            print(f"\n🌍 OVERALL MARKET ANALYSIS")
            self._print_sentiment_statistics(overall_df, "Market-Wide", formatted)
            
            # Comparative analysis if we have symbol data
            if symbol_data:
                self._print_sentiment_comparison(combined_symbol_df, overall_df, formatted)
                
        except Exception as e:
            self.logger.error(f"Error in sentiment summary: {e}")
            print(f"❌ Error generating sentiment summary: {e}")
    
    def print_cross_symbol_summary(self, symbols: List[str], start_date: datetime, end_date: datetime, formatted: bool = False):
        """Print comprehensive cross-symbol features summary"""
        print("=" * 80)
        print("CROSS-SYMBOL FEATURES SUMMARY")
        print("=" * 80)
        print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Target Symbols: {', '.join(symbols) if symbols else 'All'}")
        print("-" * 80)
        
        try:
            # Get data for specified symbols
            symbol_data = []
            for symbol in symbols:
                df = self.db_manager.get_cross_symbol_features_data(symbol=symbol, start_date=start_date, end_date=end_date)
                if not df.empty:
                    symbol_data.append(df)
            
            # Get overall data for comparison
            overall_df = self.db_manager.get_cross_symbol_features_data(start_date=start_date, end_date=end_date)
            
            if overall_df.empty:
                print("❌ No cross-symbol features data found for the specified date range.")
                return
            
            # Symbol-specific analysis
            if symbol_data:
                combined_symbol_df = pd.concat(symbol_data, ignore_index=True)
                print(f"\n📊 SYMBOL-SPECIFIC ANALYSIS ({len(symbols)} symbols)")
                self._print_cross_symbol_statistics(combined_symbol_df, "Selected Symbols", formatted)
                
                # Sector analysis for selected symbols
                if 'sector' in combined_symbol_df.columns:
                    sectors = combined_symbol_df['sector'].unique()
                    print(f"\n🏢 SECTOR BREAKDOWN:")
                    for sector in sectors:
                        if pd.notna(sector):
                            sector_df = combined_symbol_df[combined_symbol_df['sector'] == sector]
                            print(f"\n📊 {sector} Sector:")
                            self._print_cross_symbol_statistics(sector_df, sector, formatted, detailed=False)
                
                # Individual symbol breakdown
                for symbol in symbols:
                    symbol_df = self.db_manager.get_cross_symbol_features_data(symbol=symbol, start_date=start_date, end_date=end_date)
                    if not symbol_df.empty:
                        print(f"\n📈 {symbol} Breakdown:")
                        self._print_cross_symbol_statistics(symbol_df, symbol, formatted, detailed=False)
                    else:
                        print(f"\n❌ No data found for {symbol}")
            else:
                print("❌ No data found for any of the specified symbols.")
            
            # Overall market analysis
            print(f"\n🌍 OVERALL MARKET ANALYSIS")
            self._print_cross_symbol_statistics(overall_df, "Market-Wide", formatted)
            
            # Sector analysis for all data
            if 'sector' in overall_df.columns:
                sectors = overall_df['sector'].unique()
                print(f"\n🏢 ALL SECTORS BREAKDOWN:")
                for sector in sectors:
                    if pd.notna(sector):
                        sector_df = overall_df[overall_df['sector'] == sector]
                        print(f"\n📊 {sector} Sector (Market-Wide):")
                        self._print_cross_symbol_statistics(sector_df, f"{sector} Sector", formatted, detailed=False)
            
            # Comparative analysis if we have symbol data
            if symbol_data:
                self._print_cross_symbol_comparison(combined_symbol_df, overall_df, formatted)
                
        except Exception as e:
            self.logger.error(f"Error in cross-symbol summary: {e}")
            print(f"❌ Error generating cross-symbol summary: {e}")
    
    def _print_news_statistics(self, df: pd.DataFrame, label: str, formatted: bool, detailed: bool = True):
        """Print detailed news statistics"""
        if df.empty:
            print(f"❌ No data available for {label}")
            return
        
        # Basic stats
        stats = []
        stats.append(["Total Articles", len(df)])
        stats.append(["Unique Sources", df['source'].nunique() if 'source' in df.columns else 'N/A'])
        stats.append(["Unique Symbols", df['symbol'].nunique() if 'symbol' in df.columns else 'N/A'])
        
        if detailed:
            # Date range
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                stats.append(["Date Range", f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"])
            
            # Sentiment statistics
            if 'sentiment_score' in df.columns and df['sentiment_score'].notna().any():
                sentiment_stats = df['sentiment_score'].describe()
                stats.append(["Avg Sentiment", f"{sentiment_stats['mean']:.4f}"])
                stats.append(["Sentiment Std", f"{sentiment_stats['std']:.4f}"])
                stats.append(["Min Sentiment", f"{sentiment_stats['min']:.4f}"])
                stats.append(["Max Sentiment", f"{sentiment_stats['max']:.4f}"])
                
                # Sentiment distribution
                positive = (df['sentiment_score'] > 0.1).sum()
                negative = (df['sentiment_score'] < -0.1).sum()
                neutral = len(df) - positive - negative
                stats.append(["Positive Articles", f"{positive} ({positive/len(df)*100:.1f}%)"])
                stats.append(["Negative Articles", f"{negative} ({negative/len(df)*100:.1f}%)"])
                stats.append(["Neutral Articles", f"{neutral} ({neutral/len(df)*100:.1f}%)"])
        
        if formatted:
            print(f"\n{label}:")
            print(tabulate(stats, headers=["Metric", "Value"], tablefmt="grid"))
        else:
            print(f"\n{label}:")
            for stat in stats:
                print(f"  {stat[0]}: {stat[1]}")
    
    def _print_stock_statistics(self, df: pd.DataFrame, label: str, formatted: bool, detailed: bool = True):
        """Print detailed stock statistics"""
        if df.empty:
            print(f"❌ No data available for {label}")
            return
        
        stats = []
        stats.append(["Total Records", len(df)])
        stats.append(["Unique Symbols", df['symbol'].nunique() if 'symbol' in df.columns else 'N/A'])
        
        if detailed:
            # Date range
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                stats.append(["Date Range", f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"])
            
            # Price statistics
            numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            for col in numeric_cols:
                if col in df.columns and df[col].notna().any():
                    col_stats = df[col].describe()
                    stats.append([f"Avg {col.title()}", f"{col_stats['mean']:.2f}" if col != 'volume' else f"{col_stats['mean']:,.0f}"])
                    if detailed:
                        stats.append([f"{col.title()} Range", f"{col_stats['min']:.2f} - {col_stats['max']:.2f}" if col != 'volume' else f"{col_stats['min']:,.0f} - {col_stats['max']:,.0f}"])
            
            # Calculate returns if we have close prices
            if 'close' in df.columns and len(df) > 1:
                df_sorted = df.sort_values('timestamp')
                returns = df_sorted['close'].pct_change().dropna()
                if not returns.empty:
                    stats.append(["Avg Daily Return", f"{returns.mean()*100:.2f}%"])
                    stats.append(["Return Volatility", f"{returns.std()*100:.2f}%"])
                    stats.append(["Max Daily Gain", f"{returns.max()*100:.2f}%"])
                    stats.append(["Max Daily Loss", f"{returns.min()*100:.2f}%"])
        
        if formatted:
            print(f"\n{label}:")
            print(tabulate(stats, headers=["Metric", "Value"], tablefmt="grid"))
        else:
            print(f"\n{label}:")
            for stat in stats:
                print(f"  {stat[0]}: {stat[1]}")
    
    def _print_sentiment_statistics(self, df: pd.DataFrame, label: str, formatted: bool, detailed: bool = True):
        """Print detailed sentiment statistics"""
        if df.empty:
            print(f"❌ No data available for {label}")
            return
        
        stats = []
        stats.append(["Total Records", len(df)])
        stats.append(["Unique Symbols", df['symbol'].nunique() if 'symbol' in df.columns else 'N/A'])
        
        if detailed:
            # Date range
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                stats.append(["Date Range", f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"])
            
            # Sentiment feature statistics
            sentiment_cols = ['sentiment_score', 'sentiment_skew', 'sentiment_std', 'sentiment_momentum', 
                            'extreme_sentiment_ratio', 'sentiment_persistence', 'news_flow_intensity', 
                            'news_volume', 'source_diversity']
            
            for col in sentiment_cols:
                if col in df.columns and df[col].notna().any():
                    col_stats = df[col].describe()
                    if col == 'news_volume':
                        stats.append([f"Avg {col.replace('_', ' ').title()}", f"{col_stats['mean']:.0f}"])
                    else:
                        stats.append([f"Avg {col.replace('_', ' ').title()}", f"{col_stats['mean']:.4f}"])
                    
                    if detailed:
                        if col == 'news_volume':
                            stats.append([f"{col.replace('_', ' ').title()} Range", f"{col_stats['min']:.0f} - {col_stats['max']:.0f}"])
                        else:
                            stats.append([f"{col.replace('_', ' ').title()} Range", f"{col_stats['min']:.4f} - {col_stats['max']:.4f}"])
        
        if formatted:
            print(f"\n{label}:")
            print(tabulate(stats, headers=["Metric", "Value"], tablefmt="grid"))
        else:
            print(f"\n{label}:")
            for stat in stats:
                print(f"  {stat[0]}: {stat[1]}")
    
    def _print_cross_symbol_statistics(self, df: pd.DataFrame, label: str, formatted: bool, detailed: bool = True):
        """Print detailed cross-symbol statistics"""
        if df.empty:
            print(f"❌ No data available for {label}")
            return
        
        stats = []
        stats.append(["Total Records", len(df)])
        stats.append(["Unique Symbols", df['symbol'].nunique() if 'symbol' in df.columns else 'N/A'])
        stats.append(["Unique Sectors", df['sector'].nunique() if 'sector' in df.columns else 'N/A'])
        
        if detailed:
            # Date range
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                stats.append(["Date Range", f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"])
            
            # Cross-symbol feature statistics
            cross_cols = ['sector_sentiment_mean', 'sector_sentiment_skew', 'sector_sentiment_std',
                         'sector_news_volume', 'relative_sentiment_ratio', 'sector_sentiment_correlation',
                         'sector_sentiment_divergence', 'market_sentiment_correlation', 'market_sentiment_divergence']
            
            for col in cross_cols:
                if col in df.columns and df[col].notna().any():
                    col_stats = df[col].describe()
                    if 'volume' in col:
                        stats.append([f"Avg {col.replace('_', ' ').title()}", f"{col_stats['mean']:.0f}"])
                    else:
                        stats.append([f"Avg {col.replace('_', ' ').title()}", f"{col_stats['mean']:.4f}"])
                    
                    if detailed:
                        if 'volume' in col:
                            stats.append([f"{col.replace('_', ' ').title()} Range", f"{col_stats['min']:.0f} - {col_stats['max']:.0f}"])
                        else:
                            stats.append([f"{col.replace('_', ' ').title()} Range", f"{col_stats['min']:.4f} - {col_stats['max']:.4f}"])
        
        if formatted:
            print(f"\n{label}:")
            print(tabulate(stats, headers=["Metric", "Value"], tablefmt="grid"))
        else:
            print(f"\n{label}:")
            for stat in stats:
                print(f"  {stat[0]}: {stat[1]}")
    
    def _print_news_comparison(self, symbol_df: pd.DataFrame, overall_df: pd.DataFrame, formatted: bool):
        """Print comparison between symbol-specific and overall news data"""
        print(f"\n🔍 COMPARATIVE ANALYSIS")
        
        comparisons = []
        
        # Article count comparison
        symbol_count = len(symbol_df)
        total_count = len(overall_df)
        comparisons.append(["Market Share", f"{symbol_count}/{total_count} ({symbol_count/total_count*100:.1f}%)"])
        
        # Sentiment comparison
        if 'sentiment_score' in symbol_df.columns and 'sentiment_score' in overall_df.columns:
            symbol_sentiment = symbol_df['sentiment_score'].mean()
            overall_sentiment = overall_df['sentiment_score'].mean()
            diff = symbol_sentiment - overall_sentiment
            comparisons.append(["Sentiment vs Market", f"{symbol_sentiment:.4f} vs {overall_sentiment:.4f} ({diff:+.4f})"])
        
        # Source diversity comparison
        if 'source' in symbol_df.columns and 'source' in overall_df.columns:
            symbol_sources = symbol_df['source'].nunique()
            total_sources = overall_df['source'].nunique()
            comparisons.append(["Source Coverage", f"{symbol_sources}/{total_sources} ({symbol_sources/total_sources*100:.1f}%)"])
        
        if formatted:
            print(tabulate(comparisons, headers=["Metric", "Value"], tablefmt="grid"))
        else:
            for comp in comparisons:
                print(f"  {comp[0]}: {comp[1]}")
    
    def _print_stock_comparison(self, symbol_df: pd.DataFrame, overall_df: pd.DataFrame, formatted: bool):
        """Print comparison between symbol-specific and overall stock data"""
        print(f"\n🔍 COMPARATIVE ANALYSIS")
        
        comparisons = []
        
        # Record count comparison
        symbol_count = len(symbol_df)
        total_count = len(overall_df)
        comparisons.append(["Market Share", f"{symbol_count}/{total_count} ({symbol_count/total_count*100:.1f}%)"])
        
        # Price comparison
        if 'close' in symbol_df.columns and 'close' in overall_df.columns:
            symbol_avg_price = symbol_df['close'].mean()
            overall_avg_price = overall_df['close'].mean()
            comparisons.append(["Avg Price vs Market", f"${symbol_avg_price:.2f} vs ${overall_avg_price:.2f}"])
        
        # Volume comparison
        if 'volume' in symbol_df.columns and 'volume' in overall_df.columns:
            symbol_avg_volume = symbol_df['volume'].mean()
            overall_avg_volume = overall_df['volume'].mean()
            comparisons.append(["Avg Volume vs Market", f"{symbol_avg_volume:,.0f} vs {overall_avg_volume:,.0f}"])
        
        if formatted:
            print(tabulate(comparisons, headers=["Metric", "Value"], tablefmt="grid"))
        else:
            for comp in comparisons:
                print(f"  {comp[0]}: {comp[1]}")
    
    def _print_sentiment_comparison(self, symbol_df: pd.DataFrame, overall_df: pd.DataFrame, formatted: bool):
        """Print comparison between symbol-specific and overall sentiment data"""
        print(f"\n🔍 COMPARATIVE ANALYSIS")
        
        comparisons = []
        
        # Record count comparison
        symbol_count = len(symbol_df)
        total_count = len(overall_df)
        comparisons.append(["Market Share", f"{symbol_count}/{total_count} ({symbol_count/total_count*100:.1f}%)"])
        
        # Key sentiment metrics comparison
        sentiment_cols = ['sentiment_score', 'sentiment_momentum', 'news_flow_intensity']
        for col in sentiment_cols:
            if col in symbol_df.columns and col in overall_df.columns:
                symbol_val = symbol_df[col].mean()
                overall_val = overall_df[col].mean()
                diff = symbol_val - overall_val
                comparisons.append([f"{col.replace('_', ' ').title()} vs Market", f"{symbol_val:.4f} vs {overall_val:.4f} ({diff:+.4f})"])
        
        if formatted:
            print(tabulate(comparisons, headers=["Metric", "Value"], tablefmt="grid"))
        else:
            for comp in comparisons:
                print(f"  {comp[0]}: {comp[1]}")
    
    def _print_cross_symbol_comparison(self, symbol_df: pd.DataFrame, overall_df: pd.DataFrame, formatted: bool):
        """Print comparison between symbol-specific and overall cross-symbol data"""
        print(f"\n🔍 COMPARATIVE ANALYSIS")
        
        comparisons = []
        
        # Record count comparison
        symbol_count = len(symbol_df)
        total_count = len(overall_df)
        comparisons.append(["Market Share", f"{symbol_count}/{total_count} ({symbol_count/total_count*100:.1f}%)"])
        
        # Key cross-symbol metrics comparison
        cross_cols = ['relative_sentiment_ratio', 'sector_sentiment_correlation', 'market_sentiment_correlation']
        for col in cross_cols:
            if col in symbol_df.columns and col in overall_df.columns:
                symbol_val = symbol_df[col].mean()
                overall_val = overall_df[col].mean()
                diff = symbol_val - overall_val
                comparisons.append([f"{col.replace('_', ' ').title()} vs Market", f"{symbol_val:.4f} vs {overall_val:.4f} ({diff:+.4f})"])
        
        if formatted:
            print(tabulate(comparisons, headers=["Metric", "Value"], tablefmt="grid"))
        else:
            for comp in comparisons:
                print(f"  {comp[0]}: {comp[1]}")