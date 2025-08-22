import sys
import os
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
import logging
from tabulate import tabulate
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.database import DatabaseManager
from config.config import Config

class DataSummaryManager:
    """Handles comprehensive data summary output for news, stock, sentiment, and cross-symbol data"""
    
    def __init__(self, config: Config, logger: logging.Logger, db_manager: DatabaseManager):
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

    def print_trends_analysis(self, symbols: List[str], start_date: datetime, end_date: datetime, formatted: bool = False):
        """Print comprehensive trends analysis for all data types"""
        print("=" * 80)
        print("TRENDS ANALYSIS")
        print("=" * 80)
        print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Target Symbols: {', '.join(symbols) if symbols else 'All'}")
        print("-" * 80)
        
        # Check minimum date range requirement
        date_diff = (end_date - start_date).days
        if date_diff < 7:
            print(f"❌ Insufficient date range for trend analysis. Minimum 7 days required, got {date_diff} days.")
            return
        
        try:
            # News trends
            print("\n📰 NEWS TRENDS")
            self._analyze_news_trends(symbols, start_date, end_date, formatted)
            
            # Stock trends
            print("\n📈 STOCK TRENDS")
            self._analyze_stock_trends(symbols, start_date, end_date, formatted)
            
            # Sentiment trends
            print("\n😊 SENTIMENT TRENDS")
            self._analyze_sentiment_trends(symbols, start_date, end_date, formatted)
            
            # Cross-symbol trends
            print("\n🔄 CROSS-SYMBOL TRENDS")
            self._analyze_cross_symbol_trends(symbols, start_date, end_date, formatted)
            
        except Exception as e:
            self.logger.error(f"Error in trends analysis: {e}")
            print(f"❌ Error generating trends analysis: {e}")
    
    def _analyze_news_trends(self, symbols: List[str], start_date: datetime, end_date: datetime, formatted: bool):
        """Analyze trends in news data"""
        try:
            # Get data for specified symbols
            symbol_data = []
            for symbol in symbols:
                df = self.db_manager.get_news_data(symbol=symbol, start_date=start_date, end_date=end_date)
                if not df.empty:
                    df['symbol'] = symbol
                    symbol_data.append(df)
            
            if not symbol_data:
                print("❌ No news data available for trend analysis")
                return
            
            combined_df = pd.concat(symbol_data, ignore_index=True)
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
            
            # Daily aggregations
            daily_news = combined_df.groupby(combined_df['timestamp'].dt.date).agg({
                'id': 'count',  # article count
                'sentiment_score': 'mean',  # average sentiment
                'source': 'nunique'  # source diversity
            }).rename(columns={'id': 'article_count', 'source': 'source_count'})
            
            if len(daily_news) < 7:
                print("❌ Insufficient daily data points for news trend analysis")
                return
            
            trends = []
            
            # Article volume trend
            x = np.arange(len(daily_news))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, daily_news['article_count'])
            trend_indicator = self._get_trend_indicator(slope)
            pct_change = ((daily_news['article_count'].iloc[-1] - daily_news['article_count'].iloc[0]) / daily_news['article_count'].iloc[0]) * 100
            trends.append(["Article Volume", f"{trend_indicator} {slope:.2f}/day", f"{pct_change:+.1f}%", f"R²={r_value**2:.3f}"])
            
            # Sentiment trend
            if daily_news['sentiment_score'].notna().sum() >= 7:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, daily_news['sentiment_score'].fillna(0))
                trend_indicator = self._get_trend_indicator(slope)
                pct_change = ((daily_news['sentiment_score'].iloc[-1] - daily_news['sentiment_score'].iloc[0]) / abs(daily_news['sentiment_score'].iloc[0])) * 100 if daily_news['sentiment_score'].iloc[0] != 0 else 0
                trends.append(["Sentiment Score", f"{trend_indicator} {slope:.4f}/day", f"{pct_change:+.1f}%", f"R²={r_value**2:.3f}"])
            
            # Source diversity trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, daily_news['source_count'])
            trend_indicator = self._get_trend_indicator(slope)
            pct_change = ((daily_news['source_count'].iloc[-1] - daily_news['source_count'].iloc[0]) / daily_news['source_count'].iloc[0]) * 100
            trends.append(["Source Diversity", f"{trend_indicator} {slope:.2f}/day", f"{pct_change:+.1f}%", f"R²={r_value**2:.3f}"])
            
            if formatted:
                print(tabulate(trends, headers=["Metric", "Trend", "Total Change", "Correlation"], tablefmt="grid"))
            else:
                for trend in trends:
                    print(f"  {trend[0]}: {trend[1]} ({trend[2]}) - {trend[3]}")
                    
        except Exception as e:
            self.logger.error(f"Error in news trends analysis: {e}")
            print(f"❌ Error analyzing news trends: {e}")
    
    def _analyze_stock_trends(self, symbols: List[str], start_date: datetime, end_date: datetime, formatted: bool):
        """Analyze trends in stock data"""
        try:
            trends_by_symbol = []
            
            for symbol in symbols:
                df = self.db_manager.get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
                if df.empty:
                    continue
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                if len(df) < 7:
                    continue
                
                x = np.arange(len(df))
                symbol_trends = []
                
                # Price trend
                if 'close' in df.columns:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, df['close'])
                    trend_indicator = self._get_trend_indicator(slope)
                    pct_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
                    symbol_trends.append([f"{symbol} Price", f"{trend_indicator} ${slope:.2f}/day", f"{pct_change:+.1f}%", f"R²={r_value**2:.3f}"])
                
                # Volume trend
                if 'volume' in df.columns:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, df['volume'])
                    trend_indicator = self._get_trend_indicator(slope)
                    pct_change = ((df['volume'].iloc[-1] - df['volume'].iloc[0]) / df['volume'].iloc[0]) * 100
                    symbol_trends.append([f"{symbol} Volume", f"{trend_indicator} {slope:,.0f}/day", f"{pct_change:+.1f}%", f"R²={r_value**2:.3f}"])
                
                # Volatility trend (using daily returns)
                if 'close' in df.columns and len(df) > 1:
                    df['returns'] = df['close'].pct_change()
                    df['volatility'] = df['returns'].rolling(window=5, min_periods=1).std()
                    if df['volatility'].notna().sum() >= 7:
                        vol_data = df['volatility'].dropna()
                        x_vol = np.arange(len(vol_data))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vol, vol_data)
                        trend_indicator = self._get_trend_indicator(slope)
                        pct_change = ((vol_data.iloc[-1] - vol_data.iloc[0]) / vol_data.iloc[0]) * 100 if vol_data.iloc[0] != 0 else 0
                        symbol_trends.append([f"{symbol} Volatility", f"{trend_indicator} {slope:.4f}/day", f"{pct_change:+.1f}%", f"R²={r_value**2:.3f}"])
                
                trends_by_symbol.extend(symbol_trends)
            
            if not trends_by_symbol:
                print("❌ No stock data available for trend analysis")
                return
            
            if formatted:
                print(tabulate(trends_by_symbol, headers=["Metric", "Trend", "Total Change", "Correlation"], tablefmt="grid"))
            else:
                for trend in trends_by_symbol:
                    print(f"  {trend[0]}: {trend[1]} ({trend[2]}) - {trend[3]}")
                    
        except Exception as e:
            self.logger.error(f"Error in stock trends analysis: {e}")
            print(f"❌ Error analyzing stock trends: {e}")
    
    def _analyze_sentiment_trends(self, symbols: List[str], start_date: datetime, end_date: datetime, formatted: bool):
        """Analyze trends in sentiment features data"""
        try:
            all_trends = []
            
            for symbol in symbols:
                df = self.db_manager.get_sentiment_features_data(symbol=symbol, start_date=start_date, end_date=end_date)
                if df.empty:
                    continue
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                if len(df) < 7:
                    continue
                
                x = np.arange(len(df))
                
                # Key sentiment metrics to analyze
                sentiment_cols = ['sentiment_score', 'sentiment_momentum', 'news_flow_intensity', 
                                'extreme_sentiment_ratio', 'sentiment_persistence', 'news_volume']
                
                for col in sentiment_cols:
                    if col in df.columns and df[col].notna().sum() >= 7:
                        col_data = df[col].dropna()
                        if len(col_data) >= 7:
                            x_col = np.arange(len(col_data))
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x_col, col_data)
                            trend_indicator = self._get_trend_indicator(slope)
                            
                            if col_data.iloc[0] != 0:
                                pct_change = ((col_data.iloc[-1] - col_data.iloc[0]) / abs(col_data.iloc[0])) * 100
                            else:
                                pct_change = 0
                            
                            metric_name = f"{symbol} {col.replace('_', ' ').title()}"
                            if col == 'news_volume':
                                all_trends.append([metric_name, f"{trend_indicator} {slope:.0f}/day", f"{pct_change:+.1f}%", f"R²={r_value**2:.3f}"])
                            else:
                                all_trends.append([metric_name, f"{trend_indicator} {slope:.4f}/day", f"{pct_change:+.1f}%", f"R²={r_value**2:.3f}"])
            
            if not all_trends:
                print("❌ No sentiment data available for trend analysis")
                return
            
            if formatted:
                print(tabulate(all_trends, headers=["Metric", "Trend", "Total Change", "Correlation"], tablefmt="grid"))
            else:
                for trend in all_trends:
                    print(f"  {trend[0]}: {trend[1]} ({trend[2]}) - {trend[3]}")
                    
        except Exception as e:
            self.logger.error(f"Error in sentiment trends analysis: {e}")
            print(f"❌ Error analyzing sentiment trends: {e}")
    
    def _analyze_cross_symbol_trends(self, symbols: List[str], start_date: datetime, end_date: datetime, formatted: bool):
        """Analyze trends in cross-symbol features data"""
        try:
            all_trends = []
            
            for symbol in symbols:
                df = self.db_manager.get_cross_symbol_features_data(symbol=symbol, start_date=start_date, end_date=end_date)
                if df.empty:
                    continue
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                if len(df) < 7:
                    continue
                
                x = np.arange(len(df))
                
                # Key cross-symbol metrics to analyze
                cross_cols = ['relative_sentiment_ratio', 'sector_sentiment_correlation', 
                            'sector_sentiment_divergence', 'market_sentiment_correlation',
                            'market_sentiment_divergence', 'sector_news_volume']
                
                for col in cross_cols:
                    if col in df.columns and df[col].notna().sum() >= 7:
                        col_data = df[col].dropna()
                        if len(col_data) >= 7:
                            x_col = np.arange(len(col_data))
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x_col, col_data)
                            trend_indicator = self._get_trend_indicator(slope)
                            
                            if col_data.iloc[0] != 0:
                                pct_change = ((col_data.iloc[-1] - col_data.iloc[0]) / abs(col_data.iloc[0])) * 100
                            else:
                                pct_change = 0
                            
                            metric_name = f"{symbol} {col.replace('_', ' ').title()}"
                            if 'volume' in col:
                                all_trends.append([metric_name, f"{trend_indicator} {slope:.0f}/day", f"{pct_change:+.1f}%", f"R²={r_value**2:.3f}"])
                            else:
                                all_trends.append([metric_name, f"{trend_indicator} {slope:.4f}/day", f"{pct_change:+.1f}%", f"R²={r_value**2:.3f}"])
            
            if not all_trends:
                print("❌ No cross-symbol data available for trend analysis")
                return
            
            if formatted:
                print(tabulate(all_trends, headers=["Metric", "Trend", "Total Change", "Correlation"], tablefmt="grid"))
            else:
                for trend in all_trends:
                    print(f"  {trend[0]}: {trend[1]} ({trend[2]}) - {trend[3]}")
                    
        except Exception as e:
            self.logger.error(f"Error in cross-symbol trends analysis: {e}")
            print(f"❌ Error analyzing cross-symbol trends: {e}")
    
    def _get_trend_indicator(self, slope: float) -> str:
        """Convert slope to trend indicator"""
        if slope > 0.01:
            return "↗"  # Strong upward
        elif slope > 0.001:
            return "↑"  # Upward
        elif slope > -0.001:
            return "→"  # Flat
        elif slope > -0.01:
            return "↓"  # Downward
        else:
            return "↘"  # Strong downwardimport pandas as pd