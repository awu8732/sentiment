import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from src.sentiment.sentiment_analysis_runner import SentimentAnalysisRunner

config = Config()
sentiment_analysis_runner = SentimentAnalysisRunner(config)
market_manager = sentiment_analysis_runner.market_manager
cross_symbol_manager = sentiment_analysis_runner.cross_symbol_manager
SYM = 'AAPL'

lookback_date = datetime.now() - timedelta(days=7)

sector_df = cross_symbol_manager._get_sector_data(SYM)
symbol_df = cross_symbol_manager._get_symbol_sentiment_data(SYM)
market_df = market_manager._get_market_data(start_date=lookback_date)
# cross_symbol_manager._create_cross_sector_features(SYM, symbol_df, sector_df)

print(len(sector_df))
print(len(symbol_df))
print(len(market_df))
#all_new_df = market_manager._get_market_data(lookback_date)
#market_manager.create_market_features(all_new_df)
print("done.")