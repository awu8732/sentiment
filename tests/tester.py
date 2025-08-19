import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from src.sentiment.sentiment_analysis_runner import SentimentAnalysisRunner

config = Config()
sentiment_analysis_runner = SentimentAnalysisRunner(config)
market_manager = sentiment_analysis_runner.market_manager

lookback_date = datetime.now() - timedelta(days=7)
all_new_df = market_manager._get_market_data(lookback_date)
market_manager.create_market_features(all_new_df)
print("hello world")