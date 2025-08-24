import os
import sys
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from src.data.pipeline import DataPipeline
from src.data.collectors.stock_collector import AlphaVantageStockCollector
from src.utils.logger import setup_logging

config = Config()
logger = setup_logging(config)

pipeline = DataPipeline(config)

stock_collector = AlphaVantageStockCollector(config.RATE_LIMITS['alpha_vantage'])
print(stock_collector.get_intraday_data('AAPL', "60min", 3))