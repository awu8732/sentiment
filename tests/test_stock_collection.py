import os
import sys
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from config.symbols import SYMBOLS
from src.data.pipeline import DataPipeline
from src.data.database import DatabaseManager
from src.data.collectors import AlphaVantageStockCollector, AlphaVantageNewsCollector
from src.utils.logger import setup_logging

config = Config()
logger = setup_logging(config)

pipeline = DataPipeline(config)
db_manager = DatabaseManager(config.DATABASE_PATH)
news_collector = AlphaVantageNewsCollector(config.API_KEYS['alpha_vantage'])

SYMBOLS = {
    'technology': [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 
        'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'PYPL', 'UBER',
        'SHOP', 'DOCU', 'ROKU', 'SNAP', 'PINS'
    ],
    'finance': [
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SCHW', 
        'USB', 'PNC', 'TFC', 'COF', 'ALL', 'AIG', 'MET', 'PRU', 'TRV',
        'CB', 'AFL', 'CINF', 'PGR', 'HIG', 'WRB'
    ],
    'healthcare': [
        'UNH', 'JNJ', 'PFE', 'ABT', 'TMO', 'MDT', 'DHR', 'BMY', 'ABBV',
        'MRK', 'CVS', 'CI', 'HUM', 'CNC', 'MOH', 'GILD', 'AMGN',
        'BIIB', 'VRTX', 'REGN', 'ISRG', 'SYK', 'BSX'
    ],
    'energy': [
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY',
        'BKR', 'HAL', 'DVN', 'FANG', 'APA', 'HES', 'NOV', 'RRC', 'MTDR', 
        'SM', 'RIG', 'HP'
    ]
}        

symbol = 'UBER'
start_date = datetime.now().replace(month=1, day=1, year=2023)
#end_date = datetime(year=2025, month=7, day=24)
end_date = datetime.now()

symbol_data = news_collector.collect_interval_data(symbol, start_date, end_date)
db_manager.insert_news_batch(symbol_data)