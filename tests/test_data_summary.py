import os
import sys
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from src.data.database import DatabaseManager
from src.managers.data_summary_manager import DataSummaryManager
from src.utils.logger import setup_logging

# Initialize the manager
config = Config()
logger = setup_logging(config)
db_manager = DatabaseManager(config.DATABASE_PATH)
summary_manager = DataSummaryManager(config, logger, db_manager)

# Print news summary for specific symbols
symbols = ["NVDA", "AAPL", "BAC"]
start_date = datetime(2025, 8, 10)
end_date = datetime(2025, 8, 21)

# Simple output
summary_manager.print_cross_symbol_summary(symbols, start_date, end_date)

# Formatted table output
#summary_manager.print_cross_symbol_summary(symbols, start_date, end_date, formatted=True)