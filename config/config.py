import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()   

class Config:
    """Main configuration class"""
    
    # Database settings
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/financial_data.db')
    
    # API Keys (should be set as environment variables)
    API_KEYS = {
        'newsapi': os.getenv('NEWSAPI_KEY'),
        'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY'),
        'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
        'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
        'reddit_user_agent': os.getenv('REDDIT_USER_AGENT', 'NewssentimentTrader/1.0')
    }
    
    # Data collection settings
    DEFAULT_LOOKBACK_DAYS = 30
    STOCK_INTERVAL = "1h"
    STOCK_PERIOD = "2y"
    
    # Rate limiting (seconds)
    RATE_LIMITS = {
        'newsapi': 1.0,
        'alpha_vantage': 12.0,  # 5 calls per minute
        'reddit': 1.0,
        'yfinance': 0.1
    }
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'logs/trading_bot.log'
    
    # Model settings
    MODEL_CONFIGS = {
        'tft': {
            'max_encoder_length': 30,
            'max_prediction_length': 5,
            'hidden_size': 64,
            'attention_head_size': 4,
            'dropout': 0.1,
            'learning_rate': 0.03
        }
    }