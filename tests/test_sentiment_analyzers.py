import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sentiment.analyzers import (
    VADERSentimentAnalyzer, FinBERTSentimentAnalyzer, EnsembleSentimentAnalyzer
)
from src.sentiment.utils import (
    StatisticalUtils, TimeUtils, SentimentUtils, 
    NewsUtils, CrossSymbolUtils
)
from src.sentiment.feature_engineering import SentimentFeatureEngineer
from src.data.models import NewsArticle

print("hello world")