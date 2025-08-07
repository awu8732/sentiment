from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)

class BaseSentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers"""

    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False

    @abstractmethod
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text"""
        pass

    @abstractmethod
    def batch_analyze(self, articles: List[str]) -> List[Dict]:
        """Analyze sentiment for news articles"""
        pass