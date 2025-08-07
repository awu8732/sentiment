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
    def analyze_articles(self, articles: List[dict]) -> List[Dict]:
        """Analyze sentiment for news articles"""
        texts = []
        for article in articles:
            # Combine title & description for analysis
            combined_text = f"{article.get('title', '')} {article.get('description', '')}"
            texts.append(combined_text.strip())
        
        sentiments = self.batch_analyze(texts)

        # Enumerate sentiment per article
        for i, article in enumerate(articles):
            article['sentiment'] = sentiments[i]
            article['sentiment_score'] = sentiments[i]['compound'] # Primary sentiment score

        return articles