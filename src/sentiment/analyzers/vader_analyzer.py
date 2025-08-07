from typing import List, Dict, Any
import logging
import warnings
from sentiment.analyzers.base_sentiment_analyzer import BaseSentimentAnalyzer

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)

class VADERAnalyzer(BaseSentimentAnalyzer):
    """VADER Sentiment Analysis: Rule + social media-based"""
    
    def __init__(self):
        super().__init__(name="VADER")
        self.analyzer = None
        self._initialize()
    
    def _initialize(self):
        """Initialize VADER analyzer"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
            self.is_initialized = True
            logger.info("VADER analyzer initialized successfully")
        except ImportError:
            logger.error("vaderSentiment is not installed. Try running: pip install vaderSentiment")
            self.is_initialized = False
        except Exception as e:
            logger.error(f"Error initializing VADER: {e}")
            self.is_initialized = False
    
    def analyze_text(self, text : str) -> Dict[str, float]:
        """Analyze sentiment of single text using VADER"""
        if not self.is_initialized:
            return {'compound': 0.0, 'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        
        if not text or not isinstance(text, str):
            return {'compound': 0.0, 'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        
        try:
            scores = self.analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'neutral': scores['neu'],
                'negative': scores['neg']
            }
        except Exception as e:
            logger.error(f"Error analyzing text with VADER: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment of multiple text strings"""
        return [self.analyze_text(text) for text in texts]
    
    def interpret_score(self, compound_score: float) -> str:
        """Interpret VADER sentiment evaluation (ie. compound score)"""
        if compound_score >= 0.05:
            return "positive"
        elif compound_score <= -0.05:
            return "negative"
        else:
            return "neutral"
