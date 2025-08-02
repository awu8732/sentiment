from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
from datetime import datetime
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

class FinBERTAnalyzer(BaseSentimentAnalyzer):
    """FinBERT Sentiment Analysis: Transformer-based architecture for financial text"""
    
    def __init__(self):
        super().__init__("FinBERT")
        self.tokenizer = None
        self.model = None
        self.device = None
        self._initialize()

    def _initialize(self):
        """Initialize FinBERT model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            model_name = "ProsusuAI/finbert"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            self.is_initialized = True
            logger.info(f"FinBERT analyzer intiailized successfully on {self.device}")
        
        except ImportError:
            logger.error("transformers or torch not installed. Try running: pip install transformers torch")
            self.is_initialized = False
        except Exception as e:
            logger.error(f"Error initializing FinBERT: {e}")
            self.is_initialized = False

    def analyze_text(self, text: str ) -> Dict[str, float]:
        """Analyze sentiment of a """
        return super().analyze_text(text)

            
