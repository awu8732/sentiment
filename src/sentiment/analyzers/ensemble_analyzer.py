from typing import List, Dict
import logging
import warnings
from .base_sentiment_analyzer import BaseSentimentAnalyzer
from .vader_analyzer import VADERSentimentAnalyzer
from .finbert_analyzer import FinBERTSentimentAnalyzer

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)

class EnsembleSentimentAnalyzer(BaseSentimentAnalyzer):
    """Ensemble analyzer combining VADER and FinBERT"""
    
    def __init__(self, vader_weight: float = 0.3, finbert_weight: float = 0.7):
        super().__init__("Ensemble")
        self.vader_weight = vader_weight
        self.finbert_weight = finbert_weight
        
        self.vader = VADERSentimentAnalyzer()
        self.finbert = FinBERTSentimentAnalyzer()
        self.is_initialized = self.vader.is_initialized or self.finbert.is_initialized
        
        if self.is_initialized:
            logger.info(f"Ensemble analyzer initialized (VADER: {vader_weight}, FinBERT: {finbert_weight})")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze text using ensemble of VADER and FinBERT"""
        vader_result = self.vader.analyze_text(text)
        finbert_result = self.finbert.analyze_text(text)
        #print(*vader_result.items())
        #print(*finbert_result.items())
        
        # Weighted combination
        compound = (
            vader_result['compound'] * self.vader_weight +
            finbert_result['compound'] * self.finbert_weight
        )
        
        positive = (
            vader_result['positive'] * self.vader_weight +
            finbert_result['positive'] * self.finbert_weight
        )
        
        neutral = (
            vader_result['neutral'] * self.vader_weight +
            finbert_result['neutral'] * self.finbert_weight
        )
        
        negative = (
            vader_result['negative'] * self.vader_weight +
            finbert_result['negative'] * self.finbert_weight
        )
        
        return {
            'compound': compound,
            'positive': positive,
            'neutral': neutral,
            'negative': negative,
            'vader_compound': vader_result['compound'],
            'finbert_compound': finbert_result['compound']
        }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, float]]:
        """Batch analyze using ensemble"""
        return [self.analyze_text(text) for text in texts]

    def get_analyzer(analyzer_type: str = "ensemble") -> BaseSentimentAnalyzer:
        """Factory function to get sentiment analyzer"""
        analyzers = {
            'vader': VADERSentimentAnalyzer,
            'finbert': FinBERTSentimentAnalyzer,
            'ensemble': EnsembleSentimentAnalyzer
        }
        
        if analyzer_type.lower() not in analyzers:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}. Available: {list(analyzers.keys())}")
        
        return analyzers[analyzer_type.lower()]()