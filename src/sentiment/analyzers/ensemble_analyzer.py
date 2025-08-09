from datetime import datetime
from typing import List, Dict, Tuple, Optional
import math
import logging
import warnings
from .base_sentiment_analyzer import BaseSentimentAnalyzer
from .vader_analyzer import VADERSentimentAnalyzer
from .finbert_analyzer import FinBERTSentimentAnalyzer

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)

class EnsembleSentimentAnalyzer(BaseSentimentAnalyzer):
    """Ensemble analyzer combining VADER and FinBERT"""
    DEFAULT_VADER_WEIGHT = 0.3
    DEFAULT_FINBERT_WEIGHT = 0.7

    def __init__(self, 
        vader_weight: Optional[float] = DEFAULT_VADER_WEIGHT, 
        finbert_weight: Optional[float] = DEFAULT_FINBERT_WEIGHT):
        super().__init__("Ensemble")

        self.vader_weight, self.finbert_weight = self._resolve_weights(vader_weight, finbert_weight)
        self.vader = VADERSentimentAnalyzer()
        self.finbert = FinBERTSentimentAnalyzer()
        self.is_initialized = self.vader.is_initialized or self.finbert.is_initialized

    def _resolve_weights(self, vader_weight: Optional[float], finbert_weight: Optional[float]) -> Tuple[float, float]:
        """Resolve and validate VADER and FinBERT weights, with inference and normalization."""
        # Check for null values
        if vader_weight is None and finbert_weight is None:
            vader_weight = self.DEFAULT_VADER_WEIGHT
            finbert_weight = self.DEFAULT_FINBERT_WEIGHT
            logger.info(f"No weights provided. Using defaults: VADER={vader_weight}, FinBERT={finbert_weight}")

        elif vader_weight is None:
            vader_weight = 1.0 - finbert_weight
            logger.info(f"VADER weight not provided. Inferred VADER={vader_weight} from FinBERT={finbert_weight}")

        elif finbert_weight is None:
            finbert_weight = 1.0 - vader_weight
            logger.info(f"FinBERT weight not provided. Inferred FinBERT={finbert_weight} from VADER={vader_weight}")

        # Sanity check: no negative weights
        if vader_weight < 0 or finbert_weight < 0:
            logger.warning(
                f"Negative weight detected (VADER={vader_weight}, FinBERT={finbert_weight}). "
                f"Reverting to defaults."
            )
            return self.DEFAULT_VADER_WEIGHT, self.DEFAULT_FINBERT_WEIGHT

        # Normalization if sum isn't equal to 1
        weight_sum = vader_weight + finbert_weight
        if not math.isclose(weight_sum, 1.0, rel_tol=1e-5):
            logger.warning(
                f"Weights do not sum to 1 (VADER={vader_weight}, FinBERT={finbert_weight}, sum={weight_sum}). "
                f"Normalizing..."
            )
            vader_weight /= weight_sum
            finbert_weight /= weight_sum
            logger.info(f"Normalized weights → VADER={vader_weight:.4f}, FinBERT={finbert_weight:.4f}")

        return vader_weight, finbert_weight
    
    def analyze_text(self, text: str, verbose: bool = False) -> Dict[str, float]:
        """Analyze text using an ensemble of VADER and FinBERT sentiment analyzers."""
        sentiment_keys = ["compound", "positive", "neutral", "negative"]
        vader_result = self.vader.analyze_text(text)
        finbert_result = self.finbert.analyze_text(text)

        if verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Requested Text: {text}")
            for model_name, result in [("VADER", vader_result), ("FinBERT", finbert_result)]:
                scores = ", ".join(f"{k}: {result[k]:.4f}" for k in sentiment_keys)
                print(f"{model_name:<8} {scores}")

        # Weighted summation
        combined_scores = {
            key: (
                vader_result[key] * self.vader_weight +
                finbert_result[key] * self.finbert_weight
            )
            for key in sentiment_keys
        }
        # Add individual model compounds for reference
        combined_scores.update({
            "vader_compound": vader_result["compound"],
            "finbert_compound": finbert_result["compound"]
        })

        return combined_scores
    
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