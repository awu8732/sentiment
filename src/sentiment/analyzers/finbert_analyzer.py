from typing import List, Dict
import logging
import warnings
from .base_sentiment_analyzer import BaseSentimentAnalyzer

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)

class FinBERTSentimentAnalyzer(BaseSentimentAnalyzer):
    """FinBERT Sentiment Analysis: Transformer-based architecture for financial text"""
    
    def __init__(self):
        super().__init__("FinBERT")
        self.nlp_pipeline = None
        self.is_initialized = False
        self._initialize()

    def _initialize(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            import torch

            model_name = "ProsusAI/finbert"
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Check for CUDA
            if torch.cuda.is_available():
                device = 0
                model = model.cuda()
            else:
                device = -1

            self.nlp_pipeline = pipeline(
                "text-classification", 
                model=model, 
                tokenizer=tokenizer, 
                device=device,
                top_k=None
            )
            self.is_initialized = True
            logger.info(f"FinBERT analyzer initialized using '{model_name}' on {'cuda' if device == 0 else 'cpu'}")
        except ImportError:
            logger.error("transformers or torch not installed. Try running: pip install transformers torch")
            self.is_initialized = False
        except Exception as e:
            logger.error(f"Error initializing FinBERT: {e}")
            self.is_initialized = False

    def analyze_text(self, text: str) -> Dict[str, float]:
        if not self.is_initialized or not isinstance(text, str) or not text.strip():
            return {'compound': 0.0, 'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        
        try:
            results = self.nlp_pipeline(text)
            positive_prob = 0.0
            neutral_prob = 0.0
            negative_prob = 0.0
            
            # Extract probabilities from results
            for result in results[0]:
                label = result['label'].upper()
                score = result['score']
                
                if 'POSITIVE' in label:
                    positive_prob = score
                elif 'NEGATIVE' in label:
                    negative_prob = score
                elif 'NEUTRAL' in label:
                    neutral_prob = score
            
            # Calculate compound score
            compound = positive_prob - negative_prob
            
            return {
                'compound': float(compound),
                'positive': float(positive_prob),
                'neutral': float(neutral_prob),
                'negative': float(negative_prob)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text with FinBERT: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
    
    def batch_analyze(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, float]]:
        """Analyze sentiment of multiple texts in batches"""
        if not self.is_initialized:
            return [{'compound': 0.0, 'positive': 0.0, 'neutral': 0.0, 'negative': 0.0} for _ in texts]
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = [self.analyze_text(text) for text in batch_texts]
            results.extend(batch_results)
        
        return results
    
    def interpret_score(self, compound_score: float) -> str:
        """Quickly interpret FinBERT compound score"""
        if compound_score >= 0.1:
            return "positive"
        elif compound_score <= -0.1:
            return "negative"
        else:
            return "neutral"