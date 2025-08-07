from typing import List, Dict
import logging
import warnings
from sentiment.analyzers.base_sentiment_analyzer import BaseSentimentAnalyzer

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)

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

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single string of text using FinBERT"""
        if not self.is_initialized:
            return {'compound': 0.0, 'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        
        if not text or not isinstance(text, str):
            return {'compound': 0.0, 'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Tokenize and encode text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [negative, neutral, positive]
            probs = probabilities.cpu().numpy()[0]
            
            negative_prob = float(probs[0])
            neutral_prob = float(probs[1])
            positive_prob = float(probs[2])
            
            # Calculate compound score (similar to VADER scale: -1 to 1)
            compound = positive_prob - negative_prob
            
            return {
                'compound': compound,
                'positive': positive_prob,
                'neutral': neutral_prob,
                'negative': negative_prob
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