import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.sentiment.analyzers import EnsembleSentimentAnalyzer
from src.sentiment.feature_engineering import SentimentFeatureEngineer
from src.data.models import NewsArticle

def analyze_headlines_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        headlines = [line.strip() for line in f if line.strip()]

    # Iterate through each line of inputted file
    ensembleAnalyzer = EnsembleSentimentAnalyzer()
    print("Analyzing headlines...\n")
    for idx, headline in enumerate(headlines, 1):
        article = NewsArticle(
            timestamp=datetime.now(),
            title=headline,
            description="",
            source="TestInput",
            url="http://example.com",
            symbol="TEST"
        )
        print(f"{idx}: {headline}")
        ensemble_result = ensembleAnalyzer.analyze_text(headline)
        print("COMPOUND:", ", ".join(f"{k}: {round(v,4)}" for k, v in ensemble_result.items()))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_headlines.py <path_to_input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    analyze_headlines_from_file(input_file)
