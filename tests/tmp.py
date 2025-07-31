# test_newsapi.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from src.data.collectors.news_collector import NewsAPICollector

config = Config()
print(f'API Key exists: {bool(config.API_KEYS.get("newsapi"))}')

if config.API_KEYS.get('newsapi'):
    collector = NewsAPICollector(config.API_KEYS['newsapi'], rate_limit=2.0)
    print('Testing NewsAPI collector...')
    articles = collector.collect_data('AAPL', 3)
    print(f'Result: {len(articles)} articles')
    if articles:
        print(f'Sample: {articles[0].title}')
else:
    print('No NewsAPI key found')
