import os
import sys
import pytest
from unittest.mock import Mock, patch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from src.data.collectors.news_collector import NewsAPICollector, AlphaVantageNewsCollector, RedditCollector

config = Config()

class TestNewsCollection:
    @pytest.fixture(autouse=True)
    def setup_api_keys(self):
        """Fixture to ensure all API keys are available for testing"""
        # Check NewsAPI key
        if not config.API_KEYS.get('newsapi'):
            pytest.skip("NewsAPI key not configured - skipping related tests")
        
        # Check AlphaVantage key
        if not config.API_KEYS.get('alpha_vantage'):
            pytest.skip("AlphaVantage API key not configured - skipping related tests")
        
        # Check Reddit keys
        reddit_keys = ['reddit_client_id', 'reddit_client_secret', 'reddit_user_agent']
        if not all(config.API_KEYS.get(key) for key in reddit_keys):
            pytest.skip("Reddit API keys not fully configured - skipping related tests")

    @pytest.fixture
    def newsapi_collector(self):
        """Fixture to create NewsAPICollector instance"""
        if config.API_KEYS.get('newsapi'):
            return NewsAPICollector(config.API_KEYS['newsapi'], rate_limit=2.0)
        return None

    @pytest.fixture
    def alphavantage_collector(self):
        """Fixture to create AlphaVantageNewsCollector instance"""
        if config.API_KEYS.get('alpha_vantage'):
            return AlphaVantageNewsCollector(config.API_KEYS['alpha_vantage'])
        return None

    @pytest.fixture
    def reddit_collector(self):
        """Fixture to create RedditCollector instance"""
        reddit_keys = ['reddit_client_id', 'reddit_client_secret', 'reddit_user_agent']
        if all(config.API_KEYS.get(key) for key in reddit_keys):
            return RedditCollector(
                client_id=config.API_KEYS['reddit_client_id'],
                client_secret=config.API_KEYS['reddit_client_secret'],
                user_agent=config.API_KEYS['reddit_user_agent']
            )
        return None

    def test_config_has_required_api_keys(self):
        """Test that config contains the required API key structure"""
        assert hasattr(config, 'API_KEYS'), "Config should have API_KEYS attribute"
        assert isinstance(config.API_KEYS, dict), "API_KEYS should be a dictionary"

    def test_newsapi_collector_initialization(self, newsapi_collector):
        """Test NewsAPICollector can be initialized with valid API key"""
        if newsapi_collector:
            assert newsapi_collector is not None
            assert hasattr(newsapi_collector, 'collect_data')
        else:
            pytest.skip("NewsAPI key not available")

    def test_alphavantage_collector_initialization(self, alphavantage_collector):
        """Test AlphaVantageNewsCollector can be initialized with valid API key"""
        if alphavantage_collector:
            assert alphavantage_collector is not None
            assert hasattr(alphavantage_collector, 'collect_data')
        else:
            pytest.skip("AlphaVantage API key not available")

    def test_reddit_collector_initialization(self, reddit_collector):
        """Test RedditCollector can be initialized with valid API keys"""
        if reddit_collector:
            assert reddit_collector is not None
            assert hasattr(reddit_collector, 'collect_data')
        else:
            pytest.skip("Reddit API keys not available")

    @pytest.mark.integration
    def test_newsapi_collector_functionality(self, newsapi_collector):
        """Test NewsAPICollector can collect articles"""
        if not newsapi_collector:
            pytest.skip("NewsAPI key not available")
        
        print('Testing NewsAPI collector...')
        articles = newsapi_collector.collect_data('AAPL', 3)
        
        assert articles is not None, "Should return articles list"
        assert isinstance(articles, list), "Should return a list of articles"
        
        print(f'Result: {len(articles)} articles')
        
        if articles:
            # Test article structure
            article = articles[0]
            assert hasattr(article, 'title'), "Article should have a title"
            assert hasattr(article, 'description'), "Article should have a description"
            assert hasattr(article, 'timestamp'), "Article should have a timestamp"
            assert hasattr(article, 'source'), "Article should have a source"
            assert hasattr(article, 'symbol'), "Article should have a symbol"   
            print(f'Sample: {article.title}')

    @pytest.mark.integration
    def test_alphavantage_collect_data(self, alphavantage_collector):
        """Test that Alpha Vantage collector can collect articles"""
        if not alphavantage_collector:
            pytest.skip("Alpha Vantage key is not available")

        print("Testing Alpha Vantage collector...")
        articles = alphavantage_collector.collect_data('AAPL', days_back=3)
        
        assert articles is not None, "Alpha Vantage should return articles"
        assert isinstance(articles, list), "Should return a list of articles"

        print(f'Result: {len(articles)} articles')

        if articles:
            # Test article structure
            article = articles[0]
            assert hasattr(article, 'title'), "Article should have a title"
            assert hasattr(article, 'description'), "Article should have a description"
            assert hasattr(article, 'timestamp'), "Article should have a timestamp"
            assert hasattr(article, 'source'), "Article should have a source"
            assert hasattr(article, 'symbol'), "Article should have a symbol"   
            print(f'Sample: {article.title}')

    @pytest.mark.integration
    def test_reddit_collect_data(self, reddit_collector):
        """Test that Reddit collector can collect posts"""
        if not reddit_collector:
            pytest.skip("Reddit keys are not fully available")

        print("Testing Reddit collector...")
        posts = reddit_collector.collect_data('AAPL', days_back=3)
        
        assert posts is not None, "Reddit should return posts"
        assert isinstance(posts, list), "Should return a list of posts"

        print(f'Result: {len(posts)} posts')

        if posts:
            # Test post structure
            post = posts[0]
            assert hasattr(post, 'title'), "Article should have a title"
            assert hasattr(post, 'description'), "Article should have a description"
            assert hasattr(post, 'timestamp'), "Article should have a timestamp"
            assert hasattr(post, 'source'), "Article should have a source"
            assert hasattr(post, 'symbol'), "Article should have a symbol"   
            print(f'Sample: {post.title}')

    @pytest.mark.parametrize("symbol,days_back", [
        ("AAPL", 1),
        ("GOOGL", 3),
        ("MSFT", 7),
        ("TSLA", 14)
    ])
    def test_newsapi_collector_with_different_params(self, newsapi_collector, symbol, days_back):
        """Test NewsAPICollector with different symbols and days_back values"""
        if not newsapi_collector:
            pytest.skip("NewsAPI key not available")
        
        articles = newsapi_collector.collect_data(symbol, days_back)
        assert isinstance(articles, list)
        
        # Verify articles are from the specified time period if any are returned
        if articles:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Check if articles have timestamp and are within the date range
            for article in articles[:3]:  # Check first 3 articles
                if isinstance(article, dict) and 'timestamp' in article:
                    # Assuming timestamp is a datetime object or ISO string
                    if isinstance(article['timestamp'], str):
                        article_date = datetime.fromisoformat(article['timestamp'].replace('Z', '+00:00'))
                    else:
                        article_date = article['timestamp']
                    
                    # Article should be newer than cutoff date
                    assert article_date >= cutoff_date, f"Article should be from within {days_back} days"

    @pytest.mark.unit
    def test_collector_rate_limiting(self, newsapi_collector):
        """Test that rate limiting is working"""
        if not newsapi_collector:
            pytest.skip("NewsAPI key not available")
        
        import time
        start_time = time.time()
        
        # Make two quick requests
        newsapi_collector.collect_data('AAPL', days_back=1)
        newsapi_collector.collect_data('GOOGL', days_back=1)
        
        elapsed_time = time.time() - start_time

        assert elapsed_time >= 0.4, "Rate limiting should enforce delays between requests"

    @pytest.mark.unit
    def test_empty_results_handling(self, newsapi_collector):
        """Test handling of queries that return no results"""
        if not newsapi_collector:
            pytest.skip("NewsAPI key not available")

        articles = newsapi_collector.collect_data('ZZZNONEXISTENT', days_back=1)
        assert isinstance(articles, list), "Should return empty list for no results"