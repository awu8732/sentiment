import pytest
import pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sentiment.analyzers import (
    VADERSentimentAnalyzer, FinBERTSentimentAnalyzer, EnsembleSentimentAnalyzer
)
from src.sentiment.utils import (
    StatisticalUtils, TimeUtils, SentimentUtils, 
    NewsUtils, CrossSymbolUtils
)
from src.sentiment.feature_engineering import SentimentFeatureEngineer
from src.data.models import NewsArticle

class TestSentimentAnalyzers:
    """Test individual sentiment analyzer classes"""
    
    def test_vader_analyzer_basic(self):
        """Test VADER analyzer basic functionality"""
        analyzer = VADERSentimentAnalyzer()
        
        # Test positive text
        positive_result = analyzer.analyze_text("This is amazing and wonderful news!")
        assert positive_result['compound'] > 0
        assert positive_result['positive'] > 0
        
        # Test negative text
        negative_result = analyzer.analyze_text("This is terrible and awful news!")
        assert negative_result['compound'] < 0
        assert negative_result['negative'] > 0
        
        # Test empty text
        empty_result = analyzer.analyze_text("")
        assert empty_result['compound'] == 0.0

    def test_finbert_analyzer_initialization(self):
        """Test FinBERT analyzer initialization"""        
        # Create analyzer
        analyzer = FinBERTSentimentAnalyzer()

         # Test positive text
        positive_result = analyzer.analyze_text("This is amazing and wonderful news!")
        assert positive_result['compound'] > 0
        assert positive_result['positive'] > 0
        
        # Test negative text
        negative_result = analyzer.analyze_text("This is such terrible and awful news!")
        assert negative_result['compound'] < 0
        assert negative_result['negative'] > 0
        
        # Test empty text
        empty_result = analyzer.analyze_text("")
        assert empty_result['compound'] == 0.0
    
    def test_composite_analyzer(self):
        """Test composite analyzer combining multiple methods"""
        analyzer = EnsembleSentimentAnalyzer()
        result = analyzer.analyze_text("This is a test message")
        
        # Should have results from both VADER and FinBERT
        assert 'compound' in result
        assert 'finbert_compound' in result
        assert 'vader_compound' in result 

class TestStatisticalUtils:
    """Test statistical utility functions"""

    def test_safe_skew(self):
        """Test safe skewness calculation"""
        normal_data = np.random.normal(0, 1, 100)
        skew_result = StatisticalUtils.safe_skew(normal_data)
        assert isinstance(skew_result, float)
        assert not np.isnan(skew_result)
        assert abs(skew_result) < 2.0  # Should be reasonable

        # Empty data should return nan
        empty_data = np.array([])
        assert np.isnan(StatisticalUtils.safe_skew(empty_data))

        # All NaNs
        all_nan = np.array([np.nan, np.nan])
        assert np.isnan(StatisticalUtils.safe_skew(all_nan))

    def test_safe_correlation(self):
        """Test safe correlation calculation"""
        x = [1, 2, 3, 4, 5]

        # Perfect positive correlation
        y = [2, 4, 6, 8, 10]
        corr = StatisticalUtils.safe_correlation(x, y)
        assert isinstance(corr, float)
        assert abs(corr - 1.0) < 0.01

        # Perfect negative correlation
        y_neg = [10, 8, 6, 4, 2]
        corr_neg = StatisticalUtils.safe_correlation(x, y_neg)
        assert abs(corr_neg + 1.0) < 0.01

        # Insufficient data
        short_x = [1]
        short_y = [2]
        assert np.isnan(StatisticalUtils.safe_correlation(short_x, short_y))

        # Mismatched lengths
        assert np.isnan(StatisticalUtils.safe_correlation(x, [1, 2]))

        # All NaNs
        x_nan = [1, np.nan, 3]
        y_nan = [2, np.nan, 6]
        result = StatisticalUtils.safe_correlation(x_nan, y_nan)
        assert not np.isnan(result)

        # No valid pairs
        x_empty = [np.nan]
        y_empty = [np.nan]
        assert np.isnan(StatisticalUtils.safe_correlation(x_empty, y_empty))

    def test_safe_std(self):
        """Test safe standard deviation calculation"""
        data = [1, 2, 3, 4, 5]
        std_result = StatisticalUtils.safe_std(data)
        assert isinstance(std_result, float)
        assert std_result > 0

        # Single value
        single_data = [5]
        assert np.isnan(StatisticalUtils.safe_std(single_data))

        # Empty array
        assert np.isnan(StatisticalUtils.safe_std([]))

        # All NaNs
        assert np.isnan(StatisticalUtils.safe_std([np.nan, np.nan]))

    def test_weighted_average(self):
        """Test weighted average calculation"""
        values = [1, 2, 3, 4, 5]

        # No weights
        avg = StatisticalUtils.weighted_average(values)
        assert abs(avg - 3.0) < 0.01

        # With weights
        weights = [1, 1, 1, 1, 5]  # Skew toward 5
        weighted_avg = StatisticalUtils.weighted_average(values, weights)
        assert weighted_avg > 3.0

        # Mismatched lengths
        mismatch_avg = StatisticalUtils.weighted_average(values, [1, 2])
        assert abs(mismatch_avg - 3.0) < 0.01  # Falls back to unweighted mean

        # Empty values
        assert np.isnan(StatisticalUtils.weighted_average([]))

        # All NaN values
        assert np.isnan(StatisticalUtils.weighted_average([np.nan, np.nan]))

        # NaNs in values and weights
        values = [1, 2, np.nan, 4]
        weights = [1, 1, 1, np.nan]
        avg = StatisticalUtils.weighted_average(values, weights)
        assert not np.isnan(avg)

class TestTimeUtils:
    """Unit tests for TimeUtils methods (U.S. Eastern Market hours logic)"""

    @staticmethod
    def make_dt(year, month, day, hour, minute=0):
        """Helper to create an Eastern Time timezone-aware datetime object"""
        eastern = pytz.timezone("US/Eastern")
        naive_dt = datetime(year, month, day, hour, minute)
        return eastern.localize(naive_dt)

    def test_is_market_hours_within_range(self):
        """Returns True during market hours on a weekday"""
        dt = self.make_dt(2024, 1, 15, 10)  # Monday, 10 AM ET
        assert TimeUtils.is_market_hours(dt) is True

    def test_is_market_hours_before_open(self):
        """Returns False before market open on a weekday"""
        dt = self.make_dt(2024, 1, 15, 8)  # Monday, 8 AM ET
        assert TimeUtils.is_market_hours(dt) is False

    def test_is_market_hours_after_close(self):
        """Returns False after market close on a weekday"""
        dt = self.make_dt(2024, 1, 15, 17)  # Monday, 5 PM ET
        assert TimeUtils.is_market_hours(dt) is False

    def test_is_market_hours_on_weekend(self):
        """Returns False on weekends"""
        dt = self.make_dt(2024, 1, 13, 10)  # Saturday, 10 AM ET
        assert TimeUtils.is_market_hours(dt) is False

    def test_is_after_hours_during_market(self):
        """Returns False during normal market hours"""
        dt = self.make_dt(2024, 1, 15, 10)  # Monday, 10 AM
        assert TimeUtils.is_after_hours(dt) is False

    def test_is_after_hours_pre_market(self):
        """Returns True before market opens"""
        dt = self.make_dt(2024, 1, 15, 8)  # Monday, 8 AM
        assert TimeUtils.is_after_hours(dt) is True

    def test_is_after_hours_on_weekend(self):
        """Returns True on weekend (non-trading days)"""
        dt = self.make_dt(2024, 1, 13, 10)  # Saturday
        assert TimeUtils.is_after_hours(dt) is True

    def test_get_trading_day_type_market_hours(self):
        """Correctly classifies market hours"""
        dt = self.make_dt(2024, 1, 15, 10)  # Monday 10 AM
        assert TimeUtils.get_trading_day_type(dt) == "market_hours"

    def test_get_trading_day_type_pre_market(self):
        """Correctly classifies pre-market time"""
        dt = self.make_dt(2024, 1, 15, 8)  # Monday 8 AM
        assert TimeUtils.get_trading_day_type(dt) == "pre_market"

    def test_get_trading_day_type_after_hours(self):
        """Correctly classifies after-hours time"""
        dt = self.make_dt(2024, 1, 15, 17)  # Monday 5 PM
        assert TimeUtils.get_trading_day_type(dt) == "after_hours"

    def test_get_trading_day_type_weekend(self):
        """Correctly classifies weekend"""
        dt = self.make_dt(2024, 1, 13, 10)  # Saturday
        assert TimeUtils.get_trading_day_type(dt) == "weekend"

class TestNewsUtils:
    """Test news utility functions"""
    
    def test_calculate_source_diversity(self):
        """Test source diversity calculation"""
        # Single source (no diversity)
        single_source = pd.DataFrame({'source': ['Reuters', 'Reuters', 'Reuters']})
        diversity_single = NewsUtils.calculate_source_diversity(single_source)
        assert diversity_single == 0.0
        
        # Multiple sources (high diversity)
        multi_source = pd.DataFrame({'source': ['Reuters', 'Bloomberg', 'CNBC', 'WSJ']})
        diversity_multi = NewsUtils.calculate_source_diversity(multi_source)
        assert diversity_multi > 0.0
        
        # Empty DataFrame
        empty_df = pd.DataFrame({'source': []})
        diversity_empty = NewsUtils.calculate_source_diversity(empty_df)
        assert diversity_empty == 0.0
    
    def test_calculate_news_flow_intensity(self):
        """Test news flow intensity calculation"""
        # Articles spread over 10 hours
        timestamps = pd.date_range('2024-01-15 10:00', periods=10, freq='h')
        window_data = pd.DataFrame({'timestamp': timestamps})
        
        # Empty DataFrame
        empty_data = pd.DataFrame({'timestamp': []})
        intensity_empty = NewsUtils.calculate_news_flow_intensity(empty_data)
        assert intensity_empty == 0.0
        
        intensity = NewsUtils.calculate_news_flow_intensity(window_data)
        # Should be approximately 1 article per hour
        assert 0.9 <= intensity <= 1.12
    
    def test_calculate_source_credibility(self):
        """Test source credibility calculation"""
        # High credibility sources
        high_cred = pd.DataFrame({'source': ['Reuters', 'Bloomberg', 'Wall Street Journal']})
        cred_high = NewsUtils.calculate_source_credibility(high_cred)
        
        # Low credibility sources
        low_cred = pd.DataFrame({'source': ['Unknown Blog', 'Random Site', 'Another Blog']})
        cred_low = NewsUtils.calculate_source_credibility(low_cred)
        
        # High credibility should score higher
        assert cred_high > cred_low
        
        # Empty DataFrame
        empty_df = pd.DataFrame({'source': []})
        cred_empty = NewsUtils.calculate_source_credibility(empty_df)
        assert cred_empty == 0.0

class TestCrossSymbolUtils:
    """Test cross-symbol utility functions"""
    
    def test_calculate_relative_sentiment(self):
        """Test relative sentiment calculation"""
        target = [0.5, 0.3, 0.7]
        sector = [0.2, 0.1, 0.4]
        
        relative = CrossSymbolUtils.calculate_relative_sentiment(target, sector)
        assert relative > 0  # Target more positive than sector
        
        # Empty lists
        empty_relative = CrossSymbolUtils.calculate_relative_sentiment([], [])
        assert empty_relative == 0.0
    
    def test_calculate_sentiment_divergence(self):
        """Test sentiment divergence calculation"""
        target = [0.1, 0.2, 0.3, 0.4, 0.5]  # Low volatility
        sector = [0.8, -0.5, 0.9, -0.7, 0.6]  # High volatility
        
        divergence = CrossSymbolUtils.calculate_sentiment_divergence(target, sector)
        assert divergence > 0  # Should detect the volatility difference
        
        # Similar volatility
        similar_target = [0.1, 0.2, 0.3]
        similar_sector = [0.15, 0.25, 0.35]
        divergence_low = CrossSymbolUtils.calculate_sentiment_divergence(similar_target, similar_sector)
        assert divergence_low < divergence
    
    def test_calculate_sector_correlation(self):
        """Test sector correlation calculation"""
        # Perfect positive correlation
        target = [0.1, 0.2, 0.3, 0.4, 0.5]
        sector = [0.1, 0.2, 0.3, 0.4, 0.5]
        corr = CrossSymbolUtils.calculate_sentiment_correlation(target, sector)
        assert abs(corr - 1.0) < 0.01
        
        # Perfect negative correlation
        sector_neg = [-0.1, -0.2, -0.3, -0.4, -0.5]
        corr_neg = CrossSymbolUtils.calculate_sentiment_correlation(target, sector_neg)
        assert abs(corr_neg - (-1.0)) < 0.01
        
        # Insufficient data
        short_target = [0.1]
        short_sector = [0.2]
        corr_short = CrossSymbolUtils.calculate_sentiment_correlation(short_target, short_sector)
        assert corr_short == 0.0

class TestSentimentFeatureEngineer:
    """Test the main feature engineering class"""
    
    @pytest.fixture
    def engineer(self):
        """Create a SentimentFeatureEngineer instance for testing"""
        return SentimentFeatureEngineer(use_finbert=False)
    
    @pytest.fixture
    def sample_articles_df(self):
        """Create sample articles DataFrame"""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        data = []
        
        test_cases = [
            ("Apple reports record earnings", "Strong quarterly results exceed expectations", "Reuters", 0),
            ("Apple faces supply issues", "Production delays may impact Q2 results", "Bloomberg", 1),
            ("Apple launches new product", "Innovative features attract consumer interest", "TechCrunch", 2),
            ("Apple stock analysis", "Analysts remain optimistic about growth prospects", "WSJ", 3),
            ("Apple CEO interview", "Leadership discusses future strategy and vision", "CNBC", 4)
        ]
        
        for title, desc, source, hour_offset in test_cases:
            data.append({
                'timestamp': base_time + timedelta(hours=hour_offset),
                'title': title,
                'description': desc,
                'source': source,
                'url': f'https://example.com/article{hour_offset}',
                'symbol': 'AAPL'
            })
        
        return pd.DataFrame(data)
    
    def test_analyze_article_sentiment(self, engineer):
        """Test article sentiment analysis"""
        article = NewsArticle(
            timestamp=datetime.now(),
            title="Amazing breakthrough in technology",
            description="Company achieves excellent results with outstanding performance",
            source="Reuters",
            url="http://example.com",
            symbol="AAPL"
        )
        
        sentiment = engineer.analyze_article_sentiment(article)
        
        # Check that all expected keys are present
        expected_keys = ['compound', 'positive', 'negative', 'neutral', 'finbert_compound', 'vader_compound']
        assert all(key in sentiment for key in expected_keys)
        
        # Positive article should have positive sentiment
        assert sentiment['compound'] > 0
        assert sentiment['positive'] > 0.05
    
    def test_create_time_based_features(self, engineer, sample_articles_df):
        """Test time-based feature creation"""
        result = engineer.create_time_based_features(sample_articles_df, "AAPL", ['1H', '4H'])
        
        assert not result.empty
        assert 'timestamp' in result.columns
        assert 'symbol' in result.columns
        
        # Check for expected feature columns
        expected_features = [
            'sentiment_mean_1H', 'sentiment_std_1H', 'sentiment_skew_1H',
            'news_volume_1H', 'source_diversity_1H', 'sentiment_momentum_1H',
            'sentiment_mean_4H', 'sentiment_std_4H', 'sentiment_skew_4H',
            'news_volume_4H', 'source_diversity_4H', 'sentiment_momentum_4H'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
    
    def test_create_time_based_features_empty_data(self, engineer):
        """Test time-based features with empty data"""
        empty_df = pd.DataFrame()
        result = engineer.create_time_based_features(empty_df, "AAPL")
        assert result.empty
    
    def test_create_advanced_features(self, engineer, sample_articles_df):
        """Test advanced feature creation"""
        result = engineer.create_advanced_features(sample_articles_df, "AAPL")
        
        assert not result.empty
        assert 'timestamp' in result.columns
        assert 'symbol' in result.columns
        
        expected_features = [
            'sentiment_regime', 'market_hours_sentiment', 'after_hours_sentiment',
            'sentiment_volatility', 'news_flow_intensity', 'source_credibility_score',
            'sentiment_persistence', 'extreme_sentiment_ratio'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
    
    def test_create_cross_symbol_features(self, engineer):
        """Test cross-symbol feature creation"""
        # Create multi-symbol dataset
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        data = []
        
        symbols = ["AAPL", "GOOGL", "MSFT"]
        for i, symbol in enumerate(symbols):
            for j in range(3):
                data.append({
                    'timestamp': base_time + timedelta(hours=i*2 + j),
                    'title': f"{symbol} quarterly results",
                    'description': f"Analysis of {symbol} performance",
                    'source': 'Reuters',
                    'url': f'http://example.com/{symbol}_{j}',
                    'symbol': symbol
                })
        
        multi_symbol_df = pd.DataFrame(data)
        
        result = engineer.create_cross_symbol_features(
            multi_symbol_df, "AAPL", ["GOOGL", "MSFT"]
        )
        
        assert not result.empty
        assert 'timestamp' in result.columns
        assert 'symbol' in result.columns
        
        expected_features = [
            'sector_sentiment_mean', 'sentiment_sector_correlation',
            'relative_sentiment_strength', 'sector_news_volume',
            'sentiment_divergence'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
    
    def test_feature_data_types_and_ranges(self, engineer, sample_articles_df):
        """Test that features have correct data types and reasonable ranges"""
        result = engineer.create_time_based_features(sample_articles_df, "AAPL", ['1H'])
        
        if not result.empty:
            # Check data types
            assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
            assert result['symbol'].dtype == object
            assert pd.api.types.is_numeric_dtype(result['sentiment_mean_1H'])
            assert pd.api.types.is_numeric_dtype(result['news_volume_1H'])
            
            # Check ranges
            assert result['sentiment_mean_1H'].between(-1, 1).all()
            assert (result['sentiment_std_1H'].dropna() >= 0).all()
            assert (result['news_volume_1H'] >= 0).all()
            assert (result['source_diversity_1H'] >= 0).all()
    
    def test_error_handling_empty_articles(self, engineer):
        """Test handling of articles with empty content"""
        empty_article = NewsArticle(
            timestamp=datetime.now(),
            title="",
            description="",
            source="Test",
            url="http://test.com",
            symbol="TEST"
        )
        
        sentiment = engineer.analyze_article_sentiment(empty_article)
        # Should handle gracefully with neutral sentiment
        assert sentiment['compound'] == 0.0
    
    def test_single_article_processing(self, engineer):
        """Test processing with just one article"""
        single_data = [{
            'timestamp': datetime(2024, 1, 15, 10, 0, 0),
            'title': 'Single test article',
            'description': 'This is a test',
            'source': 'Test Source',
            'url': 'http://test.com',
            'symbol': 'TEST'
        }]
        
        single_df = pd.DataFrame(single_data)
        result = engineer.create_time_based_features(single_df, "TEST", ['1H'])
        
        assert not result.empty
        # Standard deviation should be 0 for single article
        non_zero_rows = result[result['news_volume_1H'] > 0]
        if not non_zero_rows.empty:
            assert (non_zero_rows['sentiment_std_1H'] == 0.0).all()

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.fixture
    def engineer(self):
        return SentimentFeatureEngineer(use_finbert=False)
    
    def test_full_pipeline_workflow(self, engineer):
        """Test complete workflow from articles to features"""
        base_time = datetime(2024, 1, 15, 9, 0, 0)
        
        # Create realistic test data
        news_events = [
            ("Company beats earnings", "Strong quarterly results", "Reuters", 0),
            ("Analyst upgrade", "Price target increased", "Bloomberg", 2),
            ("Supply chain issues", "Production delays reported", "WSJ", 4),
            ("New strategy announced", "Focus on growth markets", "CNBC", 6),
            ("Regulatory concerns", "Investigation announced", "FT", 8)
        ]
        
        articles_data = []
        for title, desc, source, hour_offset in news_events:
            articles_data.append({
                'timestamp': base_time + timedelta(hours=hour_offset),
                'title': title,
                'description': desc,
                'source': source,
                'url': f'http://example.com/{hour_offset}',
                'symbol': 'AAPL'
            })
        
        articles_df = pd.DataFrame(articles_data)
        
        # Test all feature types
        time_features = engineer.create_time_based_features(articles_df, "AAPL", ['1H', '4H'])
        advanced_features = engineer.create_advanced_features(articles_df, "AAPL")
        
        # Multi-symbol test for cross-symbol features
        multi_data = articles_data.copy()
        for row in articles_data:
            googl_row = row.copy()
            googl_row['symbol'] = 'GOOGL'
            googl_row['title'] = googl_row['title'].replace('Company', 'Google')
            multi_data.append(googl_row)
        
        multi_df = pd.DataFrame(multi_data)
        cross_features = engineer.create_cross_symbol_features(multi_df, "AAPL", ["GOOGL"])
        
        # Verify all feature sets
        assert not time_features.empty
        assert not advanced_features.empty
        assert not cross_features.empty
        
        # Verify consistency
        for features in [time_features, advanced_features, cross_features]:
            assert (features['symbol'] == 'AAPL').all()
            assert pd.api.types.is_datetime64_any_dtype(features['timestamp'])
    
    def test_performance_large_dataset(self, engineer):
        """Test performance with larger dataset"""
        import time
        
        # Create larger dataset
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        large_data = []
        
        for i in range(500):  # 500 articles
            large_data.append({
                'timestamp': base_time + timedelta(minutes=i*10),
                'title': f"Article {i} about market trends",
                'description': f"Detailed analysis {i} of current market conditions",
                'source': f"Source {i % 10}",
                'url': f'http://example.com/article{i}',
                'symbol': 'AAPL'
            })
        
        large_df = pd.DataFrame(large_data)
        
        # Time the processing
        start_time = time.time()
        result = engineer.create_time_based_features(large_df, "AAPL", ['1H'])
        processing_time = time.time() - start_time
        
        assert not result.empty
        assert processing_time < 30.0  # Should complete within 30 seconds

if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_sentiment_refactored.py -v
    pytest.main([__file__, "-v", "--tb=short"])