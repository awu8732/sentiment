# Expected database schema - backbone for all migration operations
EXPECTED_SCHEMA = {
    "news": [
        "id", "timestamp", "title", "description", "source", "url",
        "symbol", "sentiment_score", "created_at"
    ],
    "stock_prices": [
        "id", "timestamp", "symbol", "open", "high", "low", "close",
        "volume", "adj_close", "created_at"
    ],
    "market_features": [
        "id", "timestamp", "market_sentiment_mean", "market_sentiment_skew",
        "market_sentiment_std", "market_sentiment_momentum", "market_news_volume",
        "market_source_credibility", "market_source_diversity", "market_sentiment_regime",
        "market_hours_sentiment", "pre_market_sentiment", "after_market_sentiment", "created_at"
    ],
    "sentiment_features": [
        "id", "timestamp", "symbol", "sentiment_score", "sentiment_skew", "sentiment_std",
        "sentiment_momentum", "extreme_sentiment_ratio", "sentiment_persistence", 
        "news_flow_intensity", "news_volume", "source_diversity", "created_at"
    ],
    "cross_symbol_features": [
        "id", "timestamp", "symbol", "sector", "sector_sentiment_mean",
        "sector_sentiment_skew", "sector_sentiment_std", "sector_news_volume",
        "relative_sentiment_ratio", "sector_sentiment_correlation", "sector_sentiment_divergence",
        "market_sentiment_correlation", "market_sentiment_divergence", "created_at"
    ]
}

# Table creation SQL statements
TABLE_CREATION_SQL = {
    "news": """
        CREATE TABLE news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            title TEXT,
            description TEXT,
            source TEXT,
            url TEXT UNIQUE,
            symbol TEXT,
            sentiment_score REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "stock_prices": """
        CREATE TABLE stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adj_close REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol)
        )
    """,
    "market_features": """
        CREATE TABLE market_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            market_sentiment_mean REAL,
            market_sentiment_skew REAL,
            market_sentiment_std REAL,
            market_sentiment_momentum REAL,
            market_news_volume INTEGER,
            market_source_credibility REAL,
            market_source_diversity REAL,
            market_sentiment_regime REAL,
            market_hours_sentiment REAL,
            pre_market_sentiment REAL,
            after_market_sentiment REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp)
        )
    """,
    "sentiment_features": """
        CREATE TABLE sentiment_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            sentiment_score REAL,
            sentiment_skew REAL,
            sentiment_std REAL,
            sentiment_momentum REAL,
            extreme_sentiment_ratio REAL,
            sentiment_persistence REAL,
            news_flow_intensity REAL,
            news_volume INTEGER,
            source_diversity REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol)
        )
    """,
    "cross_symbol_features": """
        CREATE TABLE cross_symbol_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            sector TEXT,
            sector_sentiment_mean REAL,
            sector_sentiment_skew REAL,
            sector_sentiment_std REAL,
            sector_news_volume INTEGER,
            relative_sentiment_ratio REAL,
            sector_sentiment_correlation REAL,
            sector_sentiment_divergence REAL,
            market_sentiment_correlation REAL,
            market_sentiment_divergence REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol)
        )
    """
}

# Index creation SQL statements
INDEX_CREATION_SQL = [
    'CREATE INDEX IF NOT EXISTS idx_news_symbol_timestamp ON news(symbol, timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_stock_symbol_timestamp ON stock_prices(symbol, timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_timestamp ON sentiment_features(symbol, timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_market_features_timestamp ON market_features(timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_cross_symbol_features_timestamp ON cross_symbol_features(timestamp, symbol)'
]