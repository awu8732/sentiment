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
        "id", "timestamp", "symbol", "sentiment_score", "sentiment_momentum",
        "news_volume", "source_diversity", "sector_sentiment_mean", "market_sentiment_mean", 
        "sentiment_sector_correlation", "sentiment_market_correlation", "relative_sentiment_strength",
        "sector_news_volume", "market_news_volume", "sentiment_divergence",
        "sector_sentiment_volatility", "market_sentiment_volatility",
        "created_at"
    ],
    "cross_symbol_cache": [
        "id", "timestamp", "analysis_type", "reference_group",
        "sentiment_mean", "sentiment_volatility", "news_volume",
        "symbols_count", "created_at"
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
            timestamp DATETIME UNIQUE,
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
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "sentiment_features": """
        CREATE TABLE sentiment_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            sentiment_score REAL,
            sentiment_momentum REAL,
            news_volume INTEGER,
            source_diversity REAL,
            sector_sentiment_mean REAL,
            market_sentiment_mean REAL,
            sentiment_sector_correlation REAL,
            sentiment_market_correlation REAL,
            relative_sentiment_strength REAL,
            sector_news_volume INTEGER,
            market_news_volume INTEGER,
            sentiment_divergence REAL,
            sector_sentiment_volatility REAL,
            market_sentiment_volatility REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol),
            FOREIGN KEY (timestamp) REFERENCES market_features(timestamp) ON DELETE RESTRICT
        )
    """,
    "cross_symbol_cache": """
        CREATE TABLE cross_symbol_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            analysis_type TEXT, -- 'sector' or 'market'
            reference_group TEXT, -- sector name or 'market'
            sentiment_mean REAL,
            sentiment_volatility REAL,
            news_volume INTEGER,
            symbols_count INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, analysis_type, reference_group)
        )
    """
}

# Index creation SQL statements
INDEX_CREATION_SQL = [
    'CREATE INDEX IF NOT EXISTS idx_news_symbol_timestamp ON news(symbol, timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_stock_symbol_timestamp ON stock_prices(symbol, timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_timestamp ON sentiment_features(symbol, timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_market_features_timestamp ON market_features(timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_cross_symbol_cache ON cross_symbol_cache(timestamp, analysis_type, reference_group)'
]