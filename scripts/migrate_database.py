#!/usr/bin/env python3
"""
Database migration script to add cross-symbol sentiment features to existing database. (More to come later)

Usage:
    python scripts/migrate_database.py --backup  # Create backup before migration
    python scripts/migrate_database.py --force   # Force migration without prompts
"""

import sys
import os
import sqlite3
import argparse
import shutil
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from src.utils.logger import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Migrate databse to support cross-symbol features')
    parser.add_argument('--backup', action='store_true',
                        help='Create backup before migration')
    parser.add_argument('--force', action='store_true',
                        help='Force migration without confirmation prompts')
    
    args = parser.parse_args()
    config = Config()
    logger = setup_logging(config)
    db_path = config.DATABASE_PATH

    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return 1
    
    try:
        # Create backup if requested
        if args.backup:
            backup_path = create_backup(db_path, logger)
            logger.info(f"Backup created: {backup_path}")

        # Check if migration is needed
        if not needs_migration(db_path, logger):
            logger.info("Database is already up to date")
            return 0

        # Confirm migration
        if not args.force:
            response = input("Proceed with database migration? [y/N]: ")
            if response.lower() != 'y':
                logger.info("Migration cancelled")
                return 0
        
        # Perform / Verify migration
        migrate_database(db_path, logger)
        logger.info("Database migration completed successfully")
        if verify_migration(db_path, logger):
            logger.info("Migration verification passed")
        else:
            logger.error("Migration verification failed")
            return 1
        
        return 0
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1

    
def create_backup(db_path, logger):
    """Create a backup of the database"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{db_path}.backup_{timestamp}"
    
    shutil.copy2(db_path, backup_path)
    logger.info(f"Database backed up to: {backup_path}")
    return backup_path

def needs_migration(db_path, logger):
    """Check if database needs migration"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if cross-symbol columns exist in sentiment_features
        cursor.execute("PRAGMA table_info(sentiment_features)")
        columns = [col[1] for col in cursor.fetchall()]

        cross_symbol_columns = [
            'sector_sentiment_mean',
            'market_sentiment_mean',
            'sentiment_sector_correlation'
        ]

        missing_columns = [col for col in cross_symbol_columns if col not in columns]

        # Check if cross_symbol_cache table exists
        cursor.execute("""
                       SELECT name FROM sqlite_master
                       WHERE type = 'table' AND name = 'cross_symbol_cache'
        """)
        cache_table_exists = len(cursor.fetchall()) > 0
        
        needs_migration = len(missing_columns) > 0 or not cache_table_exists
        if needs_migration:
            logger.info(f"Migration needed. Missing columns: {missing_columns}")
            logger.info(f"Cache table exists: {cache_table_exists}")
        
        return needs_migration
    
    finally:
        conn.close()

def migrate_database(db_path, logger):
    """Perform the database migration"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        new_columns = [
            ('sector_sentiment_mean', 'REAL'),
            ('market_sentiment_mean', 'REAL'),
            ('sentiment_sector_correlation', 'REAL'),
            ('sentiment_market_correlation', 'REAL'),
            ('relative_sentiment_strength', 'REAL'),
            ('sector_news_volume', 'INTEGER'),
            ('market_news_volume', 'INTEGER'),
            ('sentiment_divergence', 'REAL'),
            ('sector_sentiment_volatility', 'REAL'),
            ('market_sentiment_volatility', 'REAL'),
        ]

        cursor.execute("PRAGMA table_info(sentiment_features)")
        columns = [col[1] for col in cursor.fetchall()]
        
        for col_name, col_type in new_columns:
            if col_name not in columns:
                logger.info(f"Adding column '{col_name}' to sentiment_features table")
                cursor.execute(f"ALTER TABLE sentiment_features ADD COLUMN {col_name} {col_type}")

         # Create cross_symbol_cache table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cross_symbol_cache (
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
        """)

        conn.commit()

    except Exception as e:
        conn.rollback()
        logger.error(f"Error during migration: {e}")
        raise
    finally:
        conn.close()

def verify_migration(db_path, logger):
    """Verify that the database schema matches the expected migrated state."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        expected_schema = {
            "news": [
                "id", "timestamp", "title", "description", "source", "url",
                "symbol", "sentiment_score", "created_at"
            ],
            "stock_prices": [
                "id", "timestamp", "symbol", "open", "high", "low", "close",
                "volume", "adj_close", "created_at"
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

        all_ok = True

        for table, expected_cols in expected_schema.items():
            # Check table existence
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name=?
            """, (table,))
            if not cursor.fetchone():
                logger.error(f"Missing table: {table}")
                all_ok = False
                continue

            # Check columns
            cursor.execute(f"PRAGMA table_info({table})")
            existing_cols = {col[1] for col in cursor.fetchall()}
            missing_cols = [col for col in expected_cols if col not in existing_cols]
            if missing_cols:
                logger.error(f"Table '{table}' is missing columns: {missing_cols}")
                all_ok = False
            else:
                logger.info(f"Table '{table}' has all expected columns.")

        return all_ok

    except Exception as e:
        logger.error(f"Error during migration verification: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    exit(main())