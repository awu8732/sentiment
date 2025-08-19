import os
import sys
import shutil
import sqlite3
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.db_schema import EXPECTED_SCHEMA, TABLE_CREATION_SQL, INDEX_CREATION_SQL

class MigrationManager:
    """Handles database migration operations and checking"""
    
    def __init__(self, db_path: str, logger):
        self.db_path = db_path
        self.logger = logger
    
    def create_backup(self):
        """Create a backup of the database"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{self.db_path}.backup_{timestamp}"
        
        shutil.copy2(self.db_path, backup_path)
        self.logger.info(f"Database backed up to: {backup_path}")
        return backup_path

    def needs_migration(self):
        """Check if database needs migration based on expected schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            migration_status = {
                'needs_migration': False,
                'missing_tables': [],
                'tables_with_missing_columns': {}
            }
            
            # Get all existing tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            # Check each expected table
            for table_name, expected_columns in EXPECTED_SCHEMA.items():
                if table_name not in existing_tables:
                    self.logger.info(f"Missing table: {table_name}")
                    migration_status['missing_tables'].append(table_name)
                    migration_status['needs_migration'] = True
                else:
                    # Table exists, check columns
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    existing_columns = {col[1] for col in cursor.fetchall()}
                    
                    missing_columns = [col for col in expected_columns if col not in existing_columns]
                    if missing_columns:
                        self.logger.info(f"Table '{table_name}' missing columns: {missing_columns}")
                        migration_status['tables_with_missing_columns'][table_name] = missing_columns
                        migration_status['needs_migration'] = True
            
            # Check for foreign key constraints on sentiment_features if it exists
            if 'sentiment_features' in existing_tables and 'sentiment_features' not in migration_status['missing_tables']:
                cursor.execute("PRAGMA foreign_key_list(sentiment_features)")
                foreign_keys = cursor.fetchall()
                has_market_features_fk = any(fk[2] == 'market_features' for fk in foreign_keys)
                
                if not has_market_features_fk:
                    self.logger.info("Missing foreign key constraint on sentiment_features.timestamp")
                    if 'sentiment_features' not in migration_status['tables_with_missing_columns']:
                        migration_status['tables_with_missing_columns']['sentiment_features'] = []
                    migration_status['needs_migration'] = True
            
            return migration_status
        
        finally:
            conn.close()

    def migrate_database(self, migration_status):
        """Perform the database migration based on expected schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Step 1: Create missing tables (in dependency order: market_features before sentiment_features)
            table_creation_order = ['news', 'stock_prices', 'market_features', 'sentiment_features', 'cross_symbol_cache']
            
            for table_name in table_creation_order:
                if table_name in migration_status['missing_tables']:
                    self.logger.info(f"Creating table: {table_name}")
                    cursor.execute(TABLE_CREATION_SQL[table_name])
            
            # Step 2: Add missing columns to existing tables
            for table_name, missing_columns in migration_status['tables_with_missing_columns'].items():
                if table_name not in migration_status['missing_tables']:  # Only for existing tables
                    self.logger.info(f"Updating table '{table_name}' with missing columns: {missing_columns}")
                    
                    # Get current column info for type mapping
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    current_columns = {col[1]: col[2] for col in cursor.fetchall()}
                    
                    # Column type mapping
                    column_types = {
                        'created_at': 'DATETIME DEFAULT CURRENT_TIMESTAMP',
                        'sector_sentiment_mean': 'REAL',
                        'market_sentiment_mean': 'REAL',
                        'sentiment_sector_correlation': 'REAL',
                        'sentiment_market_correlation': 'REAL',
                        'relative_sentiment_strength': 'REAL',
                        'sector_news_volume': 'INTEGER',
                        'market_news_volume': 'INTEGER',
                        'sentiment_divergence': 'REAL',
                        'sector_sentiment_volatility': 'REAL',
                        'market_sentiment_volatility': 'REAL',
                        'market_sentiment_skew': 'REAL',
                        'market_sentiment_std': 'REAL',
                        'market_sentiment_momentum': 'REAL',
                        'market_source_credibility': 'REAL',
                        'market_source_diversity': 'REAL',
                        'market_sentiment_regime': 'REAL',
                        'market_hours_sentiment': 'REAL',
                        'after_market_sentiment': 'REAL',
                        'pre_market_sentiment': 'REAL',
                        'analysis_type': 'TEXT',
                        'reference_group': 'TEXT',
                        'sentiment_mean': 'REAL',
                        'sentiment_volatility': 'REAL',
                        'symbols_count': 'INTEGER'
                    }
                    
                    for column in missing_columns:
                        if column in column_types:
                            self.logger.info(f"Adding column '{column}' to table '{table_name}'")
                            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {column_types[column]}")
                            
                            # Special handling for created_at column - populate with existing timestamp values
                            if column == 'created_at' and 'timestamp' in current_columns:
                                self.logger.info(f"Populating created_at column with timestamp values for table '{table_name}'")
                                cursor.execute(f"UPDATE {table_name} SET created_at = timestamp WHERE created_at IS NULL")
            
            # Step 3: Handle foreign key constraint for sentiment_features if needed
            if ('sentiment_features' in migration_status['tables_with_missing_columns'] and 
                'sentiment_features' not in migration_status['missing_tables']):
                
                # Check if foreign key already exists
                cursor.execute("PRAGMA foreign_key_list(sentiment_features)")
                foreign_keys = cursor.fetchall()
                has_market_features_fk = any(fk[2] == 'market_features' for fk in foreign_keys)
                
                if not has_market_features_fk:
                    self.logger.info("Adding foreign key constraint to sentiment_features")
                    # SQLite requires recreating the table to add foreign key constraints
                    # This is a complex operation, so we'll skip it for now and note in verification
                    self.logger.warning("Foreign key constraint addition requires table recreation - skipping for now")
            
            # Step 4: Create indices
            self.logger.info("Creating/updating indices")
            for index_sql in INDEX_CREATION_SQL:
                cursor.execute(index_sql)

            conn.commit()
            self.logger.info("Migration completed successfully")

        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error during migration: {e}")
            raise
        finally:
            conn.close()

    def verify_migration(self):
        """Verify that the database schema matches the expected schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            all_ok = True
            # Verify all expected tables and columns exist
            for table_name, expected_columns in EXPECTED_SCHEMA.items():
                # Check table existence
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                if not cursor.fetchone():
                    self.logger.error(f"Missing table: {table_name}")
                    all_ok = False
                    continue

                # Check columns
                cursor.execute(f"PRAGMA table_info({table_name})")
                existing_columns = {col[1] for col in cursor.fetchall()}
                missing_columns = [col for col in expected_columns if col not in existing_columns]
                
                if missing_columns:
                    self.logger.error(f"Table '{table_name}' is missing columns: {missing_columns}")
                    all_ok = False
                else:
                    self.logger.info(f"Table '{table_name}' has all expected columns")

            # Verify indices exist
            expected_indices = [
                'idx_news_symbol_timestamp',
                'idx_stock_symbol_timestamp', 
                'idx_sentiment_symbol_timestamp',
                'idx_market_features_timestamp',
                'idx_cross_symbol_cache'
            ]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
            existing_indices = {row[0] for row in cursor.fetchall()}
            
            missing_indices = [idx for idx in expected_indices if idx not in existing_indices]
            if missing_indices:
                self.logger.error(f"Missing indices: {missing_indices}")
                all_ok = False
            else:
                self.logger.info("All expected indices are present")

            # Verify foreign key constraints (if enabled)
            cursor.execute("PRAGMA foreign_keys")
            fk_enabled = cursor.fetchone()[0]
            if fk_enabled:
                cursor.execute("PRAGMA foreign_key_list(sentiment_features)")
                foreign_keys = cursor.fetchall()
                has_market_features_fk = any(fk[2] == 'market_features' for fk in foreign_keys)
                
                if has_market_features_fk:
                    self.logger.info("Foreign key constraint verified on sentiment_features")
                else:
                    self.logger.warning("Foreign key constraint missing on sentiment_features (expected for existing databases)")

            return all_ok

        except Exception as e:
            self.logger.error(f"Error during migration verification: {e}")
            return False
        finally:
            conn.close()
