#!/usr/bin/env python3
"""
Database migration script to add cross-symbol sentiment features and market features to existing database.

Usage:
    python scripts/migrate_database.py --backup  # Create backup before migration
    python scripts/migrate_database.py --force   # Force migration without prompts
"""

import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config
from src.managers.migration_manager import MigrationManager
from src.utils.logger import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Migrate database to support cross-symbol features and market features')
    parser.add_argument('--backup', action='store_true',
                        help='Create backup before migration')
    parser.add_argument('--force', action='store_true',
                        help='Force migration without confirmation prompts')
    
    args = parser.parse_args()
    config = Config()
    db_path, logger = config.DATABASE_PATH, setup_logging(config)
    migration_manager = MigrationManager(db_path, logger)

    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return 1
    
    try:
        if args.backup:
            backup_path = migration_manager.create_backup()
            logger.info(f"Backup created: {backup_path}")

        # Check if migration is needed
        migration_status = migration_manager.needs_migration()
        if not migration_status['needs_migration']:
            logger.info("Database is already up to date")
            return 0

        # Confirm migration
        if not args.force:
            response = input("Proceed with database migration? [y/N]: ")
            if response.lower() != 'y':
                logger.info("Migration cancelled")
                return 0

        migration_manager.migrate_database(migration_status)
        logger.info("Database migration completed successfully")

        if migration_manager.verify_migration():
            logger.info("Migration verification passed")
        else:
            logger.error("Migration verification failed")
            return 1
        
        return 0
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())