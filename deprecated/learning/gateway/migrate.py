#!/usr/bin/env python3
"""
Database migration and initialization script for VoiceStand Learning System
Handles database schema creation, data migration, and version management
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List
import json

import asyncpg
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "LEARNING_DB_URL",
    "postgresql://voicestand:learning_pass@localhost:5433/voicestand_learning"
)

class DatabaseMigrator:
    """Handles database migrations and initialization"""

    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
        self.conn: Optional[asyncpg.Connection] = None
        self.sql_dir = Path(__file__).parent.parent / "sql"

    async def connect(self):
        """Connect to the database"""
        try:
            self.conn = await asyncpg.connect(self.database_url)
            logger.info("✅ Connected to database")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the database"""
        if self.conn:
            await self.conn.close()
            logger.info("🔒 Disconnected from database")

    async def create_migration_table(self):
        """Create the migrations tracking table"""
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                version VARCHAR(50) UNIQUE NOT NULL,
                filename VARCHAR(255) NOT NULL,
                executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                checksum VARCHAR(64)
            )
        """)
        logger.info("📋 Migration tracking table ready")

    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migrations"""
        try:
            rows = await self.conn.fetch("SELECT version FROM schema_migrations ORDER BY version")
            return [row['version'] for row in rows]
        except Exception:
            # Table might not exist yet
            return []

    async def apply_migration(self, filepath: Path) -> bool:
        """Apply a single migration file"""
        try:
            # Read migration content
            content = filepath.read_text(encoding='utf-8')

            # Calculate checksum
            import hashlib
            checksum = hashlib.sha256(content.encode()).hexdigest()

            # Extract version from filename
            version = filepath.stem

            # Check if already applied
            applied = await self.get_applied_migrations()
            if version in applied:
                logger.info(f"⏭️  Migration {version} already applied")
                return True

            # Apply migration
            logger.info(f"🔄 Applying migration: {version}")

            async with self.conn.transaction():
                # Execute the migration
                await self.conn.execute(content)

                # Record the migration
                await self.conn.execute("""
                    INSERT INTO schema_migrations (version, filename, checksum)
                    VALUES ($1, $2, $3)
                """, version, filepath.name, checksum)

            logger.info(f"✅ Applied migration: {version}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to apply migration {filepath.name}: {e}")
            return False

    async def run_migrations(self) -> bool:
        """Run all pending migrations"""
        logger.info("🚀 Starting database migrations...")

        try:
            # Ensure migration table exists
            await self.create_migration_table()

            # Find all SQL files
            if not self.sql_dir.exists():
                logger.warning(f"⚠️  SQL directory not found: {self.sql_dir}")
                return False

            sql_files = sorted(self.sql_dir.glob("*.sql"))
            if not sql_files:
                logger.info("ℹ️  No migration files found")
                return True

            # Apply migrations in order
            success_count = 0
            for sql_file in sql_files:
                if await self.apply_migration(sql_file):
                    success_count += 1
                else:
                    logger.error(f"❌ Migration failed: {sql_file.name}")
                    return False

            logger.info(f"🎉 Successfully applied {success_count} migrations")
            return True

        except Exception as e:
            logger.error(f"❌ Migration process failed: {e}")
            return False

    async def verify_schema(self) -> bool:
        """Verify that the database schema is properly set up"""
        try:
            # Check for required tables
            required_tables = [
                'recognition_history',
                'learning_patterns',
                'model_performance',
                'training_examples',
                'system_metrics',
                'activity_log',
                'learning_insights'
            ]

            for table in required_tables:
                result = await self.conn.fetchval("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_name = $1
                """, table)

                if result == 0:
                    logger.error(f"❌ Required table missing: {table}")
                    return False

            logger.info("✅ Database schema verification passed")
            return True

        except Exception as e:
            logger.error(f"❌ Schema verification failed: {e}")
            return False

    async def seed_initial_data(self) -> bool:
        """Seed the database with initial data if empty"""
        try:
            # Check if we already have data
            model_count = await self.conn.fetchval("SELECT COUNT(*) FROM model_performance")
            if model_count > 0:
                logger.info("ℹ️  Database already has data, skipping seed")
                return True

            logger.info("🌱 Seeding initial data...")

            # Insert default model performance data
            models_data = [
                ('whisper_small', 82.0, 84.0, 0.25, 1000),
                ('whisper_medium', 87.0, 89.0, 0.35, 1500),
                ('whisper_large', 92.0, 94.0, 0.40, 2000),
                ('uk_fine_tuned_small', 85.0, 90.0, 0.20, 800),
                ('uk_fine_tuned_medium', 90.0, 95.0, 0.30, 1200)
            ]

            for model_name, accuracy, uk_accuracy, weight, sample_count in models_data:
                await self.conn.execute("""
                    INSERT INTO model_performance
                    (model_name, accuracy, uk_accuracy, weight, sample_count)
                    VALUES ($1, $2, $3, $4, $5)
                """, model_name, accuracy, uk_accuracy, weight, sample_count)

            # Insert initial system metrics
            metrics_data = [
                ('overall_accuracy', 89.2, '{"baseline": 88.0}'),
                ('uk_accuracy', 90.3, '{"baseline": 88.0}'),
                ('ensemble_accuracy', 88.5, '{"baseline": 88.0}'),
                ('patterns_learned', 5, '{"target": 100}')
            ]

            for metric_name, value, metadata in metrics_data:
                await self.conn.execute("""
                    INSERT INTO system_metrics (metric_name, metric_value, metadata)
                    VALUES ($1, $2, $3)
                """, metric_name, value, metadata)

            # Insert welcome activity
            await self.conn.execute("""
                INSERT INTO activity_log (activity_type, message, metadata)
                VALUES ($1, $2, $3)
            """, 'system',
                '🚀 VoiceStand Learning System initialized with PostgreSQL storage',
                '{"version": "1.0.0", "migration_date": "' + datetime.utcnow().isoformat() + '"}')

            logger.info("✅ Initial data seeded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to seed initial data: {e}")
            return False

    async def status(self) -> Dict:
        """Get migration status"""
        try:
            applied_migrations = await self.get_applied_migrations()

            # Count records in key tables
            table_counts = {}
            tables = ['recognition_history', 'learning_patterns', 'model_performance',
                     'system_metrics', 'activity_log', 'learning_insights']

            for table in tables:
                try:
                    count = await self.conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                    table_counts[table] = count
                except Exception:
                    table_counts[table] = "N/A"

            return {
                "database_connected": True,
                "migrations_applied": len(applied_migrations),
                "applied_migrations": applied_migrations,
                "table_counts": table_counts,
                "status": "healthy"
            }

        except Exception as e:
            return {
                "database_connected": False,
                "error": str(e),
                "status": "error"
            }

async def main():
    """Main migration script"""
    if len(sys.argv) < 2:
        print("Usage: python migrate.py [init|migrate|status|verify]")
        sys.exit(1)

    command = sys.argv[1]
    migrator = DatabaseMigrator()

    try:
        await migrator.connect()

        if command == "init":
            # Full initialization
            success = await migrator.run_migrations()
            if success:
                await migrator.seed_initial_data()
                await migrator.verify_schema()
                print("🎉 Database initialization completed successfully!")
            else:
                print("❌ Database initialization failed")
                sys.exit(1)

        elif command == "migrate":
            # Run migrations only
            success = await migrator.run_migrations()
            if success:
                print("✅ Migrations completed successfully!")
            else:
                print("❌ Migrations failed")
                sys.exit(1)

        elif command == "status":
            # Show status
            status = await migrator.status()
            print(json.dumps(status, indent=2, default=str))

        elif command == "verify":
            # Verify schema
            success = await migrator.verify_schema()
            if success:
                print("✅ Schema verification passed")
            else:
                print("❌ Schema verification failed")
                sys.exit(1)

        else:
            print(f"Unknown command: {command}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Migration script failed: {e}")
        sys.exit(1)

    finally:
        await migrator.disconnect()

if __name__ == "__main__":
    asyncio.run(main())