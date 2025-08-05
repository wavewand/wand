"""
Database Migration Manager

Handles database migrations for the Wand system.
"""

import os
import importlib.util
import logging
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Represents a database migration"""
    name: str
    version: str
    up: Callable
    down: Optional[Callable] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None


class MigrationManager:
    """Manages database migrations"""

    def __init__(self, migrations_dir: str = None):
        """
        Initialize the migration manager

        Args:
            migrations_dir: Directory containing migration files
        """
        if migrations_dir is None:
            migrations_dir = Path(__file__).parent
        self.migrations_dir = Path(migrations_dir)
        self.applied_migrations = set()

    def get_migrations(self) -> List[str]:
        """Get list of migration files in order"""
        migrations = []
        if self.migrations_dir.exists():
            for file in sorted(self.migrations_dir.glob("*.py")):
                if file.name != "__init__.py" and not file.name.startswith("_"):
                    migrations.append(file.stem)
        return migrations

    def load_migration(self, migration_name: str):
        """Load a migration module"""
        migration_path = self.migrations_dir / f"{migration_name}.py"
        if not migration_path.exists():
            raise FileNotFoundError(f"Migration {migration_name} not found")

        spec = importlib.util.spec_from_file_location(migration_name, migration_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    async def run_migration(self, migration_name: str, connection=None):
        """Run a single migration"""
        try:
            module = self.load_migration(migration_name)
            if hasattr(module, 'upgrade'):
                await module.upgrade(connection)
                self.applied_migrations.add(migration_name)
                logger.info(f"Applied migration: {migration_name}")
            else:
                logger.warning(f"Migration {migration_name} has no upgrade function")
        except Exception as e:
            logger.error(f"Failed to run migration {migration_name}: {e}")
            raise

    async def rollback_migration(self, migration_name: str, connection=None):
        """Rollback a single migration"""
        try:
            module = self.load_migration(migration_name)
            if hasattr(module, 'downgrade'):
                await module.downgrade(connection)
                self.applied_migrations.discard(migration_name)
                logger.info(f"Rolled back migration: {migration_name}")
            else:
                logger.warning(f"Migration {migration_name} has no downgrade function")
        except Exception as e:
            logger.error(f"Failed to rollback migration {migration_name}: {e}")
            raise

    async def run_all_migrations(self, connection=None):
        """Run all pending migrations"""
        migrations = self.get_migrations()
        for migration in migrations:
            if migration not in self.applied_migrations:
                await self.run_migration(migration, connection)

    def get_status(self) -> Dict[str, Any]:
        """Get migration status"""
        all_migrations = self.get_migrations()
        return {
            "total": len(all_migrations),
            "applied": len(self.applied_migrations),
            "pending": len(all_migrations) - len(self.applied_migrations),
            "migrations": {
                m: m in self.applied_migrations for m in all_migrations
            }
        }


def create_migration(name: str, migrations_dir: str = None) -> str:
    """
    Create a new migration file

    Args:
        name: Migration description
        migrations_dir: Directory to create migration in

    Returns:
        Path to created migration file
    """
    if migrations_dir is None:
        migrations_dir = Path(__file__).parent

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{name}.py"
    filepath = Path(migrations_dir) / filename

    template = '''"""
Migration: {name}
Created: {timestamp}
"""

async def upgrade(connection):
    """Apply migration"""
    # TODO: Add upgrade logic here
    pass

async def downgrade(connection):
    """Rollback migration"""
    # TODO: Add downgrade logic here
    pass
'''.format(name=name, timestamp=timestamp)

    filepath.write_text(template)
    logger.info(f"Created migration: {filepath}")
    return str(filepath)


async def run_migrations(connection=None, migrations_dir: str = None):
    """Run all pending migrations"""
    manager = MigrationManager(migrations_dir)
    await manager.run_all_migrations(connection)
    return manager.get_status()


async def rollback_migration(migration_name: str, connection=None, migrations_dir: str = None):
    """Rollback a specific migration"""
    manager = MigrationManager(migrations_dir)
    await manager.rollback_migration(migration_name, connection)
    return manager.get_status()


def get_migration_status(migrations_dir: str = None) -> Dict[str, Any]:
    """Get current migration status"""
    manager = MigrationManager(migrations_dir)
    return manager.get_status()


async def initialize_migrations(connection=None, migrations_dir: str = None):
    """
    Initialize the migrations system

    Args:
        connection: Database connection
        migrations_dir: Directory containing migrations

    Returns:
        True if initialization successful
    """
    try:
        manager = MigrationManager(migrations_dir)
        # Create migrations table if needed
        if connection:
            # In a real implementation, this would create a migrations tracking table
            # For now, just return success
            pass
        logger.info("Migrations system initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize migrations: {e}")
        return False
