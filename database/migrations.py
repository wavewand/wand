"""
Database Migrations System

Provides database schema versioning, migration management, and automated
database evolution with rollback capabilities.
"""

import hashlib
import importlib.util
import inspect
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, MetaData, String, Table, Text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from utils.error_handling import ErrorCategory, MCPError

from .connection import DatabaseManager, get_database_manager
from .models import Base, generate_uuid


class MigrationStatus(str, Enum):
    """Migration execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationError(MCPError):
    """Migration-specific error."""

    def __init__(self, message: str, migration_name: str = None, operation: str = None):
        super().__init__(
            message,
            category=ErrorCategory.DATABASE_ERROR,
            details={"migration": migration_name, "operation": operation},
        )


@dataclass
class Migration:
    """Migration definition."""

    name: str
    version: str
    description: str
    up_func: Callable[[Session], None]
    down_func: Callable[[Session], None]
    dependencies: List[str] = None
    checksum: str = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


# Migration tracking table
migration_table = Table(
    'schema_migrations',
    MetaData(),
    Column('id', String(36), primary_key=True, default=generate_uuid),
    Column('version', String(50), unique=True, nullable=False, index=True),
    Column('name', String(100), nullable=False),
    Column('description', Text, nullable=True),
    Column('checksum', String(64), nullable=False),
    Column('execution_time_ms', Integer, nullable=True),
    Column('status', String(20), nullable=False, default=MigrationStatus.PENDING.value),
    Column('error_message', Text, nullable=True),
    Column('applied_at', DateTime(timezone=True), nullable=True),
    Column('rolled_back_at', DateTime(timezone=True), nullable=True),
    Column('created_at', DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc)),
    Index('idx_migration_version', 'version'),
    Index('idx_migration_status', 'status'),
    Index('idx_migration_applied', 'applied_at'),
)


class MigrationManager:
    """Database migration management system."""

    def __init__(self, db_manager: DatabaseManager, migrations_dir: str = "database/migrations"):
        self.db_manager = db_manager
        self.migrations_dir = Path(migrations_dir)
        self.logger = logging.getLogger(__name__)
        self._migrations: Dict[str, Migration] = {}
        self._ensure_migrations_table()

    def _ensure_migrations_table(self):
        """Ensure the migrations tracking table exists."""
        try:
            with self.db_manager.get_session() as session:
                # Create the table if it doesn't exist
                migration_table.create(bind=session.connection(), checkfirst=True)
                session.commit()
                self.logger.info("Migration tracking table ensured")
        except Exception as e:
            self.logger.error(f"Failed to create migration table: {e}")
            raise MigrationError(f"Migration table creation failed: {e}", operation="ensure_table")

    def _calculate_checksum(self, migration: Migration) -> str:
        """Calculate checksum for migration integrity verification."""
        content = f"{migration.name}:{migration.version}:{migration.description}"

        # Include function source code if available
        try:
            up_source = inspect.getsource(migration.up_func)
            down_source = inspect.getsource(migration.down_func)
            content += f":{up_source}:{down_source}"
        except (OSError, TypeError):
            # Can't get source, use function names
            content += f":{migration.up_func.__name__}:{migration.down_func.__name__}"

        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def register_migration(self, migration: Migration):
        """Register a migration."""
        migration.checksum = self._calculate_checksum(migration)
        self._migrations[migration.version] = migration
        self.logger.debug(f"Registered migration {migration.version}: {migration.name}")

    def load_migrations_from_directory(self):
        """Load migration files from the migrations directory."""
        if not self.migrations_dir.exists():
            self.logger.warning(f"Migrations directory not found: {self.migrations_dir}")
            return

        migration_files = sorted(self.migrations_dir.glob("*.py"))

        for migration_file in migration_files:
            if migration_file.name.startswith("__"):
                continue

            try:
                self._load_migration_file(migration_file)
            except Exception as e:
                self.logger.error(f"Failed to load migration {migration_file}: {e}")
                raise MigrationError(
                    f"Failed to load migration file: {e}", migration_name=migration_file.name, operation="load_file"
                )

    def _load_migration_file(self, file_path: Path):
        """Load a single migration file."""
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Extract migration information
        if not hasattr(module, 'migration'):
            raise MigrationError(f"Migration file {file_path} missing 'migration' definition")

        migration_def = module.migration
        migration = Migration(
            name=migration_def.get('name', file_path.stem),
            version=migration_def.get('version'),
            description=migration_def.get('description', ''),
            up_func=module.up,
            down_func=module.down,
            dependencies=migration_def.get('dependencies', []),
        )

        if not migration.version:
            raise MigrationError(f"Migration {file_path} missing version")

        self.register_migration(migration)

    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration execution history."""
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(migration_table.select().order_by(migration_table.c.applied_at.desc()))
                return [dict(row._mapping) for row in result]
        except Exception as e:
            self.logger.error(f"Failed to get migration history: {e}")
            raise MigrationError(f"Failed to retrieve migration history: {e}", operation="get_history")

    def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations in dependency order."""
        try:
            with self.db_manager.get_session() as session:
                # Get applied migrations
                result = session.execute(
                    migration_table.select().where(migration_table.c.status == MigrationStatus.COMPLETED.value)
                )
                applied_versions = {row.version for row in result}

                # Find pending migrations
                pending = []
                for version, migration in self._migrations.items():
                    if version not in applied_versions:
                        pending.append(migration)

                # Sort by version (assuming semantic versioning or timestamp-based)
                pending.sort(key=lambda m: m.version)

                # Validate dependencies
                for migration in pending:
                    for dep in migration.dependencies:
                        if dep not in applied_versions and dep not in [
                            m.version for m in pending[: pending.index(migration)]
                        ]:
                            raise MigrationError(
                                f"Migration {migration.version} depends on {dep} which is not applied",
                                migration_name=migration.name,
                                operation="validate_dependencies",
                            )

                return pending

        except Exception as e:
            if isinstance(e, MigrationError):
                raise
            self.logger.error(f"Failed to get pending migrations: {e}")
            raise MigrationError(f"Failed to get pending migrations: {e}", operation="get_pending")

    def _record_migration_start(self, session: Session, migration: Migration):
        """Record migration start in tracking table."""
        session.execute(
            migration_table.insert().values(
                version=migration.version,
                name=migration.name,
                description=migration.description,
                checksum=migration.checksum,
                status=MigrationStatus.RUNNING.value,
                created_at=datetime.now(timezone.utc),
            )
        )
        session.commit()

    def _record_migration_success(self, session: Session, migration: Migration, execution_time_ms: int):
        """Record successful migration completion."""
        session.execute(
            migration_table.update()
            .where(migration_table.c.version == migration.version)
            .values(
                status=MigrationStatus.COMPLETED.value,
                execution_time_ms=execution_time_ms,
                applied_at=datetime.now(timezone.utc),
            )
        )
        session.commit()

    def _record_migration_failure(self, session: Session, migration: Migration, error: str):
        """Record migration failure."""
        session.execute(
            migration_table.update()
            .where(migration_table.c.version == migration.version)
            .values(status=MigrationStatus.FAILED.value, error_message=error)
        )
        session.commit()

    def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration."""
        self.logger.info(f"Applying migration {migration.version}: {migration.name}")
        start_time = time.time()

        try:
            with self.db_manager.get_session() as session:
                # Record migration start
                self._record_migration_start(session, migration)

                try:
                    # Execute the migration
                    migration.up_func(session)
                    session.commit()

                    # Record success
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    self._record_migration_success(session, migration, execution_time_ms)

                    self.logger.info(f"Migration {migration.version} applied successfully in {execution_time_ms}ms")
                    return True

                except Exception as e:
                    session.rollback()
                    error_msg = str(e)

                    # Record failure
                    self._record_migration_failure(session, migration, error_msg)

                    self.logger.error(f"Migration {migration.version} failed: {error_msg}")
                    raise MigrationError(
                        f"Migration {migration.version} execution failed: {error_msg}",
                        migration_name=migration.name,
                        operation="apply",
                    )

        except Exception as e:
            if isinstance(e, MigrationError):
                raise
            raise MigrationError(
                f"Migration {migration.version} failed: {e}", migration_name=migration.name, operation="apply"
            )

    def rollback_migration(self, migration: Migration) -> bool:
        """Rollback a migration."""
        self.logger.info(f"Rolling back migration {migration.version}: {migration.name}")

        try:
            with self.db_manager.get_session() as session:
                # Execute rollback
                migration.down_func(session)
                session.commit()

                # Update migration record
                session.execute(
                    migration_table.update()
                    .where(migration_table.c.version == migration.version)
                    .values(status=MigrationStatus.ROLLED_BACK.value, rolled_back_at=datetime.now(timezone.utc))
                )
                session.commit()

                self.logger.info(f"Migration {migration.version} rolled back successfully")
                return True

        except Exception as e:
            self.logger.error(f"Migration {migration.version} rollback failed: {e}")
            raise MigrationError(
                f"Migration {migration.version} rollback failed: {e}",
                migration_name=migration.name,
                operation="rollback",
            )

    def migrate(self, target_version: str = None) -> List[str]:
        """Run all pending migrations up to target version."""
        pending_migrations = self.get_pending_migrations()

        if target_version:
            # Filter migrations up to target version
            target_index = -1
            for i, migration in enumerate(pending_migrations):
                if migration.version == target_version:
                    target_index = i
                    break

            if target_index == -1:
                raise MigrationError(f"Target version {target_version} not found", operation="migrate")

            pending_migrations = pending_migrations[: target_index + 1]

        if not pending_migrations:
            self.logger.info("No pending migrations to apply")
            return []

        applied_migrations = []

        for migration in pending_migrations:
            try:
                if self.apply_migration(migration):
                    applied_migrations.append(migration.version)
            except MigrationError:
                self.logger.error(f"Migration failed at {migration.version}, stopping migration process")
                break

        self.logger.info(f"Applied {len(applied_migrations)} migrations: {applied_migrations}")
        return applied_migrations

    def get_current_version(self) -> Optional[str]:
        """Get the current database schema version."""
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(
                    migration_table.select()
                    .where(migration_table.c.status == MigrationStatus.COMPLETED.value)
                    .order_by(migration_table.c.applied_at.desc())
                    .limit(1)
                )
                row = result.first()
                return row.version if row else None
        except Exception as e:
            self.logger.error(f"Failed to get current version: {e}")
            return None

    def validate_migrations(self) -> Dict[str, Any]:
        """Validate migration integrity and consistency."""
        issues = []

        try:
            with self.db_manager.get_session() as session:
                # Check for checksum mismatches
                result = session.execute(migration_table.select())
                applied_migrations = {row.version: row for row in result}

                for version, migration in self._migrations.items():
                    if version in applied_migrations:
                        applied = applied_migrations[version]
                        if applied.checksum != migration.checksum:
                            issues.append(
                                {
                                    'type': 'checksum_mismatch',
                                    'version': version,
                                    'expected': migration.checksum,
                                    'actual': applied.checksum,
                                }
                            )

                # Check for missing migrations
                for version in applied_migrations:
                    if version not in self._migrations:
                        issues.append(
                            {
                                'type': 'missing_migration',
                                'version': version,
                                'message': 'Applied migration not found in current migration set',
                            }
                        )

                return {
                    'valid': len(issues) == 0,
                    'issues': issues,
                    'total_migrations': len(self._migrations),
                    'applied_migrations': len(
                        [m for m in applied_migrations.values() if m.status == MigrationStatus.COMPLETED.value]
                    ),
                }

        except Exception as e:
            self.logger.error(f"Migration validation failed: {e}")
            raise MigrationError(f"Migration validation failed: {e}", operation="validate")


# Global migration manager instance
_migration_manager: Optional[MigrationManager] = None


def initialize_migrations(db_manager: DatabaseManager, migrations_dir: str = "database/migrations") -> MigrationManager:
    """Initialize global migration manager."""
    global _migration_manager
    _migration_manager = MigrationManager(db_manager, migrations_dir)
    return _migration_manager


def get_migration_manager() -> MigrationManager:
    """Get global migration manager instance."""
    if _migration_manager is None:
        raise MigrationError("Migration manager not initialized", operation="get_manager")
    return _migration_manager


def create_migration(name: str, description: str = "") -> str:
    """Create a new migration file template."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"{timestamp}_{name.lower().replace(' ', '_')}"
    filename = f"{version}.py"

    migrations_dir = Path("database/migrations")
    migrations_dir.mkdir(exist_ok=True)

    template = f'''"""
Migration: {name}
Description: {description}
Created: {datetime.now().isoformat()}
"""

from sqlalchemy.orm import Session


# Migration metadata
migration = {{
    'name': '{name}',
    'version': '{version}',
    'description': '{description}',
    'dependencies': []  # List of migration versions this depends on
}}


def up(session: Session):
    """Apply migration changes."""
    # Add your schema changes here
    # Example:
    # session.execute(text("ALTER TABLE users ADD COLUMN new_field VARCHAR(50)"))
    pass


def down(session: Session):
    """Rollback migration changes."""
    # Add rollback logic here
    # Example:
    # session.execute(text("ALTER TABLE users DROP COLUMN new_field"))
    pass
'''

    migration_file = migrations_dir / filename
    migration_file.write_text(template)

    return str(migration_file)


def run_migrations(target_version: str = None) -> List[str]:
    """Run pending migrations."""
    manager = get_migration_manager()
    return manager.migrate(target_version)


def rollback_migration(version: str) -> bool:
    """Rollback specific migration."""
    manager = get_migration_manager()
    if version not in manager._migrations:
        raise MigrationError(f"Migration {version} not found", operation="rollback")

    migration = manager._migrations[version]
    return manager.rollback_migration(migration)


def get_migration_status() -> Dict[str, Any]:
    """Get migration system status."""
    manager = get_migration_manager()

    return {
        'current_version': manager.get_current_version(),
        'pending_migrations': [m.version for m in manager.get_pending_migrations()],
        'migration_history': manager.get_migration_history()[-10:],  # Last 10
        'validation': manager.validate_migrations(),
    }
