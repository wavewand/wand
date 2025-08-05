"""
Database Initialization

Provides database setup, initialization, and bootstrap functionality
for the MCP platform with proper error handling and logging.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import DatabaseSettings
from observability.logging import get_logger

from .connection import DatabaseManager, initialize_database
from .migrations import MigrationManager, initialize_migrations
from .models import Base
from .repositories import (
    APIKeyRepository,
    BatchRepository,
    BenchmarkRepository,
    CacheRepository,
    DocumentRepository,
    ErrorLogRepository,
    PerformanceRepository,
    QueryRepository,
    UserRepository,
)


class DatabaseInitializer:
    """Database initialization and setup manager."""

    def __init__(self, settings: DatabaseSettings):
        self.settings = settings
        self.logger = get_logger(__name__)
        self.db_manager: Optional[DatabaseManager] = None
        self.migration_manager: Optional[MigrationManager] = None
        self.repositories: Dict[str, Any] = {}

    def initialize(self, run_migrations: bool = True, create_tables: bool = True) -> DatabaseManager:
        """Initialize database with all components."""
        try:
            self.logger.info("Starting database initialization")

            # Initialize database manager
            self.db_manager = initialize_database(self.settings)
            self.logger.info("Database manager initialized")

            # Test connection
            self.db_manager.connect()
            self.logger.info("Database connection established")

            # Create tables if requested
            if create_tables:
                self.db_manager.create_tables()
                self.logger.info("Database tables created")

            # Initialize migration system
            if run_migrations:
                self.migration_manager = initialize_migrations(self.db_manager, "database/migrations")

                # Load migrations from directory
                self.migration_manager.load_migrations_from_directory()
                self.logger.info("Migration system initialized")

                # Run pending migrations
                applied = self.migration_manager.migrate()
                if applied:
                    self.logger.info(f"Applied migrations: {applied}")
                else:
                    self.logger.info("No pending migrations to apply")

            # Initialize repositories
            self._initialize_repositories()
            self.logger.info("Repositories initialized")

            # Perform health check
            health = self.db_manager.health_check()
            if health['status'] == 'healthy':
                self.logger.info(f"Database health check passed - response time: {health['response_time_ms']:.2f}ms")
            else:
                self.logger.warning(f"Database health check failed: {health.get('error', 'Unknown error')}")

            self.logger.info("Database initialization completed successfully")
            return self.db_manager

        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}", exc_info=True)
            raise

    def _initialize_repositories(self):
        """Initialize all repository instances."""
        self.repositories = {
            'users': UserRepository(),
            'api_keys': APIKeyRepository(),
            'documents': DocumentRepository(),
            'queries': QueryRepository(),
            'batches': BatchRepository(),
            'benchmarks': BenchmarkRepository(),
            'errors': ErrorLogRepository(),
            'performance': PerformanceRepository(),
            'cache': CacheRepository(),
        }

    def get_repository(self, name: str):
        """Get repository instance by name."""
        if name not in self.repositories:
            raise ValueError(f"Repository '{name}' not found. Available: {list(self.repositories.keys())}")
        return self.repositories[name]

    def create_initial_data(self):
        """Create initial application data."""
        if not self.db_manager:
            raise RuntimeError("Database not initialized")

        try:
            with self.db_manager.get_session() as session:
                # Create default admin user if it doesn't exist
                user_repo = self.get_repository('users')
                admin_user = user_repo.get_by_username(session, 'admin')

                if not admin_user:
                    import os
                    import secrets

                    from security.auth import hash_password

                    # Generate secure random password or use environment variable
                    admin_password = os.environ.get('WAND_ADMIN_PASSWORD')
                    if not admin_password:
                        admin_password = secrets.token_urlsafe(16)
                        self.logger.warning(f"Generated admin password: {admin_password}")
                        self.logger.warning("Set WAND_ADMIN_PASSWORD environment variable to use a custom password")

                    admin_user = user_repo.create(
                        session,
                        username='admin',
                        email='admin@localhost',
                        password_hash=hash_password(admin_password),
                        role='admin',
                        is_active=True,
                    )
                    self.logger.info(f"Created default admin user: {admin_user.id}")

                # Create initial frameworks
                framework_repo = self.get_repository('documents')  # Framework will be handled by documents for now

                session.commit()
                self.logger.info("Initial data created successfully")

        except Exception as e:
            self.logger.error(f"Failed to create initial data: {e}", exc_info=True)
            raise

    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information."""
        if not self.db_manager:
            return {'status': 'not_initialized'}

        try:
            info = {
                'status': 'initialized',
                'connection_info': self.db_manager.get_connection_info(),
                'health_check': self.db_manager.health_check(),
                'table_info': self.db_manager.get_table_info(),
            }

            if self.migration_manager:
                from .migrations import get_migration_status

                info['migration_status'] = get_migration_status()

            return info

        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}")
            return {'status': 'error', 'error': str(e)}

    def cleanup(self):
        """Cleanup database connections."""
        if self.db_manager:
            try:
                self.db_manager.close()
                self.logger.info("Database connections closed")
            except Exception as e:
                self.logger.error(f"Error during database cleanup: {e}")

    async def async_cleanup(self):
        """Cleanup async database connections."""
        if self.db_manager:
            try:
                await self.db_manager.async_close()
                self.logger.info("Async database connections closed")
            except Exception as e:
                self.logger.error(f"Error during async database cleanup: {e}")


def initialize_database_system(
    settings: DatabaseSettings,
    run_migrations: bool = True,
    create_tables: bool = True,
    create_initial_data: bool = False,
) -> DatabaseInitializer:
    """
    Initialize the complete database system.

    Args:
        settings: Database configuration settings
        run_migrations: Whether to run pending migrations
        create_tables: Whether to create database tables
        create_initial_data: Whether to create initial application data

    Returns:
        DatabaseInitializer instance
    """
    initializer = DatabaseInitializer(settings)
    initializer.initialize(run_migrations, create_tables)

    if create_initial_data:
        initializer.create_initial_data()

    return initializer


def quick_database_setup(database_url: str = "sqlite:///mcp_platform.db") -> DatabaseInitializer:
    """Quick database setup with sensible defaults for development."""
    settings = DatabaseSettings(
        url=database_url,
        pool_size=5 if not database_url.startswith('sqlite') else 1,
        echo=False,  # Set to True for SQL debugging
    )

    return initialize_database_system(
        settings=settings, run_migrations=True, create_tables=True, create_initial_data=True
    )
