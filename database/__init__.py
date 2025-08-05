"""
Database Package

Provides comprehensive database integration with SQLAlchemy ORM,
migrations, connection pooling, and data access layers.
"""

from .connection import (
    DatabaseManager,
    create_tables,
    drop_tables,
    get_async_session,
    get_database_manager,
    get_session,
)
from .migrations import MigrationManager, create_migration, get_migration_status, rollback_migration, run_migrations
from .models import (
    APIKey,
    Base,
    BatchOperation,
    Benchmark,
    CacheEntry,
    Document,
    ErrorLog,
    Framework,
    PerformanceMetric,
    Query,
    User,
)
from .repositories import (
    APIKeyRepository,
    BaseRepository,
    BatchRepository,
    BenchmarkRepository,
    CacheRepository,
    DocumentRepository,
    ErrorLogRepository,
    PerformanceRepository,
    QueryRepository,
    UserRepository,
)

__all__ = [
    # Models
    'Base',
    'User',
    'APIKey',
    'Framework',
    'Document',
    'Query',
    'BatchOperation',
    'Benchmark',
    'ErrorLog',
    'PerformanceMetric',
    'CacheEntry',
    # Connection
    'DatabaseManager',
    'get_database_manager',
    'get_session',
    'get_async_session',
    'create_tables',
    'drop_tables',
    # Repositories
    'BaseRepository',
    'UserRepository',
    'APIKeyRepository',
    'DocumentRepository',
    'QueryRepository',
    'BatchRepository',
    'BenchmarkRepository',
    'ErrorLogRepository',
    'PerformanceRepository',
    'CacheRepository',
    # Migrations
    'MigrationManager',
    'create_migration',
    'run_migrations',
    'rollback_migration',
    'get_migration_status',
]
