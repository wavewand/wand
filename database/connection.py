"""
Database Connection Management

Provides connection pooling, session management, and database initialization
with support for both PostgreSQL and SQLite.
"""

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional
from urllib.parse import urlparse

from sqlalchemy import MetaData, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from config.settings import DatabaseSettings
from utils.error_handling import ErrorCategory, MCPError

from .models import Base


class DatabaseError(MCPError):
    """Database-specific error."""

    def __init__(self, message: str, operation: str = None):
        super().__init__(message, category=ErrorCategory.DATABASE_ERROR, details={"operation": operation})


class DatabaseManager:
    """Central database management with connection pooling and session handling."""

    def __init__(self, settings: DatabaseSettings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        # Sync engine and session factory
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None

        # Async engine and session factory
        self.async_engine = None
        self.async_session_factory = None

        # Connection status
        self.is_connected = False

        self._initialize_engines()

    def _initialize_engines(self):
        """Initialize both sync and async database engines."""
        try:
            # Parse database URL to determine database type
            parsed_url = urlparse(self.settings.url)
            db_type = parsed_url.scheme.split('+')[0]  # Handle postgres+asyncpg

            # Configure engine parameters based on database type
            engine_kwargs = {
                'echo': self.settings.echo,
                'future': True,  # Use SQLAlchemy 2.0 style
            }

            if db_type == 'sqlite':
                # SQLite-specific configuration
                engine_kwargs.update(
                    {'poolclass': StaticPool, 'connect_args': {'check_same_thread': False, 'timeout': 30}}
                )
            else:
                # PostgreSQL/other database configuration
                engine_kwargs.update(
                    {
                        'pool_size': self.settings.pool_size,
                        'max_overflow': self.settings.max_overflow,
                        'pool_timeout': self.settings.pool_timeout,
                        'pool_recycle': self.settings.pool_recycle,
                        'poolclass': QueuePool,
                    }
                )

            # Create synchronous engine
            self.engine = create_engine(self.settings.url, **engine_kwargs)

            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine, class_=Session, expire_on_commit=False)

            # Create async engine if URL supports it
            if db_type in ['postgresql', 'postgres']:
                # Convert to async URL
                async_url = self.settings.url.replace('postgresql://', 'postgresql+asyncpg://')
                async_engine_kwargs = engine_kwargs.copy()
                async_engine_kwargs.pop('connect_args', None)  # Remove SQLite-specific args

                self.async_engine = create_async_engine(async_url, **async_engine_kwargs)
                self.async_session_factory = async_sessionmaker(
                    bind=self.async_engine, class_=AsyncSession, expire_on_commit=False
                )

            # Add event listeners
            self._setup_event_listeners()

            self.logger.info(f"Database engines initialized for {db_type}")

        except Exception as e:
            self.logger.error(f"Failed to initialize database engines: {e}")
            raise DatabaseError(f"Database initialization failed: {e}", "initialize")

    def _setup_event_listeners(self):
        """Setup database event listeners for monitoring and optimization."""

        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance."""
            if 'sqlite' in str(self.engine.url):
                cursor = dbapi_connection.cursor()
                # Enable WAL mode for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL")
                # Set synchronous mode for better performance
                cursor.execute("PRAGMA synchronous=NORMAL")
                # Enable foreign key constraints
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        @event.listens_for(self.engine, "before_cursor_execute")
        def log_slow_queries(conn, cursor, statement, parameters, context, executemany):
            """Log slow queries for performance monitoring."""
            context._query_start_time = context._query_start_time = conn.info.setdefault('query_start_time', 0)
            import time

            conn.info['query_start_time'] = time.time()

        @event.listens_for(self.engine, "after_cursor_execute")
        def log_query_duration(conn, cursor, statement, parameters, context, executemany):
            """Log query execution time."""
            import time

            total = time.time() - conn.info['query_start_time']

            if total > 1.0:  # Log queries taking more than 1 second
                self.logger.warning(
                    f"Slow query detected: {total:.2f}s",
                    query=statement[:200] + "..." if len(statement) > 200 else statement,
                    duration_seconds=total,
                )

    def connect(self):
        """Test database connection."""
        try:
            from sqlalchemy import text

            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                self.is_connected = True
                self.logger.info("Database connection established")
        except Exception as e:
            self.is_connected = False
            self.logger.error(f"Database connection failed: {e}")
            raise DatabaseError(f"Connection failed: {e}", "connect")

    async def async_connect(self):
        """Test async database connection."""
        if not self.async_engine:
            raise DatabaseError("Async engine not available", "async_connect")

        try:
            from sqlalchemy import text

            async with self.get_async_session() as session:
                await session.execute(text("SELECT 1"))
                self.logger.info("Async database connection established")
        except Exception as e:
            self.logger.error(f"Async database connection failed: {e}")
            raise DatabaseError(f"Async connection failed: {e}", "async_connect")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        if not self.session_factory:
            raise DatabaseError("Database not initialized", "get_session")

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Session error: {e}", "session_operation")
        finally:
            session.close()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup."""
        if not self.async_session_factory:
            raise DatabaseError("Async database not initialized", "get_async_session")

        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            self.logger.error(f"Async database session error: {e}")
            raise DatabaseError(f"Async session error: {e}", "async_session_operation")
        finally:
            await session.close()

    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise DatabaseError(f"Table creation failed: {e}", "create_tables")

    def drop_tables(self):
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            self.logger.info("Database tables dropped successfully")
        except Exception as e:
            self.logger.error(f"Failed to drop tables: {e}")
            raise DatabaseError(f"Table drop failed: {e}", "drop_tables")

    def get_table_info(self) -> dict:
        """Get information about database tables."""
        try:
            from sqlalchemy import inspect

            with self.get_session() as session:
                inspector = inspect(session.connection())
                tables = {}

                for table_name in Base.metadata.tables.keys():
                    try:
                        columns = inspector.get_columns(table_name)
                        indexes = inspector.get_indexes(table_name)

                        tables[table_name] = {
                            'columns': [col['name'] for col in columns],
                            'indexes': [idx['name'] for idx in indexes],
                            'column_count': len(columns),
                            'index_count': len(indexes),
                        }
                    except Exception as e:
                        self.logger.warning(f"Could not inspect table {table_name}: {e}")

                return tables

        except Exception as e:
            self.logger.error(f"Failed to get table info: {e}")
            raise DatabaseError(f"Table inspection failed: {e}", "get_table_info")

    def get_connection_info(self) -> dict:
        """Get database connection information."""
        return {
            'url': str(self.engine.url).replace(self.engine.url.password or '', '***'),
            'dialect': self.engine.dialect.name,
            'driver': self.engine.dialect.driver,
            'pool_size': getattr(self.engine.pool, 'size', None),
            'checked_in': getattr(self.engine.pool, 'checkedin', None),
            'checked_out': getattr(self.engine.pool, 'checkedout', None),
            'is_connected': self.is_connected,
            'async_available': self.async_engine is not None,
        }

    def health_check(self) -> dict:
        """Perform database health check."""
        import time

        try:
            start_time = time.time()

            from sqlalchemy import text

            with self.get_session() as session:
                # Test basic query
                session.execute(text("SELECT 1"))

                # Get connection pool stats
                pool_stats = {
                    'size': getattr(self.engine.pool, 'size', 0),
                    'checked_in': getattr(self.engine.pool, 'checkedin', 0),
                    'checked_out': getattr(self.engine.pool, 'checkedout', 0),
                    'overflow': getattr(self.engine.pool, 'overflow', 0),
                }

            response_time = (time.time() - start_time) * 1000

            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'pool_stats': pool_stats,
                'connection_info': self.get_connection_info(),
            }

        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e), 'connection_info': self.get_connection_info()}

    def close(self):
        """Close database connections."""
        try:
            if self.engine:
                self.engine.dispose()
                self.logger.info("Sync database engine disposed")

            if self.async_engine:
                # Note: async engine disposal should be awaited in async context
                self.logger.info("Async database engine marked for disposal")

            self.is_connected = False

        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")

    async def async_close(self):
        """Close async database connections."""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
                self.logger.info("Async database engine disposed")
        except Exception as e:
            self.logger.error(f"Error closing async database connections: {e}")


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


def initialize_database(settings: DatabaseSettings) -> DatabaseManager:
    """Initialize global database manager."""
    global _database_manager
    _database_manager = DatabaseManager(settings)
    return _database_manager


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    if _database_manager is None:
        raise DatabaseError("Database manager not initialized", "get_manager")
    return _database_manager


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get database session from global manager."""
    manager = get_database_manager()
    with manager.get_session() as session:
        yield session


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session from global manager."""
    manager = get_database_manager()
    async with manager.get_async_session() as session:
        yield session


def create_tables():
    """Create all database tables using global manager."""
    manager = get_database_manager()
    manager.create_tables()


def drop_tables():
    """Drop all database tables using global manager."""
    manager = get_database_manager()
    manager.drop_tables()
