"""
Database Repositories

Data access layer with repository pattern for clean separation of concerns
and consistent database operations across all entities.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sqlalchemy import and_, asc, delete, desc, func, or_, select, update
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from utils.error_handling import ErrorCategory, MCPError

from .connection import get_async_session, get_session
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

T = TypeVar('T', bound=Base)


class RepositoryError(MCPError):
    """Repository-specific error."""

    def __init__(self, message: str, operation: str = None, entity: str = None):
        super().__init__(
            message, category=ErrorCategory.DATABASE_ERROR, details={"operation": operation, "entity": entity}
        )


class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations."""

    def __init__(self, model_class: Type[T]):
        self.model_class = model_class
        self.logger = logging.getLogger(f"{__name__}.{model_class.__name__}")

    def create(self, session: Session, **kwargs) -> T:
        """Create new entity."""
        try:
            entity = self.model_class(**kwargs)
            session.add(entity)
            session.flush()  # Get the ID without committing
            session.refresh(entity)
            self.logger.debug(f"Created {self.model_class.__name__} with ID: {entity.id}")
            return entity
        except IntegrityError as e:
            session.rollback()
            raise RepositoryError(
                f"Failed to create {self.model_class.__name__}: integrity constraint violation",
                "create",
                self.model_class.__name__,
            ) from e
        except Exception as e:
            session.rollback()
            raise RepositoryError(
                f"Failed to create {self.model_class.__name__}: {e}", "create", self.model_class.__name__
            ) from e

    def get_by_id(self, session: Session, entity_id: str) -> Optional[T]:
        """Get entity by ID."""
        try:
            return session.get(self.model_class, entity_id)
        except Exception as e:
            raise RepositoryError(
                f"Failed to get {self.model_class.__name__} by ID: {e}", "get_by_id", self.model_class.__name__
            ) from e

    def get_all(self, session: Session, limit: int = 100, offset: int = 0) -> List[T]:
        """Get all entities with pagination."""
        try:
            query = session.query(self.model_class).offset(offset).limit(limit)
            return query.all()
        except Exception as e:
            raise RepositoryError(
                f"Failed to get all {self.model_class.__name__}: {e}", "get_all", self.model_class.__name__
            ) from e

    def update(self, session: Session, entity_id: str, **kwargs) -> Optional[T]:
        """Update entity by ID."""
        try:
            entity = session.get(self.model_class, entity_id)
            if not entity:
                return None

            for key, value in kwargs.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)

            # Update timestamp if available
            if hasattr(entity, 'updated_at'):
                entity.updated_at = datetime.now(timezone.utc)

            session.flush()
            session.refresh(entity)
            self.logger.debug(f"Updated {self.model_class.__name__} with ID: {entity_id}")
            return entity
        except Exception as e:
            session.rollback()
            raise RepositoryError(
                f"Failed to update {self.model_class.__name__}: {e}", "update", self.model_class.__name__
            ) from e

    def delete(self, session: Session, entity_id: str) -> bool:
        """Delete entity by ID."""
        try:
            entity = session.get(self.model_class, entity_id)
            if not entity:
                return False

            session.delete(entity)
            session.flush()
            self.logger.debug(f"Deleted {self.model_class.__name__} with ID: {entity_id}")
            return True
        except Exception as e:
            session.rollback()
            raise RepositoryError(
                f"Failed to delete {self.model_class.__name__}: {e}", "delete", self.model_class.__name__
            ) from e

    def count(self, session: Session, **filters) -> int:
        """Count entities with optional filters."""
        try:
            query = session.query(self.model_class)

            # Apply filters
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    query = query.filter(getattr(self.model_class, key) == value)

            return query.count()
        except Exception as e:
            raise RepositoryError(
                f"Failed to count {self.model_class.__name__}: {e}", "count", self.model_class.__name__
            ) from e


class UserRepository(BaseRepository[User]):
    """Repository for User operations."""

    def __init__(self):
        super().__init__(User)

    def get_by_username(self, session: Session, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            return session.query(User).filter(User.username == username).first()
        except Exception as e:
            raise RepositoryError(f"Failed to get user by username: {e}", "get_by_username", "User")

    def get_by_email(self, session: Session, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            return session.query(User).filter(User.email == email.lower()).first()
        except Exception as e:
            raise RepositoryError(f"Failed to get user by email: {e}", "get_by_email", "User")

    def get_active_users(self, session: Session, limit: int = 100) -> List[User]:
        """Get all active users."""
        try:
            return session.query(User).filter(User.is_active).limit(limit).all()
        except Exception as e:
            raise RepositoryError(f"Failed to get active users: {e}", "get_active_users", "User")

    def update_last_login(self, session: Session, user_id: str) -> bool:
        """Update user's last login timestamp."""
        try:
            user = session.get(User, user_id)
            if user:
                user.last_login = datetime.now(timezone.utc)
                session.flush()
                return True
            return False
        except Exception as e:
            raise RepositoryError(f"Failed to update last login: {e}", "update_last_login", "User")


class APIKeyRepository(BaseRepository[APIKey]):
    """Repository for API Key operations."""

    def __init__(self):
        super().__init__(APIKey)

    def get_by_key_id(self, session: Session, key_id: str) -> Optional[APIKey]:
        """Get API key by key_id."""
        try:
            return session.query(APIKey).filter(APIKey.key_id == key_id).first()
        except Exception as e:
            raise RepositoryError(f"Failed to get API key by key_id: {e}", "get_by_key_id", "APIKey")

    def get_by_hash(self, session: Session, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash."""
        try:
            return session.query(APIKey).filter(APIKey.key_hash == key_hash).first()
        except Exception as e:
            raise RepositoryError(f"Failed to get API key by hash: {e}", "get_by_hash", "APIKey")

    def get_user_keys(self, session: Session, user_id: str, active_only: bool = True) -> List[APIKey]:
        """Get all API keys for a user."""
        try:
            query = session.query(APIKey).filter(APIKey.user_id == user_id)
            if active_only:
                query = query.filter(APIKey.is_active)
            return query.all()
        except Exception as e:
            raise RepositoryError(f"Failed to get user keys: {e}", "get_user_keys", "APIKey")

    def update_usage(self, session: Session, key_id: str) -> bool:
        """Update API key usage statistics."""
        try:
            api_key = session.query(APIKey).filter(APIKey.key_id == key_id).first()
            if api_key:
                api_key.usage_count += 1
                api_key.last_used = datetime.now(timezone.utc)
                session.flush()
                return True
            return False
        except Exception as e:
            raise RepositoryError(f"Failed to update API key usage: {e}", "update_usage", "APIKey")

    def cleanup_expired_keys(self, session: Session) -> int:
        """Remove expired API keys."""
        try:
            now = datetime.now(timezone.utc)
            result = session.query(APIKey).filter(and_(APIKey.expires_at.isnot(None), APIKey.expires_at < now)).delete()
            session.flush()
            return result
        except Exception as e:
            raise RepositoryError(f"Failed to cleanup expired keys: {e}", "cleanup_expired", "APIKey")


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document operations."""

    def __init__(self):
        super().__init__(Document)

    def get_by_document_id(self, session: Session, document_id: str, framework_id: str = None) -> Optional[Document]:
        """Get document by document_id and optionally framework."""
        try:
            query = session.query(Document).filter(Document.document_id == document_id)
            if framework_id:
                query = query.filter(Document.framework_id == framework_id)
            return query.first()
        except Exception as e:
            raise RepositoryError(f"Failed to get document by ID: {e}", "get_by_document_id", "Document")

    def get_by_framework(self, session: Session, framework_id: str, limit: int = 100) -> List[Document]:
        """Get documents by framework."""
        try:
            return session.query(Document).filter(Document.framework_id == framework_id).limit(limit).all()
        except Exception as e:
            raise RepositoryError(f"Failed to get documents by framework: {e}", "get_by_framework", "Document")

    def get_by_user(self, session: Session, user_id: str, limit: int = 100) -> List[Document]:
        """Get documents by user."""
        try:
            return session.query(Document).filter(Document.user_id == user_id).limit(limit).all()
        except Exception as e:
            raise RepositoryError(f"Failed to get documents by user: {e}", "get_by_user", "Document")

    def get_by_status(self, session: Session, status: str, limit: int = 100) -> List[Document]:
        """Get documents by processing status."""
        try:
            return session.query(Document).filter(Document.processing_status == status).limit(limit).all()
        except Exception as e:
            raise RepositoryError(f"Failed to get documents by status: {e}", "get_by_status", "Document")

    def update_processing_status(self, session: Session, document_id: str, status: str, error: str = None) -> bool:
        """Update document processing status."""
        try:
            doc = session.query(Document).filter(Document.document_id == document_id).first()
            if doc:
                doc.processing_status = status
                if error:
                    doc.processing_error = error
                session.flush()
                return True
            return False
        except Exception as e:
            raise RepositoryError(f"Failed to update processing status: {e}", "update_status", "Document")


class QueryRepository(BaseRepository[Query]):
    """Repository for Query operations."""

    def __init__(self):
        super().__init__(Query)

    def get_by_correlation_id(self, session: Session, correlation_id: str) -> List[Query]:
        """Get queries by correlation ID."""
        try:
            return session.query(Query).filter(Query.correlation_id == correlation_id).all()
        except Exception as e:
            raise RepositoryError(f"Failed to get queries by correlation ID: {e}", "get_by_correlation_id", "Query")

    def get_by_framework(self, session: Session, framework_id: str, limit: int = 100) -> List[Query]:
        """Get queries by framework."""
        try:
            return (
                session.query(Query)
                .filter(Query.framework_id == framework_id)
                .order_by(desc(Query.created_at))
                .limit(limit)
                .all()
            )
        except Exception as e:
            raise RepositoryError(f"Failed to get queries by framework: {e}", "get_by_framework", "Query")

    def get_by_user(self, session: Session, user_id: str, limit: int = 100) -> List[Query]:
        """Get queries by user."""
        try:
            return (
                session.query(Query)
                .filter(Query.user_id == user_id)
                .order_by(desc(Query.created_at))
                .limit(limit)
                .all()
            )
        except Exception as e:
            raise RepositoryError(f"Failed to get queries by user: {e}", "get_by_user", "Query")

    def get_failed_queries(self, session: Session, since: datetime = None, limit: int = 100) -> List[Query]:
        """Get failed queries."""
        try:
            query = session.query(Query).filter(Query.success == False)
            if since:
                query = query.filter(Query.created_at >= since)
            return query.order_by(desc(Query.created_at)).limit(limit).all()
        except Exception as e:
            raise RepositoryError(f"Failed to get failed queries: {e}", "get_failed_queries", "Query")

    def get_performance_stats(self, session: Session, framework_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get query performance statistics."""
        try:
            since = datetime.now(timezone.utc) - timedelta(hours=hours)
            query = session.query(Query).filter(Query.created_at >= since)

            if framework_id:
                query = query.filter(Query.framework_id == framework_id)

            queries = query.all()

            if not queries:
                return {
                    'total_queries': 0,
                    'success_rate': 0,
                    'avg_response_time_ms': 0,
                    'min_response_time_ms': 0,
                    'max_response_time_ms': 0,
                }

            successful_queries = [q for q in queries if q.success]
            response_times = [q.execution_time_ms for q in queries]

            return {
                'total_queries': len(queries),
                'successful_queries': len(successful_queries),
                'failed_queries': len(queries) - len(successful_queries),
                'success_rate': len(successful_queries) / len(queries) * 100,
                'avg_response_time_ms': sum(response_times) / len(response_times),
                'min_response_time_ms': min(response_times),
                'max_response_time_ms': max(response_times),
            }
        except Exception as e:
            raise RepositoryError(f"Failed to get performance stats: {e}", "get_performance_stats", "Query")


class BatchRepository(BaseRepository[BatchOperation]):
    """Repository for Batch Operation operations."""

    def __init__(self):
        super().__init__(BatchOperation)

    def get_by_batch_id(self, session: Session, batch_id: str) -> Optional[BatchOperation]:
        """Get batch operation by batch_id."""
        try:
            return session.query(BatchOperation).filter(BatchOperation.batch_id == batch_id).first()
        except Exception as e:
            raise RepositoryError(f"Failed to get batch by ID: {e}", "get_by_batch_id", "BatchOperation")

    def get_active_batches(self, session: Session) -> List[BatchOperation]:
        """Get all active (running) batch operations."""
        try:
            return session.query(BatchOperation).filter(BatchOperation.status.in_(['pending', 'running'])).all()
        except Exception as e:
            raise RepositoryError(f"Failed to get active batches: {e}", "get_active_batches", "BatchOperation")

    def get_by_user(self, session: Session, user_id: str, limit: int = 50) -> List[BatchOperation]:
        """Get batch operations by user."""
        try:
            return (
                session.query(BatchOperation)
                .filter(BatchOperation.user_id == user_id)
                .order_by(desc(BatchOperation.created_at))
                .limit(limit)
                .all()
            )
        except Exception as e:
            raise RepositoryError(f"Failed to get batches by user: {e}", "get_by_user", "BatchOperation")

    def update_progress(self, session: Session, batch_id: str, processed: int, failed: int = 0) -> bool:
        """Update batch progress."""
        try:
            batch = session.query(BatchOperation).filter(BatchOperation.batch_id == batch_id).first()
            if batch:
                batch.processed_items = processed
                batch.failed_items = failed
                session.flush()
                return True
            return False
        except Exception as e:
            raise RepositoryError(f"Failed to update batch progress: {e}", "update_progress", "BatchOperation")

    def complete_batch(self, session: Session, batch_id: str, results: Dict[str, Any]) -> bool:
        """Mark batch as completed."""
        try:
            batch = session.query(BatchOperation).filter(BatchOperation.batch_id == batch_id).first()
            if batch:
                batch.status = 'completed'
                batch.completed_at = datetime.now(timezone.utc)
                batch.results = results
                session.flush()
                return True
            return False
        except Exception as e:
            raise RepositoryError(f"Failed to complete batch: {e}", "complete_batch", "BatchOperation")


class BenchmarkRepository(BaseRepository[Benchmark]):
    """Repository for Benchmark operations."""

    def __init__(self):
        super().__init__(Benchmark)

    def get_by_benchmark_id(self, session: Session, benchmark_id: str) -> Optional[Benchmark]:
        """Get benchmark by benchmark_id."""
        try:
            return session.query(Benchmark).filter(Benchmark.benchmark_id == benchmark_id).first()
        except Exception as e:
            raise RepositoryError(f"Failed to get benchmark by ID: {e}", "get_by_benchmark_id", "Benchmark")

    def get_by_type(self, session: Session, benchmark_type: str, limit: int = 50) -> List[Benchmark]:
        """Get benchmarks by type."""
        try:
            return (
                session.query(Benchmark)
                .filter(Benchmark.benchmark_type == benchmark_type)
                .order_by(desc(Benchmark.created_at))
                .limit(limit)
                .all()
            )
        except Exception as e:
            raise RepositoryError(f"Failed to get benchmarks by type: {e}", "get_by_type", "Benchmark")

    def get_completed_benchmarks(self, session: Session, limit: int = 50) -> List[Benchmark]:
        """Get completed benchmarks."""
        try:
            return (
                session.query(Benchmark)
                .filter(Benchmark.status == 'completed')
                .order_by(desc(Benchmark.completed_at))
                .limit(limit)
                .all()
            )
        except Exception as e:
            raise RepositoryError(f"Failed to get completed benchmarks: {e}", "get_completed", "Benchmark")


class ErrorLogRepository(BaseRepository[ErrorLog]):
    """Repository for Error Log operations."""

    def __init__(self):
        super().__init__(ErrorLog)

    def get_by_error_id(self, session: Session, error_id: str) -> Optional[ErrorLog]:
        """Get error by error_id."""
        try:
            return session.query(ErrorLog).filter(ErrorLog.error_id == error_id).first()
        except Exception as e:
            raise RepositoryError(f"Failed to get error by ID: {e}", "get_by_error_id", "ErrorLog")

    def get_by_correlation_id(self, session: Session, correlation_id: str) -> List[ErrorLog]:
        """Get errors by correlation ID."""
        try:
            return session.query(ErrorLog).filter(ErrorLog.correlation_id == correlation_id).all()
        except Exception as e:
            raise RepositoryError(f"Failed to get errors by correlation ID: {e}", "get_by_correlation_id", "ErrorLog")

    def get_recent_errors(self, session: Session, hours: int = 24, limit: int = 100) -> List[ErrorLog]:
        """Get recent errors."""
        try:
            since = datetime.now(timezone.utc) - timedelta(hours=hours)
            return (
                session.query(ErrorLog)
                .filter(ErrorLog.created_at >= since)
                .order_by(desc(ErrorLog.created_at))
                .limit(limit)
                .all()
            )
        except Exception as e:
            raise RepositoryError(f"Failed to get recent errors: {e}", "get_recent_errors", "ErrorLog")

    def get_by_severity(self, session: Session, severity: str, limit: int = 100) -> List[ErrorLog]:
        """Get errors by severity."""
        try:
            return (
                session.query(ErrorLog)
                .filter(ErrorLog.severity == severity)
                .order_by(desc(ErrorLog.created_at))
                .limit(limit)
                .all()
            )
        except Exception as e:
            raise RepositoryError(f"Failed to get errors by severity: {e}", "get_by_severity", "ErrorLog")

    def get_unresolved_errors(self, session: Session, limit: int = 100) -> List[ErrorLog]:
        """Get unresolved errors."""
        try:
            return (
                session.query(ErrorLog)
                .filter(ErrorLog.resolved == False)
                .order_by(desc(ErrorLog.created_at))
                .limit(limit)
                .all()
            )
        except Exception as e:
            raise RepositoryError(f"Failed to get unresolved errors: {e}", "get_unresolved", "ErrorLog")

    def resolve_error(self, session: Session, error_id: str, resolution_notes: str) -> bool:
        """Mark error as resolved."""
        try:
            error = session.query(ErrorLog).filter(ErrorLog.error_id == error_id).first()
            if error:
                error.resolved = True
                error.resolved_at = datetime.now(timezone.utc)
                error.resolution_notes = resolution_notes
                session.flush()
                return True
            return False
        except Exception as e:
            raise RepositoryError(f"Failed to resolve error: {e}", "resolve_error", "ErrorLog")


class PerformanceRepository(BaseRepository[PerformanceMetric]):
    """Repository for Performance Metric operations."""

    def __init__(self):
        super().__init__(PerformanceMetric)

    def get_framework_metrics(self, session: Session, framework_id: str, hours: int = 24) -> List[PerformanceMetric]:
        """Get performance metrics for a framework."""
        try:
            since = datetime.now(timezone.utc) - timedelta(hours=hours)
            return (
                session.query(PerformanceMetric)
                .filter(and_(PerformanceMetric.framework_id == framework_id, PerformanceMetric.time_bucket >= since))
                .order_by(PerformanceMetric.time_bucket)
                .all()
            )
        except Exception as e:
            raise RepositoryError(f"Failed to get framework metrics: {e}", "get_framework_metrics", "PerformanceMetric")

    def get_operation_metrics(self, session: Session, operation: str, hours: int = 24) -> List[PerformanceMetric]:
        """Get performance metrics for an operation."""
        try:
            since = datetime.now(timezone.utc) - timedelta(hours=hours)
            return (
                session.query(PerformanceMetric)
                .filter(and_(PerformanceMetric.operation == operation, PerformanceMetric.time_bucket >= since))
                .order_by(PerformanceMetric.time_bucket)
                .all()
            )
        except Exception as e:
            raise RepositoryError(f"Failed to get operation metrics: {e}", "get_operation_metrics", "PerformanceMetric")

    def cleanup_old_metrics(self, session: Session, days: int = 30) -> int:
        """Remove old performance metrics."""
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            result = session.query(PerformanceMetric).filter(PerformanceMetric.time_bucket < cutoff).delete()
            session.flush()
            return result
        except Exception as e:
            raise RepositoryError(f"Failed to cleanup old metrics: {e}", "cleanup_old_metrics", "PerformanceMetric")


class CacheRepository(BaseRepository[CacheEntry]):
    """Repository for Cache Entry operations."""

    def __init__(self):
        super().__init__(CacheEntry)

    def get_by_key(self, session: Session, cache_key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        try:
            return session.query(CacheEntry).filter(CacheEntry.cache_key == cache_key).first()
        except Exception as e:
            raise RepositoryError(f"Failed to get cache entry by key: {e}", "get_by_key", "CacheEntry")

    def get_by_hash(self, session: Session, key_hash: str) -> Optional[CacheEntry]:
        """Get cache entry by hash."""
        try:
            return session.query(CacheEntry).filter(CacheEntry.key_hash == key_hash).first()
        except Exception as e:
            raise RepositoryError(f"Failed to get cache entry by hash: {e}", "get_by_hash", "CacheEntry")

    def get_framework_entries(self, session: Session, framework_name: str, limit: int = 100) -> List[CacheEntry]:
        """Get cache entries for a framework."""
        try:
            return (
                session.query(CacheEntry)
                .filter(CacheEntry.framework_name == framework_name)
                .order_by(desc(CacheEntry.last_accessed))
                .limit(limit)
                .all()
            )
        except Exception as e:
            raise RepositoryError(f"Failed to get framework cache entries: {e}", "get_framework_entries", "CacheEntry")

    def update_access(self, session: Session, cache_key: str) -> bool:
        """Update cache entry access statistics."""
        try:
            entry = session.query(CacheEntry).filter(CacheEntry.cache_key == cache_key).first()
            if entry:
                entry.access_count += 1
                entry.last_accessed = datetime.now(timezone.utc)
                session.flush()
                return True
            return False
        except Exception as e:
            raise RepositoryError(f"Failed to update cache access: {e}", "update_access", "CacheEntry")

    def cleanup_expired_entries(self, session: Session) -> int:
        """Remove expired cache entries."""
        try:
            now = datetime.now(timezone.utc)
            result = session.query(CacheEntry).filter(CacheEntry.expires_at < now).delete()
            session.flush()
            return result
        except Exception as e:
            raise RepositoryError(f"Failed to cleanup expired cache entries: {e}", "cleanup_expired", "CacheEntry")

    def get_cache_stats(self, session: Session) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            now = datetime.now(timezone.utc)

            total_entries = session.query(CacheEntry).count()
            expired_entries = session.query(CacheEntry).filter(CacheEntry.expires_at < now).count()
            active_entries = total_entries - expired_entries

            # Get framework distribution
            framework_stats = (
                session.query(CacheEntry.framework_name, func.count(CacheEntry.id).label('count'))
                .group_by(CacheEntry.framework_name)
                .all()
            )

            # Get operation distribution
            operation_stats = (
                session.query(CacheEntry.operation, func.count(CacheEntry.id).label('count'))
                .group_by(CacheEntry.operation)
                .all()
            )

            return {
                'total_entries': total_entries,
                'active_entries': active_entries,
                'expired_entries': expired_entries,
                'hit_rate_estimate': 0.85,  # Would need to calculate from access patterns
                'framework_distribution': {name: count for name, count in framework_stats},
                'operation_distribution': {op: count for op, count in operation_stats},
            }

        except Exception as e:
            raise RepositoryError(f"Failed to get cache stats: {e}", "get_cache_stats", "CacheEntry")
