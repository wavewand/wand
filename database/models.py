"""
Database Models

SQLAlchemy ORM models for all platform entities with relationships,
indexes, and constraints for optimal performance.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

Base = declarative_base()


def generate_uuid():
    """Generate UUID for primary keys."""
    return str(uuid.uuid4())


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""

    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)


class User(Base, TimestampMixin):
    """User model for authentication and authorization."""

    __tablename__ = 'users'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)  # Nullable for external auth
    role = Column(String(20), nullable=False, default='user')
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    metadata_ = Column('metadata', JSON, default=dict)

    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    queries = relationship("Query", back_populates="user")
    batch_operations = relationship("BatchOperation", back_populates="user")

    # Constraints
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'user', 'service', 'readonly')", name='valid_role'),
        Index('idx_user_active_role', 'is_active', 'role'),
    )

    @validates('email')
    def validate_email(self, key, email):
        """Validate email format."""
        import re

        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            raise ValueError("Invalid email format")
        return email.lower()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'is_active': self.is_active,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata_,
        }


class APIKey(Base, TimestampMixin):
    """API Key model for authentication."""

    __tablename__ = 'api_keys'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    key_id = Column(String(50), unique=True, nullable=False, index=True)
    key_hash = Column(String(255), nullable=False)
    name = Column(String(100), nullable=False)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    permissions = Column(JSON, default=list)
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_used = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)
    rate_limit = Column(Integer, nullable=True)  # requests per minute

    # Relationships
    user = relationship("User", back_populates="api_keys")

    # Indexes
    __table_args__ = (
        Index('idx_apikey_active_user', 'is_active', 'user_id'),
        Index('idx_apikey_expires', 'expires_at'),
    )

    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'key_id': self.key_id,
            'name': self.name,
            'user_id': self.user_id,
            'permissions': self.permissions,
            'is_active': self.is_active,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'usage_count': self.usage_count,
            'rate_limit': self.rate_limit,
            'created_at': self.created_at.isoformat(),
            'is_expired': self.is_expired(),
        }


class Framework(Base, TimestampMixin):
    """AI Framework model for tracking framework configurations."""

    __tablename__ = 'frameworks'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(50), unique=True, nullable=False, index=True)
    display_name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=True)
    is_enabled = Column(Boolean, default=True, nullable=False)
    configuration = Column(JSON, default=dict)
    capabilities = Column(JSON, default=list)
    health_status = Column(String(20), default='unknown')
    last_health_check = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    documents = relationship("Document", back_populates="framework")
    queries = relationship("Query", back_populates="framework")
    performance_metrics = relationship("PerformanceMetric", back_populates="framework")

    # Constraints
    __table_args__ = (
        CheckConstraint("health_status IN ('healthy', 'unhealthy', 'unknown')", name='valid_health_status'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'version': self.version,
            'is_enabled': self.is_enabled,
            'configuration': self.configuration,
            'capabilities': self.capabilities,
            'health_status': self.health_status,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }


class Document(Base, TimestampMixin):
    """Document model for tracking ingested documents."""

    __tablename__ = 'documents'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(100), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    size_bytes = Column(Integer, nullable=False)
    framework_id = Column(String(36), ForeignKey('frameworks.id'), nullable=False)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True)
    metadata_ = Column('metadata', JSON, default=dict)
    processing_status = Column(String(20), default='pending')
    processing_error = Column(Text, nullable=True)

    # Relationships
    framework = relationship("Framework", back_populates="documents")
    user = relationship("User")

    # Indexes
    __table_args__ = (
        Index('idx_document_framework_status', 'framework_id', 'processing_status'),
        Index('idx_document_user_created', 'user_id', 'created_at'),
        UniqueConstraint('document_id', 'framework_id', name='uq_document_framework'),
        CheckConstraint(
            "processing_status IN ('pending', 'processing', 'completed', 'failed')", name='valid_processing_status'
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'filename': self.filename,
            'content_type': self.content_type,
            'content_hash': self.content_hash,
            'size_bytes': self.size_bytes,
            'framework_id': self.framework_id,
            'user_id': self.user_id,
            'metadata': self.metadata_,
            'processing_status': self.processing_status,
            'processing_error': self.processing_error,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }


class Query(Base, TimestampMixin):
    """Query model for tracking all queries across frameworks."""

    __tablename__ = 'queries'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    query_text = Column(Text, nullable=False)
    query_hash = Column(String(64), nullable=False, index=True)
    query_type = Column(String(50), nullable=False)  # rag, search, summarization
    framework_id = Column(String(36), ForeignKey('frameworks.id'), nullable=False)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True)

    # Request/Response data
    request_data = Column(JSON, default=dict)
    response_data = Column(JSON, default=dict)

    # Performance metrics
    execution_time_ms = Column(Float, nullable=False)
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)

    # Context
    correlation_id = Column(String(36), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)

    # Relationships
    framework = relationship("Framework", back_populates="queries")
    user = relationship("User", back_populates="queries")

    # Indexes
    __table_args__ = (
        Index('idx_query_framework_type', 'framework_id', 'query_type'),
        Index('idx_query_user_created', 'user_id', 'created_at'),
        Index('idx_query_success_time', 'success', 'execution_time_ms'),
        Index('idx_query_created_desc', 'created_at', postgresql_using='btree'),
        CheckConstraint("query_type IN ('rag', 'search', 'summarization', 'ingestion')", name='valid_query_type'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'query_text': self.query_text[:200] + '...' if len(self.query_text) > 200 else self.query_text,
            'query_hash': self.query_hash,
            'query_type': self.query_type,
            'framework_id': self.framework_id,
            'user_id': self.user_id,
            'execution_time_ms': self.execution_time_ms,
            'success': self.success,
            'error_message': self.error_message,
            'correlation_id': self.correlation_id,
            'created_at': self.created_at.isoformat(),
        }


class BatchOperation(Base, TimestampMixin):
    """Batch operation model for tracking bulk operations."""

    __tablename__ = 'batch_operations'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    batch_id = Column(String(50), unique=True, nullable=False, index=True)
    batch_type = Column(String(50), nullable=False)  # rag, search, ingestion
    framework_id = Column(String(36), ForeignKey('frameworks.id'), nullable=False)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True)

    # Progress tracking
    total_items = Column(Integer, nullable=False)
    processed_items = Column(Integer, default=0, nullable=False)
    failed_items = Column(Integer, default=0, nullable=False)
    status = Column(String(20), default='pending')

    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Configuration and results
    configuration = Column(JSON, default=dict)
    results = Column(JSON, default=dict)
    error_details = Column(JSON, default=list)

    # Relationships
    framework = relationship("Framework")
    user = relationship("User", back_populates="batch_operations")

    # Indexes
    __table_args__ = (
        Index('idx_batch_user_status', 'user_id', 'status'),
        Index('idx_batch_framework_type', 'framework_id', 'batch_type'),
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')", name='valid_batch_status'
        ),
        CheckConstraint("batch_type IN ('rag', 'search', 'ingestion', 'benchmark')", name='valid_batch_type'),
    )

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'batch_id': self.batch_id,
            'batch_type': self.batch_type,
            'framework_id': self.framework_id,
            'user_id': self.user_id,
            'total_items': self.total_items,
            'processed_items': self.processed_items,
            'failed_items': self.failed_items,
            'status': self.status,
            'progress_percent': self.progress_percent,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }


class Benchmark(Base, TimestampMixin):
    """Benchmark model for performance comparison results."""

    __tablename__ = 'benchmarks'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    benchmark_id = Column(String(50), unique=True, nullable=False, index=True)
    benchmark_type = Column(String(50), nullable=False)  # performance, throughput, resource
    frameworks = Column(JSON, nullable=False)  # List of framework names
    configuration = Column(JSON, default=dict)

    # Results
    results = Column(JSON, default=dict)
    winner = Column(String(50), nullable=True)

    # Timing
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Status
    status = Column(String(20), default='running')

    # Indexes
    __table_args__ = (
        Index('idx_benchmark_type_status', 'benchmark_type', 'status'),
        Index('idx_benchmark_completed', 'completed_at', postgresql_using='btree'),
        CheckConstraint("benchmark_type IN ('performance', 'throughput', 'resource')", name='valid_benchmark_type'),
        CheckConstraint("status IN ('running', 'completed', 'failed')", name='valid_benchmark_status'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'benchmark_id': self.benchmark_id,
            'benchmark_type': self.benchmark_type,
            'frameworks': self.frameworks,
            'configuration': self.configuration,
            'results': self.results,
            'winner': self.winner,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
        }


class ErrorLog(Base, TimestampMixin):
    """Error log model for tracking system errors."""

    __tablename__ = 'error_logs'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    error_id = Column(String(50), unique=True, nullable=False, index=True)
    category = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)

    # Context
    framework_name = Column(String(50), nullable=True)
    operation = Column(String(100), nullable=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True)
    correlation_id = Column(String(36), nullable=True, index=True)

    # Technical details
    stack_trace = Column(Text, nullable=True)
    details = Column(JSON, default=dict)
    retry_count = Column(Integer, default=0, nullable=False)

    # Resolution
    resolved = Column(Boolean, default=False, nullable=False)
    resolution_notes = Column(Text, nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User")

    # Indexes
    __table_args__ = (
        Index('idx_error_category_severity', 'category', 'severity'),
        Index('idx_error_framework_created', 'framework_name', 'created_at'),
        Index('idx_error_resolved', 'resolved', 'created_at'),
        CheckConstraint("severity IN ('low', 'medium', 'high', 'critical')", name='valid_error_severity'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'error_id': self.error_id,
            'category': self.category,
            'severity': self.severity,
            'message': self.message,
            'framework_name': self.framework_name,
            'operation': self.operation,
            'user_id': self.user_id,
            'correlation_id': self.correlation_id,
            'retry_count': self.retry_count,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'created_at': self.created_at.isoformat(),
        }


class PerformanceMetric(Base, TimestampMixin):
    """Performance metrics model for system monitoring."""

    __tablename__ = 'performance_metrics'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    framework_id = Column(String(36), ForeignKey('frameworks.id'), nullable=False)
    operation = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # response_time, throughput, error_rate

    # Metric values
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=False)  # ms, rps, percent

    # Aggregation info
    sample_count = Column(Integer, nullable=False, default=1)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)

    # Time bucket (for aggregated metrics)
    time_bucket = Column(DateTime(timezone=True), nullable=False, index=True)
    bucket_size_minutes = Column(Integer, nullable=False, default=1)

    # Additional context
    tags = Column(JSON, default=dict)

    # Relationships
    framework = relationship("Framework", back_populates="performance_metrics")

    # Indexes
    __table_args__ = (
        Index('idx_perf_framework_operation', 'framework_id', 'operation'),
        Index('idx_perf_metric_time', 'metric_type', 'time_bucket'),
        Index('idx_perf_bucket_desc', 'time_bucket', postgresql_using='btree'),
        CheckConstraint(
            "metric_type IN ('response_time', 'throughput', 'error_rate', 'success_rate')", name='valid_metric_type'
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'framework_id': self.framework_id,
            'operation': self.operation,
            'metric_type': self.metric_type,
            'value': self.value,
            'unit': self.unit,
            'sample_count': self.sample_count,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'time_bucket': self.time_bucket.isoformat(),
            'bucket_size_minutes': self.bucket_size_minutes,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
        }


class CacheEntry(Base, TimestampMixin):
    """Cache entry model for persistent caching."""

    __tablename__ = 'cache_entries'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    cache_key = Column(String(255), unique=True, nullable=False, index=True)
    key_hash = Column(String(64), nullable=False, index=True)

    # Cache metadata
    framework_name = Column(String(50), nullable=False)
    operation = Column(String(100), nullable=False)
    query_text = Column(Text, nullable=True)

    # Cache data
    response_data = Column(JSON, nullable=False)
    content_hash = Column(String(64), nullable=False)

    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    ttl_seconds = Column(Integer, nullable=False)

    # Usage statistics
    access_count = Column(Integer, default=0, nullable=False)
    last_accessed = Column(DateTime(timezone=True), nullable=True)

    # Metadata
    tags = Column(JSON, default=dict)
    size_bytes = Column(Integer, nullable=True)

    # Indexes
    __table_args__ = (
        Index('idx_cache_framework_operation', 'framework_name', 'operation'),
        Index('idx_cache_expires', 'expires_at'),
        Index('idx_cache_accessed', 'last_accessed', postgresql_using='btree'),
    )

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'cache_key': self.cache_key,
            'framework_name': self.framework_name,
            'operation': self.operation,
            'expires_at': self.expires_at.isoformat(),
            'ttl_seconds': self.ttl_seconds,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'size_bytes': self.size_bytes,
            'is_expired': self.is_expired(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }
