"""
Response Caching System

Provides intelligent caching for AI framework responses to improve performance
and reduce redundant processing for similar queries.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CacheStrategy(str, Enum):
    """Caching strategies for different types of operations."""

    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_MATCH = "semantic_match"
    NO_CACHE = "no_cache"


class CacheStatus(str, Enum):
    """Status of cache operations."""

    HIT = "hit"
    MISS = "miss"
    EXPIRED = "expired"
    INVALID = "invalid"


@dataclass
class CacheEntry:
    """Represents a cached response."""

    key: str
    query_hash: str
    framework: str
    operation: str
    query: str
    response: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int = 3600  # 1 hour default

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return datetime.now() > (self.created_at + timedelta(seconds=self.ttl_seconds))

    def access(self):
        """Mark the entry as accessed."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "query_hash": self.query_hash,
            "framework": self.framework,
            "operation": self.operation,
            "query": self.query,
            "response": self.response,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "ttl_seconds": self.ttl_seconds,
            "expired": self.is_expired(),
        }


class ResponseCache:
    """Intelligent response caching system for AI frameworks."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Cache strategies per operation type
        self.operation_strategies = {
            "rag_query": CacheStrategy.FUZZY_MATCH,
            "document_search": CacheStrategy.FUZZY_MATCH,
            "text_summarization": CacheStrategy.EXACT_MATCH,
            "document_ingestion": CacheStrategy.NO_CACHE,  # Don't cache ingestion
        }

        # TTL per operation type (in seconds)
        self.operation_ttls = {
            "rag_query": 3600,  # 1 hour
            "document_search": 1800,  # 30 minutes
            "text_summarization": 7200,  # 2 hours
        }

        self.logger.info(f"Response cache initialized with max_size={max_size}, default_ttl={default_ttl}s")

    def _generate_cache_key(self, framework: str, operation: str, query: str, **kwargs) -> str:
        """Generate a cache key for the given parameters."""
        # Create a deterministic key from all parameters
        key_data = {
            "framework": framework,
            "operation": operation,
            "query": query.lower().strip(),  # Normalize query
            **{k: v for k, v in kwargs.items() if k not in ['timestamp', 'request_id']},  # Exclude non-cacheable params
        }

        # Create hash from sorted key data
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _generate_query_hash(self, query: str) -> str:
        """Generate a hash for the query text for fuzzy matching."""
        # Normalize query for better matching
        normalized = query.lower().strip()
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _should_cache(self, operation: str) -> bool:
        """Check if an operation should be cached."""
        strategy = self.operation_strategies.get(operation, CacheStrategy.EXACT_MATCH)
        return strategy != CacheStrategy.NO_CACHE

    def _find_fuzzy_match(self, query_hash: str, framework: str, operation: str) -> Optional[CacheEntry]:
        """Find a fuzzy match for similar queries."""
        # For fuzzy matching, we look for entries with the same query hash
        # but allow for slight variations in other parameters

        candidates = []
        for entry in self.cache.values():
            if (
                entry.query_hash == query_hash
                and entry.framework == framework
                and entry.operation == operation
                and not entry.is_expired()
            ):
                candidates.append(entry)

        if candidates:
            # Return the most recently accessed entry
            return max(candidates, key=lambda e: e.last_accessed)

        return None

    async def get(
        self, framework: str, operation: str, query: str, **kwargs
    ) -> Tuple[Optional[Dict[str, Any]], CacheStatus]:
        """Get a cached response if available."""
        if not self._should_cache(operation):
            return None, CacheStatus.INVALID

        cache_key = self._generate_cache_key(framework, operation, query, **kwargs)
        query_hash = self._generate_query_hash(query)

        # Try exact match first
        if cache_key in self.cache:
            entry = self.cache[cache_key]

            if entry.is_expired():
                # Remove expired entry
                del self.cache[cache_key]
                self.logger.debug(f"Cache entry expired: {cache_key}")
                return None, CacheStatus.EXPIRED

            # Cache hit
            entry.access()
            self.hits += 1
            self.logger.debug(f"Cache hit: {cache_key} (accessed {entry.access_count} times)")
            return entry.response, CacheStatus.HIT

        # Try fuzzy match
        strategy = self.operation_strategies.get(operation, CacheStrategy.EXACT_MATCH)
        if strategy == CacheStrategy.FUZZY_MATCH:
            fuzzy_entry = self._find_fuzzy_match(query_hash, framework, operation)
            if fuzzy_entry:
                fuzzy_entry.access()
                self.hits += 1
                self.logger.debug(f"Cache fuzzy hit: {fuzzy_entry.key} for query hash {query_hash}")
                return fuzzy_entry.response, CacheStatus.HIT

        # Cache miss
        self.misses += 1
        self.logger.debug(f"Cache miss: {cache_key}")
        return None, CacheStatus.MISS

    async def set(self, framework: str, operation: str, query: str, response: Dict[str, Any], **kwargs):
        """Store a response in the cache."""
        if not self._should_cache(operation):
            return

        cache_key = self._generate_cache_key(framework, operation, query, **kwargs)
        query_hash = self._generate_query_hash(query)

        # Get TTL for this operation
        ttl = self.operation_ttls.get(operation, self.default_ttl)

        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            query_hash=query_hash,
            framework=framework,
            operation=operation,
            query=query,
            response=response,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=ttl,
        )

        # Check if we need to evict entries
        if len(self.cache) >= self.max_size:
            await self._evict_entries()

        # Store entry
        self.cache[cache_key] = entry
        self.logger.debug(f"Cached response: {cache_key} (TTL: {ttl}s)")

    async def _evict_entries(self):
        """Evict least recently used entries to make space."""
        if not self.cache:
            return

        # Remove expired entries first
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self.cache[key]

        # If still at max size, remove LRU entries
        while len(self.cache) >= self.max_size:
            # Find least recently used entry
            lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
            del self.cache[lru_key]
            self.evictions += 1

        if expired_keys or self.evictions > 0:
            self.logger.debug(f"Evicted {len(expired_keys)} expired and {self.evictions} LRU entries")

    def invalidate(self, framework: str = None, operation: str = None, pattern: str = None):
        """Invalidate cache entries matching the criteria."""
        keys_to_remove = []

        for key, entry in self.cache.items():
            should_remove = True

            if framework and entry.framework != framework:
                should_remove = False

            if operation and entry.operation != operation:
                should_remove = False

            if pattern and pattern.lower() not in entry.query.lower():
                should_remove = False

            if should_remove:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]

        self.logger.info(f"Invalidated {len(keys_to_remove)} cache entries")
        return len(keys_to_remove)

    def clear(self):
        """Clear all cache entries."""
        count = len(self.cache)
        self.cache.clear()
        self.logger.info(f"Cleared {count} cache entries")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        # Calculate cache sizes by framework and operation
        framework_counts = {}
        operation_counts = {}

        for entry in self.cache.values():
            framework_counts[entry.framework] = framework_counts.get(entry.framework, 0) + 1
            operation_counts[entry.operation] = operation_counts.get(entry.operation, 0) + 1

        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "utilization_percent": (len(self.cache) / self.max_size * 100),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "framework_distribution": framework_counts,
            "operation_distribution": operation_counts,
            "strategies": {op: strategy.value for op, strategy in self.operation_strategies.items()},
            "ttls": self.operation_ttls,
        }

    def get_entries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get cache entries for inspection."""
        entries = sorted(self.cache.values(), key=lambda e: e.last_accessed, reverse=True)
        return [entry.to_dict() for entry in entries[:limit]]

    async def cleanup_expired(self):
        """Remove all expired entries."""
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def save_to_disk(self, filepath: str):
        """Save cache to disk."""
        try:
            cache_data = {
                "entries": [entry.to_dict() for entry in self.cache.values()],
                "stats": {"hits": self.hits, "misses": self.misses, "evictions": self.evictions},
                "saved_at": datetime.now().isoformat(),
            }

            with open(filepath, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)

            self.logger.info(f"Saved cache to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save cache to disk: {e}")
            return False

    def load_from_disk(self, filepath: str):
        """Load cache from disk."""
        try:
            if not os.path.exists(filepath):
                self.logger.info("No cache file found, starting with empty cache")
                return False

            with open(filepath, 'r') as f:
                cache_data = json.load(f)

            # Restore entries (only non-expired ones)
            loaded_count = 0
            for entry_data in cache_data.get("entries", []):
                created_at = datetime.fromisoformat(entry_data["created_at"])
                ttl = entry_data["ttl_seconds"]

                # Skip expired entries
                if datetime.now() > (created_at + timedelta(seconds=ttl)):
                    continue

                entry = CacheEntry(
                    key=entry_data["key"],
                    query_hash=entry_data["query_hash"],
                    framework=entry_data["framework"],
                    operation=entry_data["operation"],
                    query=entry_data["query"],
                    response=entry_data["response"],
                    created_at=created_at,
                    last_accessed=datetime.fromisoformat(entry_data["last_accessed"]),
                    access_count=entry_data["access_count"],
                    ttl_seconds=ttl,
                )

                self.cache[entry.key] = entry
                loaded_count += 1

            # Restore stats
            stats = cache_data.get("stats", {})
            self.hits = stats.get("hits", 0)
            self.misses = stats.get("misses", 0)
            self.evictions = stats.get("evictions", 0)

            self.logger.info(f"Loaded {loaded_count} cache entries from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load cache from disk: {e}")
            return False


# Global cache instance
response_cache = ResponseCache()


class CacheMiddleware:
    """Middleware for automatic response caching."""

    def __init__(self, cache: ResponseCache = None):
        self.cache = cache or response_cache

    async def __call__(self, framework: str, operation: str, query: str, func, *args, **kwargs):
        """Execute function with caching."""
        # Try to get from cache
        cached_response, status = await self.cache.get(framework, operation, query, **kwargs)

        if status == CacheStatus.HIT:
            return cached_response

        # Execute function
        try:
            result = await func(*args, **kwargs)

            # Cache successful results
            if isinstance(result, dict) and result.get("success", True):
                await self.cache.set(framework, operation, query, result, **kwargs)

            return result

        except Exception as e:
            # Don't cache errors
            raise


# Decorator for easy caching
def cache_response(framework: str, operation: str):
    """Decorator to automatically cache framework responses."""

    def decorator(func):
        async def wrapper(query: str, *args, **kwargs):
            middleware = CacheMiddleware()
            return await middleware(framework, operation, query, func, query, *args, **kwargs)

        return wrapper

    return decorator
