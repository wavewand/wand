"""
Caching manager for Wand integrations
"""

import asyncio
import hashlib
import json
import os
import pickle
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


class CacheManager:
    """
    Multi-level cache manager with:
    - Memory cache (fastest)
    - Disk cache (persistent)
    - TTL-based expiration
    - LRU eviction
    - Size limits
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)

        # Cache configuration
        self.memory_cache_size = config.get("memory_cache_size", 1000)  # Max items
        self.disk_cache_size = config.get("disk_cache_size", 10000)  # Max items
        self.default_ttl = config.get("default_ttl", 3600)  # 1 hour
        self.disk_cache_dir = config.get("disk_cache_dir", ".cache")

        # Memory cache storage
        self.memory_cache = {}
        self.memory_access_times = {}
        self.memory_expiry_times = {}

        # Disk cache tracking
        self.disk_cache_index = {}
        self.disk_access_times = {}

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "disk_hits": 0,
            "evictions": 0,
            "writes": 0,
            "deletes": 0,
        }

        # Ensure cache directory exists
        if self.enabled:
            os.makedirs(self.disk_cache_dir, exist_ok=True)
            self._load_disk_cache_index()

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None

        cache_key = self._hash_key(key)
        current_time = time.time()

        # Check memory cache first
        if cache_key in self.memory_cache:
            # Check expiration
            if cache_key not in self.memory_expiry_times or current_time < self.memory_expiry_times[cache_key]:
                # Update access time for LRU
                self.memory_access_times[cache_key] = current_time
                self.stats["hits"] += 1
                self.stats["memory_hits"] += 1

                return self.memory_cache[cache_key]
            else:
                # Expired, remove from memory
                await self._remove_from_memory(cache_key)

        # Check disk cache
        if cache_key in self.disk_cache_index:
            disk_path = self._get_disk_path(cache_key)

            try:
                if os.path.exists(disk_path):
                    # Check expiration
                    if current_time < self.disk_cache_index[cache_key]["expires_at"]:
                        # Load from disk
                        with open(disk_path, 'rb') as f:
                            value = pickle.load(f)

                        # Update access time
                        self.disk_access_times[cache_key] = current_time
                        self.stats["hits"] += 1
                        self.stats["disk_hits"] += 1

                        # Promote to memory cache
                        await self._set_memory(cache_key, value, self.disk_cache_index[cache_key]["expires_at"])

                        return value
                    else:
                        # Expired, remove from disk
                        await self._remove_from_disk(cache_key)
            except Exception:
                # Error reading from disk, remove entry
                await self._remove_from_disk(cache_key)

        self.stats["misses"] += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)

        Returns:
            True if successfully cached
        """
        if not self.enabled:
            return False

        cache_key = self._hash_key(key)
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl

        try:
            # Set in memory cache
            await self._set_memory(cache_key, value, expires_at)

            # Set in disk cache for persistence
            await self._set_disk(cache_key, value, expires_at)

            self.stats["writes"] += 1
            return True

        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache

        Args:
            key: Cache key to delete

        Returns:
            True if deleted, False if not found
        """
        if not self.enabled:
            return False

        cache_key = self._hash_key(key)
        deleted = False

        # Remove from memory
        if cache_key in self.memory_cache:
            await self._remove_from_memory(cache_key)
            deleted = True

        # Remove from disk
        if cache_key in self.disk_cache_index:
            await self._remove_from_disk(cache_key)
            deleted = True

        if deleted:
            self.stats["deletes"] += 1

        return deleted

    async def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries

        Args:
            pattern: Optional pattern to match keys (supports wildcards)

        Returns:
            Number of entries cleared
        """
        if not self.enabled:
            return 0

        cleared_count = 0

        if pattern is None:
            # Clear all
            cleared_count = len(self.memory_cache) + len(self.disk_cache_index)

            # Clear memory
            self.memory_cache.clear()
            self.memory_access_times.clear()
            self.memory_expiry_times.clear()

            # Clear disk
            for cache_key in list(self.disk_cache_index.keys()):
                await self._remove_from_disk(cache_key)

        else:
            # Clear matching pattern
            import fnmatch

            # Check memory cache
            for key in list(self.memory_cache.keys()):
                if fnmatch.fnmatch(key, pattern):
                    await self._remove_from_memory(key)
                    cleared_count += 1

            # Check disk cache
            for key in list(self.disk_cache_index.keys()):
                if fnmatch.fnmatch(key, pattern):
                    await self._remove_from_disk(key)
                    cleared_count += 1

        return cleared_count

    async def _set_memory(self, cache_key: str, value: Any, expires_at: float):
        """Set value in memory cache with LRU eviction"""
        current_time = time.time()

        # Check if we need to evict
        if len(self.memory_cache) >= self.memory_cache_size and cache_key not in self.memory_cache:
            await self._evict_memory_lru()

        # Set value
        self.memory_cache[cache_key] = value
        self.memory_access_times[cache_key] = current_time
        self.memory_expiry_times[cache_key] = expires_at

    async def _set_disk(self, cache_key: str, value: Any, expires_at: float):
        """Set value in disk cache"""
        # Check if we need to evict
        if len(self.disk_cache_index) >= self.disk_cache_size and cache_key not in self.disk_cache_index:
            await self._evict_disk_lru()

        # Write to disk
        disk_path = self._get_disk_path(cache_key)
        with open(disk_path, 'wb') as f:
            pickle.dump(value, f)

        # Update index
        self.disk_cache_index[cache_key] = {"expires_at": expires_at, "created_at": time.time(), "path": disk_path}
        self.disk_access_times[cache_key] = time.time()

        # Save index
        await self._save_disk_cache_index()

    async def _evict_memory_lru(self):
        """Evict least recently used item from memory cache"""
        if not self.memory_access_times:
            return

        # Find LRU item
        lru_key = min(self.memory_access_times.keys(), key=lambda k: self.memory_access_times[k])

        await self._remove_from_memory(lru_key)
        self.stats["evictions"] += 1

    async def _evict_disk_lru(self):
        """Evict least recently used item from disk cache"""
        if not self.disk_access_times:
            return

        # Find LRU item
        lru_key = min(self.disk_access_times.keys(), key=lambda k: self.disk_access_times[k])

        await self._remove_from_disk(lru_key)
        self.stats["evictions"] += 1

    async def _remove_from_memory(self, cache_key: str):
        """Remove item from memory cache"""
        self.memory_cache.pop(cache_key, None)
        self.memory_access_times.pop(cache_key, None)
        self.memory_expiry_times.pop(cache_key, None)

    async def _remove_from_disk(self, cache_key: str):
        """Remove item from disk cache"""
        if cache_key in self.disk_cache_index:
            disk_path = self._get_disk_path(cache_key)
            if os.path.exists(disk_path):
                os.remove(disk_path)

            self.disk_cache_index.pop(cache_key, None)
            self.disk_access_times.pop(cache_key, None)

            await self._save_disk_cache_index()

    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key"""
        return hashlib.sha256(key.encode()).hexdigest()

    def _get_disk_path(self, cache_key: str) -> str:
        """Get disk path for cache key"""
        return os.path.join(self.disk_cache_dir, f"{cache_key}.cache")

    def _load_disk_cache_index(self):
        """Load disk cache index"""
        index_path = os.path.join(self.disk_cache_dir, "index.json")

        try:
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    self.disk_cache_index = data.get("index", {})
                    self.disk_access_times = data.get("access_times", {})
        except Exception:
            self.disk_cache_index = {}
            self.disk_access_times = {}

    async def _save_disk_cache_index(self):
        """Save disk cache index"""
        index_path = os.path.join(self.disk_cache_dir, "index.json")

        try:
            data = {"index": self.disk_cache_index, "access_times": self.disk_access_times}
            with open(index_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass

    async def cleanup_expired(self) -> int:
        """Remove expired entries from cache"""
        if not self.enabled:
            return 0

        current_time = time.time()
        cleaned_count = 0

        # Clean memory cache
        expired_memory_keys = [
            key for key, expires_at in self.memory_expiry_times.items() if current_time >= expires_at
        ]

        for key in expired_memory_keys:
            await self._remove_from_memory(key)
            cleaned_count += 1

        # Clean disk cache
        expired_disk_keys = [key for key, info in self.disk_cache_index.items() if current_time >= info["expires_at"]]

        for key in expired_disk_keys:
            await self._remove_from_disk(key)
            cleaned_count += 1

        return cleaned_count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            "enabled": self.enabled,
            "stats": self.stats.copy(),
            "hit_rate": hit_rate,
            "memory_cache": {
                "size": len(self.memory_cache),
                "max_size": self.memory_cache_size,
                "utilization": len(self.memory_cache) / self.memory_cache_size,
            },
            "disk_cache": {
                "size": len(self.disk_cache_index),
                "max_size": self.disk_cache_size,
                "utilization": len(self.disk_cache_index) / self.disk_cache_size,
            },
        }

    def reset_stats(self):
        """Reset cache statistics"""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "disk_hits": 0,
            "evictions": 0,
            "writes": 0,
            "deletes": 0,
        }
