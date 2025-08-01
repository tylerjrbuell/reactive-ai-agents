"""Tool result caching system."""

import time
import json
from typing import Any, Dict, Optional, List, Union
from reactive_agents.utils.logging import Logger


class ToolCache:
    """Handles caching of tool execution results."""

    def __init__(self, enabled: bool = True, ttl: int = 3600, max_entries: int = 1000):
        self.enabled = enabled
        self.ttl = ttl  # Time to live in seconds
        self.max_entries = max_entries
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def generate_cache_key(
        self, tool_name: str, params: Dict[str, Any]
    ) -> Optional[str]:
        """Generate a cache key for the given tool and parameters."""
        if not self.enabled:
            return None

        try:
            return f"{tool_name}:{json.dumps(params, sort_keys=True)}"
        except (TypeError, ValueError):
            # Cannot serialize params, skip caching
            return None

    def get(self, cache_key: str) -> Optional[Any]:
        """Get a cached result if it exists and is not expired."""
        if not self.enabled or not cache_key:
            return None

        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.ttl:
                self.hits += 1
                return cache_entry["result"]
            else:
                # Expired, remove from cache
                del self.cache[cache_key]

        self.misses += 1
        return None

    def put(
        self,
        cache_key: str,
        result: Any,
        execution_time: float = 0.0,
        skip_conditions: Optional[List[str]] = None,
    ) -> bool:
        """Store a result in the cache."""
        if not self.enabled or not cache_key:
            return False

        # Default skip conditions
        if skip_conditions is None:
            skip_conditions = ["nocache"]

        # Check skip conditions in params (if they're part of the cache key)
        for condition in skip_conditions:
            if condition in cache_key:
                return False

        # Don't cache errors, very short results, or failed executions
        result_str = str(result)
        if (
            result_str.lower().startswith("error")
            or execution_time <= 0.1
            or len(result_str.strip()) < 10
        ):
            return False

        # Store in cache
        self.cache[cache_key] = {
            "result": result,
            "timestamp": time.time(),
        }

        # Limit cache size
        self._prune_if_needed()
        return True

    def _prune_if_needed(self):
        """Prune cache if it exceeds max entries."""
        if len(self.cache) > self.max_entries:
            # Simple strategy: remove oldest half
            sorted_cache = sorted(
                self.cache.items(), key=lambda item: item[1]["timestamp"]
            )
            num_to_remove = len(sorted_cache) - (self.max_entries // 2)
            keys_to_remove = [k for k, v in sorted_cache[:num_to_remove]]
            for key in keys_to_remove:
                del self.cache[key]

    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0

        return {
            "enabled": self.enabled,
            "entries": len(self.cache),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl": self.ttl,
        }

    def remove_expired(self):
        """Remove all expired entries from cache."""
        if not self.enabled:
            return

        now = time.time()
        expired_keys = [
            key
            for key, entry in self.cache.items()
            if now - entry["timestamp"] >= self.ttl
        ]

        for key in expired_keys:
            del self.cache[key]

    def get_cache_info(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed cache information, optionally filtered by tool name."""
        info = self.get_stats()

        if tool_name:
            tool_entries = {
                k: v for k, v in self.cache.items() if k.startswith(f"{tool_name}:")
            }
            info.update(
                {
                    "tool_name": tool_name,
                    "tool_entries": len(tool_entries),
                    "tool_entries_detail": {
                        k: {
                            "timestamp": v["timestamp"],
                            "age_seconds": time.time() - v["timestamp"],
                        }
                        for k, v in tool_entries.items()
                    },
                }
            )

        return info
