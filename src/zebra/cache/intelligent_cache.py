# src/zebra/cache/intelligent_cache.py
from __future__ import annotations
import asyncio
import json
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional
import time
import logging

try:
    import aioredis  # pip install redis[async]
except ImportError:
    aioredis = None

logger = logging.getLogger("zebra.cache")

class CachePredictor:
    """
    PoC: lightweight predictor using recent access Markov-ish counts.
    Can be replaced by an LSTM/LightGBM that learns sequences.
    """
    def __init__(self):
        self.transition = defaultdict(lambda: defaultdict(int))
        self.history = defaultdict(lambda: deque(maxlen=500))

    def observe(self, key: str, timestamp: float):
        last = self.history[key][-1] if self.history[key] else None
        self.history[key].append(timestamp)
        if last:
            self.transition[last][key] += 1

    def predict_next_accesses(self, key: str, access_patterns: Dict) -> List[str]:
        # simple heuristic: most-likely next keys from transition counts
        counts = self.transition.get(key, {})
        sorted_keys = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in sorted_keys[:5]]

class IntelligentCache:
    def __init__(self, redis_url: str = "redis://redis:6379", namespace: str = "zebra:cache"):
        if aioredis is None:
            raise ImportError("aioredis is not installed. Please run `pip install redis[async]`")
        self.redis = aioredis.from_url(redis_url, decode_responses=True)
        self.namespace = namespace
        self.access_patterns = defaultdict(list)
        self.predictor = CachePredictor()

    def _k(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    async def fetch_from_source(self, key: str) -> Any:
        # implement actual backend fetch (DB, object storage, computation)
        # placeholder:
        await asyncio.sleep(0.01)
        return {"value": f"value-of-{key}", "fetched_at": time.time()}

    async def record_access(self, key: str):
        ts = time.time()
        self.access_patterns[key].append(ts)
        self.predictor.observe(key, ts)

    async def prefetch(self, keys: List[str]):
        tasks = [self.get_with_prediction(k, prefetching=True) for k in keys]
        await asyncio.gather(*tasks)

    async def adaptive_ttl_seconds(self, key: str) -> int:
        accesses = self.access_patterns.get(key, [])
        if not accesses:
            return 60 # Default TTL
        freq_per_hour = len(accesses) / ((time.time() - accesses[0]) / 3600.0 + 1e-9)
        if freq_per_hour > 100:
            return 3600
        if freq_per_hour > 10:
            return 600
        return 60

    async def get_with_prediction(self, key: str, prefetching: bool = False) -> Any:
        k = self._k(key)
        val = await self.redis.get(k)
        if val is not None:
            await self.record_access(key)
            return json.loads(val)

        # not in cache: predict and prefetch
        predicted = self.predictor.predict_next_accesses(key, self.access_patterns)
        if predicted and not prefetching:
            # fire-and-forget prefetch
            asyncio.create_task(self.prefetch(predicted))

        value = await self.fetch_from_source(key)
        ttl = await self.adaptive_ttl_seconds(key)
        await self.redis.set(k, json.dumps(value), ex=ttl)
        await self.record_access(key)
        return value

    async def causal_cache_invalidation(self, changed_key: str):
        """
        Use causal dependency graph stored in external store (e.g. Redis graph or service).
        We assume a key `zebra:causal:deps:{key}` contains list of dependent keys.
        """
        dep_key = f"zebra:causal:deps:{changed_key}"
        dependents = await self.redis.smembers(dep_key)
        keys_to_delete = [self._k(d) for d in dependents]
        if keys_to_delete:
            await self.redis.delete(*keys_to_delete)
            logger.info("Invalidated %s due to causal change in %s", dependents, changed_key)
            # optionally trigger async prefetch of dependents' sources