# utils/cache.py
import redis
import json
import inspect
import pickle
from functools import wraps
from config import config
from utils.logging import logger

class RedisCache:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=config.redis.HOST,
            port=config.redis.PORT,
            db=config.redis.DB,
            password=config.redis.PASSWORD,
            decode_responses=False
        )
    
    def get(self, key):
        """Get data from cache"""
        try:
            data = self.redis_client.get(key)
            return pickle.loads(data) if data else None
        except (redis.RedisError, pickle.UnpicklingError) as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key, value, expire=None):
        """Set data in cache"""
        try:
            if expire is None:
                expire = config.CACHE_EXPIRATION
            self.redis_client.setex(key, expire, pickle.dumps(value))
            return True
        except (redis.RedisError, pickle.PicklingError) as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key):
        """Delete data from cache"""
        try:
            return self.redis_client.delete(key)
        except redis.RedisError as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
        
# Create cache instance
cache = RedisCache()

def cached(key_pattern, expire=None):
    """Decorator to cache function results"""
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Bind args and kwargs to parameter names
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            arg_map = {k: str(v) for k, v in bound.arguments.items()}

            # Generate cache key safely
            try:
                key = key_pattern.format(**arg_map)
            except KeyError as e:
                logger.error(f"Cache key generation failed: missing {e} in {arg_map}")
                return func(*args, **kwargs)

            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {key}")
                return cached_result

            # Execute function if not in cache
            result = func(*args, **kwargs)

            # Store result in cache
            cache.set(key, result, expire)
            return result

        return wrapper
    return decorator