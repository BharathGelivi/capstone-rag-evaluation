"""Serialize lazy cache misses so concurrent requests load models once."""
from functools import lru_cache, wraps
from threading import RLock


def serialized_cache(maxsize=1):
    def decorate(function):
        lock = RLock()
        cached = lru_cache(maxsize=maxsize)(function)

        @wraps(function)
        def call(*args, **kwargs):
            with lock:
                result = cached(*args, **kwargs)
                if result is None or (isinstance(result, tuple) and result and result[-1] is not None):
                    cached.cache_clear()
                return result

        def clear():
            with lock:
                cached.cache_clear()

        call.cache_clear = clear
        call.cache_info = cached.cache_info
        return call
    return decorate
