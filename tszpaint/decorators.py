from functools import wraps
from time import perf_counter
from typing import Any, Callable

from loguru import logger


def timer[T](fn: Callable[..., T]):
    """Decorator to measure function runtime. Enabled/disabled by customising the loguru log level:
    https://loguru.readthedocs.io/en/stable/overview.html
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):  # pyright: ignore[reportAny]
        start = perf_counter()
        rv = fn(args, kwargs)
        elapsed = perf_counter() - start
        logger.debug(f"{fn}({args}, {kwargs}) took {elapsed}s")
        return rv

    return wrapper
