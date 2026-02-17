from contextlib import contextmanager
from time import perf_counter
from typing import Any, Callable, ParamSpec, TypeVar

import numpy as np
import psutil
import wrapt
from loguru import logger
import os

P = ParamSpec("P")
R = TypeVar("R")


@contextmanager
def non_verbose_numpy_printing():
    with np.printoptions(threshold=10, edgeitems=2):
        yield


@wrapt.decorator
def time_calls(
    wrapped: Callable[P, R],
    instance: object | None,  # pyright: ignore[reportUnusedParameter]
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> R:
    start = perf_counter()
    result = wrapped(*args, **kwargs)  # pyright: ignore[reportAny]
    elapsed = perf_counter() - start
    with non_verbose_numpy_printing():
        logger.debug(f"{wrapped.__qualname__} took {elapsed:.6f}s")
    return result


@wrapt.decorator
def trace_calls(
    wrapped: Callable[P, R],
    instance: object | None,  # pyright: ignore[reportUnusedParameter]
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> R:
    with non_verbose_numpy_printing():
        logger.trace(f"→ {wrapped.__qualname__}{args}{kwargs}")
    result = wrapped(*args, **kwargs)  # pyright: ignore[reportAny]
    logger.trace(f"← {wrapped.__qualname__}")
    return result


@contextmanager
def timer(label: str):
    """Used to measure runtime of specific code snippets.
    Usage:
        with timer("some operation"):
            print("performing some operation...")
    """
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    logger.debug(f"{label} took {elapsed:.6f}s")


@wrapt.decorator
def memory_usage(
    wrapped: Callable[P, R],
    instance: object | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> R:

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1e6
    
    result = wrapped(*args, **kwargs)
    
    mem_after = process.memory_info().rss / 1e6
    delta = mem_after - mem_before
    logger.info(f"[MEMORY] {wrapped.__qualname__}: {mem_before:.1f}MB → {mem_after:.1f}MB (Δ{delta:+.1f}MB)")
    
    return result


@wrapt.decorator
def array_size(
    wrapped: Callable[P, R],
    instance: object | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> R:
    result = wrapped(*args, **kwargs)
    
    if isinstance(result, np.ndarray):
        size_mb = result.nbytes / 1e6
        logger.info(f"[ARRAY] {wrapped.__qualname__}: {size_mb:.1f}MB (dtype={result.dtype}, shape={result.shape})")
    elif isinstance(result, tuple) and any(isinstance(r, np.ndarray) for r in result):
        for i, r in enumerate(result):
            if isinstance(r, np.ndarray):
                size_mb = r.nbytes / 1e6
                logger.info(f"[ARRAY] {wrapped.__qualname__}[{i}]: {size_mb:.1f}MB (dtype={r.dtype}, shape={r.shape})")
    
    return result
