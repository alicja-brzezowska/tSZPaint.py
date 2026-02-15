from contextlib import contextmanager
from time import perf_counter
from typing import Any, Callable, ParamSpec, TypeVar

import numpy as np
import wrapt
from loguru import logger

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
