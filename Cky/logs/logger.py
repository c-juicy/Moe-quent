# Copy from vllm
import logging
from types import MethodType
from functools import lru_cache
from logging import Logger
from typing import Hashable, cast

@lru_cache
def _print_debug_once(logger: Logger, msg: str, *args: Hashable) -> None:
    logger.debug(msg, *args, stacklevel=2)

@lru_cache
def _print_info_once(logger: Logger, msg: str, *args: Hashable) -> None:
    logger.info(msg, *args, stacklevel=2)

@lru_cache
def _print_warning_once(logger: Logger, msg: str, *args: Hashable) -> None:
    logger.warning(msg, *args, stacklevel=2)
    

_METHODS_TO_PATCH = {
    "debug_once": _print_debug_once,
    "info_once": _print_info_once,
    "warning_once": _print_warning_once,
}

def init_logger(name: str) -> Logger:
    logger = logging.getLogger(name)

    for method_name, method in _METHODS_TO_PATCH.items():
        setattr(logger, method_name, MethodType(method, logger))

    # 默认简单格式
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s",
            datefmt="%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return cast(Logger, logger)