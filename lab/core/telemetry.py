"""Structured logging helper.

One logger (``lab``) is configured from the global settings. Everywhere else
uses ``logging.getLogger("lab.<module>")``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from lab.config import Settings


_CONFIGURED = False


def configure(settings: Settings) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("lab")
    if _CONFIGURED:
        return logger

    level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")

    if settings.logging.console:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    log_file = settings.abspath(settings.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    _CONFIGURED = True
    return logger


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"lab.{name}")
