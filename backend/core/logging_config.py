"""
Logging configuration for AI Memory API.
Uses loguru for structured logging with JSON format support.
"""
import sys
import logging
from pathlib import Path
from loguru import logger as loguru_logger

from backend.core.config import settings


# Remove default handler
loguru_logger.remove()


# Configure loguru format based on settings
if settings.LOG_FORMAT == "json":
    log_format = (
        '{"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", "level": "{level}", '
        '"message": "{message}", "extra": {extra}}'
    )
else:
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )


# Add console handler
loguru_logger.add(
    sys.stderr,
    format=log_format,
    level=settings.LOG_LEVEL,
    colorize=settings.LOG_FORMAT != "json",
    serialize=settings.LOG_FORMAT == "json",
)


# Add file handler for production
if settings.APP_ENV == "production":
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    loguru_logger.add(
        log_dir / "app.log",
        rotation="500 MB",
        retention="10 days",
        compression="zip",
        format=log_format,
        level=settings.LOG_LEVEL,
        serialize=settings.LOG_FORMAT == "json",
    )


# Intercept standard logging
class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages and redirect to loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Intercept uvicorn and fastapi logs
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]:
    logging.getLogger(logger_name).handlers = [InterceptHandler()]


# Export logger
logger = loguru_logger
