"""
Centralized logging utility for Traffic Violation Detection System.

Features:
- Colored console output using colorama
- File logging with rotation
- Configurable log levels
- Helper functions for violation and performance logging
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback: define empty color strings
    class Fore:
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""
        RESET = ""
    
    class Style:
        BRIGHT = ""
        DIM = ""
        RESET_ALL = ""


# Color mapping for log levels
LEVEL_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output."""
    
    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt, datefmt)
        self.fmt = fmt or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        self.datefmt = datefmt or "%H:%M:%S"
    
    def format(self, record: logging.LogRecord) -> str:
        # Get color for this level
        color = LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        
        # Format the message
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        
        return super().format(record)


class PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no colors)."""
    
    def __init__(self, fmt: str = None, datefmt: str = None):
        fmt = fmt or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = datefmt or "%Y-%m-%d %H:%M:%S"
        super().__init__(fmt, datefmt)


# Global logger cache
_loggers: Dict[str, logging.Logger] = {}


def get_logger(
    name: str = "traffic_system",
    level: int = logging.INFO,
    console: bool = True,
    file_logging: bool = False,
    log_dir: str = "logs",
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Get or create a configured logger.
    
    Args:
        name: Logger name (usually module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Enable colored console output
        file_logging: Enable file logging
        log_dir: Directory for log files
        max_file_size: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    # Return cached logger if exists
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Clear any existing handlers
    
    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_logging:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(PlainFormatter())
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Cache the logger
    _loggers[name] = logger
    
    return logger


def log_violation(
    logger: logging.Logger,
    vehicle_id: int,
    violation_type: str,
    license_plate: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a traffic violation with structured information.
    
    Args:
        logger: Logger instance
        vehicle_id: ID of the violating vehicle
        violation_type: Type of violation (e.g., "Red Light", "Speeding")
        license_plate: Detected license plate (if available)
        details: Additional details dictionary
    """
    plate_info = f" | Plate: {license_plate}" if license_plate else ""
    details_info = f" | {details}" if details else ""
    
    logger.warning(
        f"VIOLATION DETECTED | Vehicle ID: {vehicle_id} | "
        f"Type: {violation_type}{plate_info}{details_info}"
    )


def log_performance(
    logger: logging.Logger,
    fps: float,
    frame_count: int,
    processing_time_ms: Optional[float] = None,
    tracked_objects: Optional[int] = None
) -> None:
    """
    Log performance metrics.
    
    Args:
        logger: Logger instance
        fps: Current frames per second
        frame_count: Current frame number
        processing_time_ms: Processing time in milliseconds
        tracked_objects: Number of tracked objects
    """
    metrics = [f"FPS: {fps:.1f}", f"Frame: {frame_count}"]
    
    if processing_time_ms is not None:
        metrics.append(f"Processing: {processing_time_ms:.1f}ms")
    
    if tracked_objects is not None:
        metrics.append(f"Tracked: {tracked_objects}")
    
    logger.debug(" | ".join(metrics))


def log_upload(
    logger: logging.Logger,
    bucket: str,
    filename: str,
    success: bool,
    error: Optional[str] = None
) -> None:
    """
    Log file upload status.
    
    Args:
        logger: Logger instance
        bucket: Storage bucket name
        filename: Uploaded file name
        success: Whether upload was successful
        error: Error message if failed
    """
    if success:
        logger.info(f"Uploaded to {bucket}/{filename}")
    else:
        logger.error(f"Failed to upload to {bucket}/{filename}: {error}")


# Convenience function to get the default system logger
def get_system_logger() -> logging.Logger:
    """Get the default traffic system logger."""
    return get_logger("traffic_system")
