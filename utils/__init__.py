"""
Utils package for Traffic Violation Detection System.

This module re-exports commonly used utilities for convenient imports.
"""

# Configuration
from utils.config import load_config, save_config

# Zone management
from utils.zones import load_zones, save_zones

# Storage
from utils.storage import MinioClient

# Logging
from utils.logger import (
    get_logger,
    get_system_logger,
    log_violation,
    log_performance,
    log_upload
)

# Drawing and rendering
from utils.drawing import (
    draw_polygon_zone,
    draw_light_zone,
    draw_line_zone
)
from utils.rendering import draw_violation_overlay, draw_traffic_light_state, render_frame

# I/O utilities
from utils.workers import violation_save_worker
from utils.file_utils import ensure_output_dirs, handle_result_filename

# CLI argument parsing
from utils.parse_args import parse_args_tracking, parse_args_eval


__all__ = [
    # Config
    'load_config', 'save_config',
    # Zones
    'load_zones', 'save_zones',
    # Storage
    'MinioClient',
    # Logging
    'get_logger', 'get_system_logger', 'log_violation', 'log_performance', 'log_upload',
    # Drawing
    'draw_polygon_zone', 'draw_light_zone', 'draw_line_zone',
    'render_frame', 'draw_violation_overlay', 'draw_traffic_light_state',
    # I/O
    'handle_result_filename', 'violation_save_worker', 'ensure_output_dirs',
    # Args
    'parse_args_tracking', 'parse_args_eval'
]
