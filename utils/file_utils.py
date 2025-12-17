"""
File path and naming utilities.

This module contains functions for generating output filenames
and handling file paths.
"""

import os
import re
from typing import Tuple


def handle_result_filename(data_path: str, tracker_name: str) -> Tuple[str, str]:
    """
    Generate result filename based on data_path and tracker_name.

    Args:
        data_path: Path to input data (video file or directory)
        tracker_name: Name of the tracking algorithm (e.g., 'sort', 'bytetrack')

    Returns:
        Tuple of (result_filename, extension)
    
    Examples:
        >>> handle_result_filename("data/video.mp4", "bytetrack")
        ('video_bytetrack', '.mp4')
        
        >>> handle_result_filename("data/MOT16-02/img1", "sort")
        ('MOT16-02_sort', '.mp4')
    """
    if os.path.isdir(data_path):
        # Try to extract MOT dataset name from path
        mot_pattern = re.compile(r"(MOT\d{2}-\d{2})", re.IGNORECASE)
        parts = os.path.normpath(data_path).split(os.sep)
        base_name = "result"
        
        for part in reversed(parts):
            match = mot_pattern.search(part)
            if match:
                base_name = match.group(1)
                break

        result_filename = f"{base_name}_{tracker_name}"
        ext = ".mp4"
        return result_filename, ext
    
    else:
        # Extract filename and extension from file path
        base_name, ext = os.path.splitext(os.path.basename(data_path))
        result_filename = f"{base_name}_{tracker_name}"
        return result_filename, ext


def ensure_output_dirs(output_dir: str) -> Tuple[str, str]:
    """
    Ensure output directories exist for video and CSV results.

    Args:
        output_dir: Base output directory

    Returns:
        Tuple of (video_dir, csv_dir)
    """
    video_dir = os.path.join(output_dir, "video")
    csv_dir = os.path.join(output_dir, "csv")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    return video_dir, csv_dir
