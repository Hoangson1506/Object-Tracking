"""
Frame rendering and annotation utilities.

This module contains functions for annotating video frames with
tracking information, bounding boxes, and labels.
"""

import cv2
import numpy as np
import supervision as sv
from typing import List, Any


def render_frame(
    tracked_objs: List[Any],
    frame: np.ndarray,
    sv_detections: sv.Detections,
    box_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
) -> np.ndarray:
    """
    Process a single detection result, draws bbox, writes the frame.

    Args:
        tracked_objs: List of tracked objects (KalmanBoxTracker instances)
        frame: The frame to annotate
        sv_detections: Detections result in the supervision format
        box_annotator: Supervision BoxAnnotator instance
        label_annotator: Supervision LabelAnnotator instance

    Returns:
        Annotated frame with bounding boxes and labels
    """
    frame = box_annotator.annotate(
        scene=frame,
        detections=sv_detections
    )

    labels = [
        f"ID: {obj.id} {'[VIOLATION]' if len(obj.violation_type) > 0 else ''}" 
        for obj in tracked_objs
    ]
    frame = label_annotator.annotate(
        scene=frame,
        detections=sv_detections,
        labels=labels
    )

    return frame


def draw_violation_overlay(
    frame: np.ndarray,
    text: str,
    position: tuple = (10, 30),
    color: tuple = (0, 0, 255),
    font_scale: float = 0.8,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a text overlay on the frame (e.g., for violation warnings).

    Args:
        frame: The frame to annotate
        text: Text to display
        position: (x, y) position for text
        color: BGR color tuple
        font_scale: Font scale factor
        thickness: Text thickness

    Returns:
        Frame with text overlay
    """
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    return frame


def draw_traffic_light_state(
    frame: np.ndarray,
    states: List[str],
    position: tuple = (10, 60)
) -> np.ndarray:
    """
    Draw traffic light state indicator on frame.

    Args:
        frame: The frame to annotate
        states: List of states [left, straight, right] - can be 'RED', 'GREEN', 'YELLOW', or None
        position: Starting (x, y) position

    Returns:
        Frame with traffic light state overlay
    """
    labels = ['L', 'S', 'R']  # Left, Straight, Right
    colors = {
        'RED': (0, 0, 255),
        'GREEN': (0, 255, 0),
        'YELLOW': (0, 255, 255),
        None: (128, 128, 128)  # Gray for undefined
    }
    
    x, y = position
    for i, (label, state) in enumerate(zip(labels, states)):
        color = colors.get(state, (128, 128, 128))
        # Draw circle
        cv2.circle(frame, (x + i * 40, y), 15, color, -1)
        # Draw label
        cv2.putText(frame, label, (x + i * 40 - 5, y + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame
