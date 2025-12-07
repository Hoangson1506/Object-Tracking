import supervision as sv
import numpy as np
from core.vehicle import Vehicle
from typing import List
from utils.drawing import draw_line_zone

class Violation:
    """Base class of all type of traffic violations
    """
    def __init__(self, name: str, **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def check_violation(self, vehicles: List[Vehicle]):
        """Check violation of vehicles

        Args:
            vehicles (List[Vehicle]): List of vehicles to check
        """
        raise NotImplementedError


class RedLightViolation(Violation):
    """Red light violation"""
    def __init__(self, **kwargs):
        super().__init__(name="RedLightViolation", **kwargs)

    def check_violation(self, vehicles: List[Vehicle], traffic_light_state: str):
        """Check the violation state of vehicles tracked

        Args:
            vehicles (List[Vehicle]): List of vehicles to check
            traffic_light_state (str): State of the traffic light ("RED", "GREEN", "YELLOW")
        """

        pass

    def draw_line(self, frame: np.ndarray):
        """Draw the violation line on the frame

        Args:
            frame (np.ndarray): Frame to draw the line on
        """
        pass