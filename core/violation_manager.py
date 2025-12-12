from typing import List
from core.violation import Violation
from core.vehicle import Vehicle
from core.license_plate_recognizer import LicensePlateRecognizer
from supervision import Detections

class ViolationManager:
    """
    Manage violation of tracked vehicles
    """
    def __init__(self, violations: List[Violation], recognizer: LicensePlateRecognizer, **kwargs):
        self.violation_count = {violation.name: 0 for violation in violations}
        self.violations = violations
        self.recognizer = recognizer

    def update(self, vehicles: List[Vehicle], sv_detections: Detections, frame, **kwargs):
        """
        Update violation of tracked vehicles

        Args:
            vehicles (List[Vehicle]): List of tracked vehicles
        """
        for violation in self.violations:
            self.violation_count[violation.name] += len(violation.check_violation(self.recognizer, vehicles, sv_detections, frame, **kwargs))

        return self.violation_count