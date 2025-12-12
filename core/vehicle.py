from track.kalman_box_tracker import KalmanBoxTracker
import time
from utils.storage import MinioClient
from utils.config import load_config

# Load config once
config = load_config()

class Vehicle(KalmanBoxTracker):
    def __init__(self, bbox, class_id=None, **kwargs):
        super().__init__(bbox, **kwargs)

        self.class_id = int(class_id)
        self.is_being_tracked = False
        self.has_violated = False
        self.straight_light_signal_when_crossing = None
        self.going_straight = True
        self.frame_of_violation = None
        self.violation_type = []
        self.violation_time = []
        self.license_plate = None
        self.proof = [] # np.array (Crop images)

    def mark_violation(self, violation_type, frame=None, padding=None, frame_buffer=None, fps=30, save_queue=None):
        if padding is None:
            padding = config['violation']['padding']

        if self.has_violated is True:
            self.has_violated = None
            self.violation_type.append(violation_type)
            self.violation_time.append(time.time())

            if frame is not None:
                x1, y1, x2, y2 = map(int, self.get_state()[0])
                h, w, _ = frame.shape

                self.proof = frame[max(0, y1 - padding):min(h, y2 + padding),
                                    max(0, x1 - padding):min(w, x2 + padding)].copy()
                
                # Save proof and retraining data
                identifier = self.license_plate if self.license_plate else self.id
                
                violation_data = {
                    'vehicle_id': self.id,
                    'identifier': identifier,
                    'violation_type': violation_type,
                    'frame': frame.copy(),
                    'bbox': (x1, y1, x2, y2),
                    'frame_buffer': list(frame_buffer) if frame_buffer else [],
                    'fps': fps,
                    'proof_crop': self.proof
                }

                save_queue.put(violation_data)
