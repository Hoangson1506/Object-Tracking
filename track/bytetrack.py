from track.utils import *
from track.kalman_box_tracker import KalmanBoxTracker
from track.base_tracker import BaseTracker

class ByteTrack(BaseTracker):
    """This is the ByteTrack algorithm for Object Tracking
    """

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        super().__init__()
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0