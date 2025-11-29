import unittest
import numpy as np
from track.bytetrack import ByteTrack
from track.kalman_box_tracker import KalmanBoxTracker

# --- TEST DATA (5 columns: x1, y1, x2, y2, score) ---
DET_1 = np.array([[10, 10, 50, 50, 0.9], [60, 60, 100, 100, 0.8]])
DET_2 = np.array([[12, 12, 52, 52, 0.95], [63, 63, 103, 103, 0.85]])

# Low confidence detection that overlaps with Object 1
DET_LOW_MATCH = np.array([[10, 10, 50, 50, 0.4]]) 

# Low confidence detection that is far away (Noise)
DET_LOW_NOISE = np.array([[200, 200, 250, 250, 0.4]]) 

# High confidence detection in a new area
DET_NEW = np.array([[200, 200, 250, 250, 0.9]])

DET_EMPTY = np.empty((0, 5))

class TestByteTrackAlgorithm(unittest.TestCase):

    def setUp(self):
        """
        Runs before EACH test method.
        Resets the ID counter so each test starts with ID 0.
        """
        KalmanBoxTracker.count = 0

    def test_01_basic_high_conf_tracking(self):
        """Check standard tracking with high confidence detections."""
        # min_hits=1 to allow immediate output for testing
        tracker_system = ByteTrack(min_hits=1, iou_threshold=0.3)
        
        # Frame 1: Initialization
        tracks1 = tracker_system.update(DET_1)
        self.assertEqual(len(tracks1), 2)
        
        # Frame 2: Association
        tracks2 = tracker_system.update(DET_2)
        self.assertEqual(len(tracks2), 2)
        
        # IDs should persist (e.g. 1 and 2)
        ids1 = set(tracks1[:, 4])
        ids2 = set(tracks2[:, 4])
        self.assertEqual(ids1, ids2)

    def test_02_low_conf_rescue(self):
        """
        ByteTrack Feature: Low confidence detection should match 
        an EXISTING track if IoU is good.
        """
        tracker_system = ByteTrack(min_hits=1)
        
        # 1. Initialize with High Conf
        tracker_system.update(DET_1) 
        
        # 2. Update with Low Conf (Score 0.4 < 0.5 default threshold)
        # This detection overlaps with the first object.
        tracks = tracker_system.update(DET_LOW_MATCH)
        
        # Should return the matched track
        self.assertEqual(len(tracks), 1)
        # The ID should match one of the original IDs (1 or 2)
        self.assertTrue(tracks[0, 4] in [1, 2])

    def test_03_low_conf_noise_filtration(self):
        """
        ByteTrack Feature: Low confidence detection should be IGNORED
        if it does not match an existing track. It should NOT create a new track.
        """
        tracker_system = ByteTrack(min_hits=1)
        
        # 1. Update with standard detections
        tracker_system.update(DET_1)
        initial_track_count = len(tracker_system.trackers) # Should be 2
        
        # 2. Update with Low Conf Noise (Score 0.4, far away)
        tracks = tracker_system.update(DET_LOW_NOISE)
        
        # Expectation: 
        # - The noise is NOT returned in output
        # - The internal tracker list did NOT grow
        
        # Note: 'tracks' might contain the old objects (predicted) depending on logic,
        # but we care about NEW IDs.
        current_ids = [t.id for t in tracker_system.trackers]
        
        self.assertEqual(len(tracker_system.trackers), initial_track_count, 
                         "Low confidence noise spuriously created a new tracker!")

    def test_04_new_track_creation(self):
        """High confidence unmatched detection SHOULD create a new track."""
        tracker_system = ByteTrack(min_hits=1)
        tracker_system.update(DET_1) # IDs 1, 2 created
        
        # New object appears with High Confidence (0.9)
        tracks = tracker_system.update(DET_NEW)
        
        # Output should now have 3 objects (or 1 if others were lost, depending on overlap)
        # But specifically, we check if the new ID exists in the system
        all_ids = [t.id + 1 for t in tracker_system.trackers]
        # ID sequence: 0->1, 1->2, 2->3. So we look for 3.
        self.assertIn(3, all_ids)

    def test_05_track_disappearance_and_deletion(self):
        """Tracks should be deleted after max_age exceeded."""
        tracker_system = ByteTrack(max_age=2, min_hits=1)
        
        tracker_system.update(DET_1) # Age 0
        
        # Frame 2: Empty
        tracker_system.update(DET_EMPTY) # Age 1
        
        # Frame 3: Empty -> max_age reached
        tracker_system.update(DET_EMPTY) # Age 2
        
        # Frame 4: Empty -> Should be deleted
        tracker_system.update(DET_EMPTY)
        
        self.assertEqual(len(tracker_system.trackers), 0)

if __name__ == '__main__':
    unittest.main()