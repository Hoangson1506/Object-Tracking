import unittest
import numpy as np
from track.sort import SORT
from track.kalman_box_tracker import KalmanBoxTracker
from track.utils import *

# --- KHỞI TẠO DỮ LIỆU TEST ---
DET_1 = np.array([[10, 10, 50, 50, 0.9], [60, 60, 100, 100, 0.8]])
DET_2 = np.array([[12, 12, 52, 52, 0.95], [63, 63, 103, 103, 0.85]])
DET_EMPTY = np.empty((0, 5))
DET_NEW = np.array([[200, 200, 250, 250, 0.9]])

class TestSORTAlgorithm(unittest.TestCase):

    def setUp(self):
        """
        Runs before EACH test method.
        Resets the ID counter so each test starts with ID 0.
        """
        KalmanBoxTracker.count = 0

    def test_01_utility_functions(self):
        """Kiểm tra các hàm tiện ích: IoU và Chuyển đổi BBox."""
        bbox_a = np.array([0, 0, 10, 10])
        bbox_b = np.array([5, 5, 15, 15])
        bbox_c = np.array([100, 100, 110, 110])
        
        expected_iou = 25 / 175
        self.assertAlmostEqual(iou(bbox_a, bbox_b), expected_iou, places=4)
        self.assertAlmostEqual(iou(bbox_a, bbox_c), 0.0)

        # Test convert functions
        z = convert_bbox_to_z(bbox_a).flatten()
        recon = convert_x_to_bbox(z).flatten()
        self.assertTrue(np.allclose(recon, bbox_a, atol=1e-5))

    def test_02_kalman_box_tracker_initialization(self):
        """Kiểm tra khởi tạo và dự đoán ban đầu."""
        tracker = KalmanBoxTracker(DET_1[0, :4])
        # Because we reset count in setUp, this should be 0
        self.assertEqual(tracker.id, 0)
        self.assertEqual(tracker.hit_streak, 0)
        
        tracker.predict()
        self.assertEqual(tracker.time_since_update, 1)

    def test_03_simple_sort_tracking(self):
        """Kiểm tra theo dõi đơn giản (2 frame, 2 track, không mất mát)."""
        tracker_system = SORT(min_hits=3)
        
        # Frame 1: Khởi tạo
        tracker_system.update(DET_1)
        self.assertEqual(len(tracker_system.trackers), 2)
        # Check initial streak (should be 0 or 1 depending on implementation, usually 1 after first update)
        self.assertEqual(tracker_system.trackers[0].hit_streak, 0)

        # Frame 2: Update (Match)
        tracker_system.update(DET_2)
        self.assertEqual(tracker_system.trackers[0].hit_streak, 1, "Hit streak failed to increase. Check iou() broadcasting!")
        
        # Frame 3: Update (Match)
        output_frame3 = tracker_system.update(DET_2)
        
        # Check IDs (Should be 1 and 2 because SORT outputs id+1)
        returned_ids = sorted(output_frame3[:, 4])
        self.assertTrue(np.allclose(returned_ids, [1.0, 2.0]))

    def test_04_track_disappearance_and_deletion(self):
        """Kiểm tra mất mát và xóa track (max_age)."""
        tracker_system = SORT(max_age=3, min_hits=3)
        
        # 3 Updates -> streak should be 3
        tracker_system.update(DET_1)
        tracker_system.update(DET_2)
        tracker_system.update(DET_2)
        
        self.assertEqual(tracker_system.trackers[0].hit_streak, 2)
        
        # Frame 4: Mất detections
        tracker_system.update(DET_EMPTY) 
        self.assertEqual(tracker_system.trackers[0].time_since_update, 1)

        # Frame 5: Mất detections
        tracker_system.update(DET_EMPTY)
        self.assertEqual(tracker_system.trackers[0].time_since_update, 2)
        
        # Frame 6: Mất detections
        tracker_system.update(DET_EMPTY) 

        # Frame 7: Mất detections -> Track should be deleted
        tracker_system.update(DET_EMPTY)
        self.assertEqual(len(tracker_system.trackers), 0)

    def test_05_unmatched_detections_and_new_tracks(self):
        """Kiểm tra việc tạo track mới."""
        tracker_system = SORT(min_hits=3)
        tracker_system.update(DET_1) # Creates ID 0, 1

        all_dets = np.concatenate((DET_2, DET_NEW), axis=0)
        tracker_system.update(all_dets)
        
        self.assertEqual(len(tracker_system.trackers), 3)
        # Because we reset count in setUp, ID sequence is 0, 1 -> New is 2
        self.assertEqual(tracker_system.trackers[-1].id, 2)

if __name__ == '__main__':
    unittest.main()