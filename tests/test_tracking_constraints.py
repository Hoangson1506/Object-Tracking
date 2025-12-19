import pytest
import numpy as np
from track.sort import SORT
from track.bytetrack import ByteTrack
from track.kalman_box_tracker import KalmanBoxTracker

class MockTracker(KalmanBoxTracker):
    def __init__(self, bbox, class_id=-1, **kwargs):
        super().__init__(bbox, class_id=class_id, **kwargs)

@pytest.mark.parametrize("tracker_cls", [SORT, ByteTrack])
def test_class_constraint(tracker_cls):
    """
    Test that detections of different classes are not associated with existing trackers.
    """
    tracker = tracker_cls(tracker_class=MockTracker, min_hits=0) # min_hits=0 to return trackers immediately
    
    # Frame 1: Detect object A (Class 0)
    # [x1, y1, x2, y2, score, class_id]
    dets_frame1 = np.array([
        [100, 100, 200, 200, 0.9, 0] 
    ])
    
    trackers_frame1 = tracker.update(dets_frame1)
    assert len(trackers_frame1) == 1
    id_a = trackers_frame1[0].id
    assert trackers_frame1[0].class_id == 0
    
    # Frame 2: Object A moves slightly, and Object B (Class 1) appears exactly where Object A was
    # This simulates a scenario where if class constraint wasn't there, it might wrongly associate B to A's track
    # if A disappeared or moved far away. But here let's put A slightly moved and B at A's old spot.
    
    dets_frame2 = np.array([
        [105, 105, 205, 205, 0.9, 0], # A moved slightly
        [100, 100, 200, 200, 0.9, 1]  # B appears at A's old spot
    ])
    
    trackers_frame2 = tracker.update(dets_frame2)
    
    # Should have 2 trackers now
    assert len(trackers_frame2) == 2
    
    # Verify IDs and Classes
    ids = [t.id for t in trackers_frame2]
    classes = [t.class_id for t in trackers_frame2]
    
    assert id_a in ids
    assert 0 in classes
    assert 1 in classes
    
    # Verify that the tracker with id_a still has class 0
    tracker_a = next(t for t in trackers_frame2 if t.id == id_a)
    assert tracker_a.class_id == 0
    
    # Verify that the new tracker has class 1
    tracker_b = next(t for t in trackers_frame2 if t.id != id_a)
    assert tracker_b.class_id == 1

@pytest.mark.parametrize("tracker_cls", [SORT, ByteTrack])
def test_class_constraint_swap(tracker_cls):
    """
    Test that even if a detection of class 1 perfectly overlaps a tracker of class 0,
    it is NOT associated.
    """
    tracker = tracker_cls(tracker_class=MockTracker, min_hits=0)
    
    # Frame 1: Class 0
    dets_frame1 = np.array([[100, 100, 200, 200, 0.9, 0]])
    trackers1 = tracker.update(dets_frame1)
    id_1 = trackers1[0].id
    
    # Frame 2: Class 1 exactly same spot
    dets_frame2 = np.array([[100, 100, 200, 200, 0.9, 1]])
    trackers2 = tracker.update(dets_frame2)
    
    # Should be a NEW tracker, not the old one
    assert len(trackers2) == 1
    assert trackers2[0].id != id_1
    assert trackers2[0].class_id == 1
