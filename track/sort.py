from track.utils import *
from track.kalman_box_tracker import KalmanBoxTracker

class SORT:
    """This is the SORT (Simple Online and Realtime Tracking) algorithm for Object Tracking
    """

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
            dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
            Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
            Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        self.frame_count += 1
        tracks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for i in range(len(tracks)):
            pos = self.trackers[i].predict()[0]
            tracks[i, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(i)

        tracks = np.ma.compress_rows(np.ma.masked_invalid(tracks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, tracks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        for i in unmatched_dets:
            tracker = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(tracker)

        i = len(self.trackers)
        for tracker in reversed(self.trackers):
            d = tracker.get_state()[0]
            if (tracker.time_since_update < 1) and (tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [tracker.id + 1])).reshape(1, -1))
            i -= 1

            if (tracker.time_since_update > self.max_age):
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        
        return np.empty((0, 5))
    


def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
    """Assigns detections to tracked object

    Args:
        detections (ArrayLike): bbox detections
        trackers (ArrayLike): Estimated bbox from trackers
        iou_threshold (float, optional): IoU threshold. Defaults to 0.3.

    Returns:
        matches, unmatched_detections and unmatched_trackers
    """

    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    iou_matrix = iou(detections[:, np.newaxis], trackers[np.newaxis, :])

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_detections = []
    for d, _ in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, _ in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

