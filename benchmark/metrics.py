import numpy as np
from benchmark.utils import *
from track.utils import iou, linear_assignment

def compute_mota(preds, gts, iou_threshold=0.5):
    """Compute the MOTA score for object tracking

    Args:
        preds (ArrayLike): predicted tracking [frame_id, x1, y1, x2, y2, track_id]
        gts (ArrayLike): ground truth tracking [frame_id, x1, y1, x2, y2, track_id]
        iou_threshold (float, optional): IoU threshold. Defaults to 0.5.
    """
    pred_frames = group_by_frame(preds)
    gt_frames = group_by_frame(gts)

    all_frames = sorted(set(pred_frames.keys()).union(set(gt_frames.keys())))

    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    total_gt = 0
    prev_id_map = {}

    for frame_id in all_frames:
        pred_t = np.array(pred_frames.get(frame_id, []))
        gt_t = np.array(gt_frames.get(frame_id, []))

        # Total GT objects in this frame
        GT_t = len(gt_t)
        total_gt += GT_t

        if GT_t == 0 and len(pred_t) == 0:
            prev_id_map = {}
            continue

        gt_boxes = gt_t[:, 1:5]
        pred_boxes = pred_t[:, 1:5]
        gt_ids = gt_t[:, 5]
        pred_ids = pred_t[:, 5]

        iou_matrix = iou(gt_boxes[:, np.newaxis], pred_boxes[np.newaxis, :])
        matches = linear_assignment(-iou_matrix)

        matched_pred = set([m[1] for m in matches if iou_matrix[m[0], m[1]] >= iou_threshold])
        matched_gt = set([m[0] for m in matches if iou_matrix[m[0], m[1]] >= iou_threshold])

        FP_t = len(pred_t - len(matched_pred))
        FN_t = GT_t - len(matched_gt)

        total_fn += FN_t
        total_fp += FP_t

        id_switches_t, current_id_map = count_id_switches(matches, gt_ids, pred_ids, prev_id_map)
        total_id_switches += id_switches_t
        prev_id_map = current_id_map

    if total_gt == 0:
        return 1.0
    
    mota = 1.0 - (total_fn + total_fp + total_id_switches) / total_gt
    return mota