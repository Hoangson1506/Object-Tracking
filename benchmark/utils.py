import numpy as np
from collections import defaultdict

def group_by_frame(data_list):
    """Groups detections/tracking data by frame id

    Args:
        data_list (ArrayLike): list of data in the format [frame_id, x1, y1, x2, y2, track_id]
    """
    frames = defaultdict(list)
    for row in data_list:
        frame_id = int(row[0])
        frames[frame_id].append(row)
    return frames

def count_id_switches(matches, gt_ids, pred_ids, prev_id_map):
    """Count ID switches given matches and previous ID mapping

    Args:
        matches (List[Tuple[int, int]]): list of matched (gt_index, pred_index)
        gt_ids (ArrayLike): ground truth IDs for the current frame
        pred_ids (ArrayLike): predicted IDs for the current frame
        prev_id_map (Dict[int, int]): previous mapping from gt_id to pred_id
    """
    id_switches = 0
    current_id_map = {}

    for gt_idx, pred_idx in matches:
        gt_id = gt_ids[gt_idx]
        pred_id = pred_ids[pred_idx]
        current_id_map[gt_id] = pred_id

        if gt_id in prev_id_map:
            if prev_id_map[gt_id] != pred_id:
                id_switches += 1

    return id_switches, current_id_map

