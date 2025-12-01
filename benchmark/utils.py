import numpy as np
import pandas as pd
import motmetrics as mm

def convert_pred(pred_df):
    x1 = pred_df['x_topleft'].values
    y1 = pred_df['y_topleft'].values
    x2 = pred_df['x_bottomright'].values
    y2 = pred_df['y_bottomright'].values

    w = x2 - x1
    h = y2 - y1

    converted = pd.DataFrame({
        'frame': pred_df['frame'],
        'id': pred_df['id'],
        'x': x1,
        'y': y1,
        'w': w,
        'h': h
    })

    return converted

def get_mot_accum(pred_df, gt_df, min_vis=0, iou_threshold=0.5):
    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(gt_df['frame'].unique())

    for frame in all_frames:
        gt_frame = gt_df[gt_df['frame'] == frame]
        pred_frame = pred_df[pred_df['frame'] == frame]
        
        valid_mask = (gt_frame['class'] == 1) & (gt_frame['visibility'] > min_vis)
        ignore_mask = (gt_frame['class'].isin([2, 7, 8, 12])) | ((gt_frame['class'] == 1) & (gt_frame['visibility'] <= min_vis))
        
        gt_valid = gt_frame[valid_mask]
        gt_ignore = gt_frame[ignore_mask]

        gt_valid_ids = gt_valid['id'].values
        gt_valid_box = gt_valid[['x', 'y', 'w', 'h']].values
        
        gt_ignore_box = gt_ignore[['x', 'y', 'w', 'h']].values

        pred_ids = pred_frame['id'].values
        pred_box = pred_frame[['x', 'y', 'w', 'h']].values

        # filter tracker output agaisnt extractor (refllection, static person, ...)
        keep_tracker_mask = np.ones(len(pred_ids), dtype=bool)
        
        if len(gt_ignore_box) > 0 and len(pred_box) > 0:
            # Calculate IoU between all tracker boxes and all ignore boxes
            # iou_matrix returns distance (1 - IoU). So IoU = 1 - distance
            dist_matrix_ignore = mm.distances.iou_matrix(gt_ignore_box, pred_box, max_iou=iou_threshold)
            
            # If a tracker box is "close" (IoU > 0.5) to ANY ignore box, we drop it.
            # In iou_matrix, NaN means "too far". Real number means "overlap".
            # We look for columns (tracker boxes) that have at least one valid overlap (non-NaN)
            # with a distractor row.
            
            for t_idx in range(len(pred_ids)):
                # Check if this tracker box (column t_idx) overlaps any ignore gt (rows)
                # The distance matrix contains NaNs for no-overlap.
                overlaps = ~np.isnan(dist_matrix_ignore[:, t_idx])
                if np.any(overlaps):
                    keep_tracker_mask[t_idx] = False
        
        res_ids_clean = pred_ids[keep_tracker_mask]
        res_box_clean = pred_box[keep_tracker_mask]

        dist_matrix = mm.distances.iou_matrix(gt_valid_box, res_box_clean, max_iou=0.5)

        acc.update(
            gt_valid_ids,         # Ground Truth IDs
            res_ids_clean,        # Tracker IDs
            dist_matrix           # Distance matrix (IoU)
        )
        
    return acc