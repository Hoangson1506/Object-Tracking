import numpy as np
import lap
import lap

np.random.seed(42)

def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0])

def iou(bbox_pred, bbox_gt, eps=1e-6):
    """Compute IoU between predicted bbox and ground truth bbox

    Args:
        bbox_pred (ArrayLike): (..., x1, y1, x2, y2)
        bbox_gt (ArrayLike): (..., x1, y1, x2, y2)
    """

    x1 = np.maximum(bbox_pred[..., 0], bbox_gt[..., 0])
    y1 = np.maximum(bbox_pred[..., 1], bbox_gt[..., 1])
    x2 = np.minimum(bbox_pred[..., 2], bbox_gt[..., 2])
    y2 = np.minimum(bbox_pred[..., 3], bbox_gt[..., 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_pred = (bbox_pred[..., 2] - bbox_pred[..., 0]) * (bbox_pred[..., 3] - bbox_pred[..., 1])
    area_gt = (bbox_gt[..., 2] - bbox_gt[..., 0]) * (bbox_gt[..., 3] - bbox_gt[..., 1])

    union = area_pred + area_gt - intersection

    result = intersection / (union + eps)
    return result

def convert_bbox_to_z(bbox):
    """Convert bounding box to z format (x, y, s, r). s is scale/area, r is aspect ratio

    Args:
        bbox (ArrayLike): (x1, y1, x2, y2)
    Returns:
        ArrayLike: (x, y, s, r)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """Convert x state to bounding box format (x1, y1, x2, y2)

    Args:
        x (ArrayLike): (x, y, s, r)
        score (float, optional): confidence score. Defaults to None.
    Returns:
        ArrayLike: (x1, y1, x2, y2) or (x1, y1, x2, y2, score) if score is provided
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    x1 = x[0] - w / 2.0
    y1 = x[1] - h / 2.0
    x2 = x[0] + w / 2.0
    y2 = x[1] + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))