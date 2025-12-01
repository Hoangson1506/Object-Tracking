import torch

def xywh_to_xyxy(box):
    """
    box: (..., 4) tensor in format (x_center, y_center, width, height)
    return: (..., 4) in format (x1, y1, x2, y2)
    """
    x, y, w, h = box.unbind(-1)

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return torch.stack((x1, y1, x2, y2), dim=-1)


def xywh_iou(box1, box2):
    """
    box1: (N, 4)
    box2: (M, 4)
    returns: (N, M) IoU matrix
    """

    # Convert to xyxy
    b1 = xywh_to_xyxy(box1)
    b2 = xywh_to_xyxy(box2)

    # Intersection
    inter_x1 = torch.max(b1[:, None, 0], b2[:, 0])  # (N, M)
    inter_y1 = torch.max(b1[:, None, 1], b2[:, 1])
    inter_x2 = torch.min(b1[:, None, 2], b2[:, 2])
    inter_y2 = torch.min(b1[:, None, 3], b2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h  # (N, M)

    # Areas
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])  # (N,)
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])  # (M,)

    # Union
    union = area1[:, None] + area2 - inter  # (N, M)

    return inter / union.clamp(min=1e-6)
