import cv2
import torch
import numpy as np
import os
from pathlib import Path

class YOLODataset:
    def __init__(self, img_dir, lbl_dir, img_size=640):
        self.paths = sorted(list(Path(img_dir).rglob("*.jpg")))
        self.lbl_dir = lbl_dir
        self.img_size = img_size

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label_path = f"{self.lbl_dir}/{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = torch.tensor(img / 255., dtype=torch.float32)

        labels = []
        if os.path.exists(label_path):
            for line in open(label_path):
                cls, x, y, w, h = map(float, line.split())
                labels.append([cls, x, y, w, h])
        labels = torch.tensor(labels)

        return img, labels

    def __len__(self):
        return len(self.paths)
