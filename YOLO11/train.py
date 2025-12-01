import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Backbone.backbone import DarkNet
from Neck.neck import DarkFPN
from Head.head import Head

from dataset import YOLODataset
from utils import xywh_iou


class YOLOLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.num_classes = num_classes

    def forward(self, preds, targets):
        """
        preds: (B, 4 + nc, N) → bbox + cls prediction
        targets: list of labels per image
        """
        bs = preds.size(0)
        
        box = preds[:, :4, :]      
        cls = preds[:, 4:, :]     
        
        loss_box = 0
        loss_cls = 0

        for b in range(bs):
            t = targets[b].to(preds.device)  # (num_gt, 5) → cls, x, y, w, h
            
            if t.numel() == 0:
                loss_cls += cls[b].sigmoid().mean() * 0
                continue

            gt_cls = t[:, 0]
            gt_box = t[:, 1:]

            iou = xywh_iou(box[b].transpose(0,1), gt_box)
            loss_box += (1 - iou).mean()

            gt_cls_onehot = torch.zeros(cls.size(1), device=cls.device)
            gt_cls_onehot[gt_cls.long()] = 1
            loss_cls += self.bce(cls[b].mean(1), gt_cls_onehot)

        return loss_box + loss_cls * 0.5

class YOLO(nn.Module):
    def __init__(self, width, depth, csp, nc, filters):
        super().__init__()
        self.backbone = DarkNet(width, depth, csp)
        self.neck = DarkFPN(width, depth, csp)
        self.head = Head(nc=nc, filters=filters)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck((p3, p4, p5))
        return self.head([p3, p4, p5])

def train():
    num_classes = 1   
    img_size = 640
    lr = 1e-4
    batch = 8
    epochs = 50

    width  = [3, 32, 64, 128, 256, 512]
    depth  = [3, 6, 6, 3, 2, 2]
    csp    = [True, True]
    filters = [128, 256, 512]

    train_ds = YOLODataset(
        img_dir="data/images/train",
        lbl_dir="data/labels/train",
        img_size=img_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=4,
        collate_fn=lambda x: list(zip(*x))
    )

    model = YOLO(width, depth, csp, num_classes, filters).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = YOLOLoss(num_classes)

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for imgs, labels in train_loader:
            imgs = torch.stack(imgs).cuda()       
            preds = model(imgs)                 
            
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss / len(train_loader):.4f}")

        torch.save(model.state_dict(), "weights/yolo_custom.pt")


if __name__ == "__main__":
    train()
