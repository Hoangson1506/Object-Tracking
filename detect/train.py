from ultralytics import YOLO
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv12 Object Detection")
    parser.add_argument("--model", type=str, default="yolo12s.pt", help="Path to the YOLOv8 model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--data", type=str, default="data/strap_data.yaml", help="Path to the dataset YAML file")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size for training")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--lr0", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer to use for training")
    parser.add_argument("--name", type=str, default="object_tracking", help="Name of the training run")
    args = parser.parse_args()
    
    model = YOLO(args.model)  # load a pretrained YOLO model

    model.train(data=args.data,
                warmup_epochs=args.warmup_epochs,
                lr0=args.lr0,
                lrf=args.lrf,
                optimizer=args.optimizer,
                cls=0.25,
                box=10,
                dfl=1.5,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                degrees=20.0,         # Rotation: +/- 15 deg
                shear=10.0,           # Shear: +/- 10 deg
                scale=0.5,            # Crop/Zoom: +/- 20%
                fliplr=0.5,           # Flip Horizontal
                flipud=0.5,
                hsv_h=0.15,           # Hue: +/- 15 deg
                hsv_s=0.25,           # Saturation: +/- 25%
                hsv_v=0.25,           # Brightness (15%) + Exposure (10%)
                close_mosaic=5,
                name=args.name
            )