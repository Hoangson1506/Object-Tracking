import os
import shutil
import configparser

OUT = r"C:\MOT16_YOLO"
CLASS_ID = 0

def convert_sequence(split, seq):
    seq_path = os.path.join(split, seq)
    img_dir = os.path.join(seq_path, "img1")
    gt_path = os.path.join(seq_path, "gt", "gt.txt")
    seqinfo = os.path.join(seq_path, "seqinfo.ini")

    print(f"\n🔹 Đang xử lý: {seq_path}")

    # Lấy kích thước ảnh
    config = configparser.ConfigParser()
    config.read(seqinfo)
    W = int(config["Sequence"]["imWidth"])
    H = int(config["Sequence"]["imHeight"])

    # output folder
    out_img = os.path.join(OUT, "images", split, seq)
    out_lbl = os.path.join(OUT, "labels", split, seq)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    # Nếu không có ground truth (TEST)
    if not os.path.exists(gt_path):
        print("⚠ Không có gt.txt → chỉ copy ảnh")
        for img in os.listdir(img_dir):
            shutil.copy(os.path.join(img_dir, img), os.path.join(out_img, img))
        return

    # Đọc annotation từ GT
    anns = {}
    for line in open(gt_path, "r"):
        parts = line.strip().split(",")

        frame = int(parts[0])
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        vis = float(parts[8])  # visibility

        # Bỏ bbox không hợp lệ
        if w <= 1 or h <= 1:
            continue
        if vis <= 0:
            continue

        if frame not in anns:
            anns[frame] = []

        anns[frame].append((x, y, w, h))

    # convert từng frame
    for f, boxes in anns.items():
        img_name = f"{f:06d}.jpg"
        src_img = os.path.join(img_dir, img_name)

        if not os.path.exists(src_img):
            print("⚠ Thiếu ảnh:", src_img)
            continue

        shutil.copy(src_img, os.path.join(out_img, img_name))

        # Tạo label YOLO
        with open(os.path.join(out_lbl, img_name.replace(".jpg", ".txt")), "w") as out:
            for (x, y, w, h) in boxes:
                xc = (x + w/2) / W
                yc = (y + h/2) / H
                nw = w / W
                nh = h / H

                # clamp vào [0,1]
                xc = max(0, min(1, xc))
                yc = max(0, min(1, yc))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))

                out.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

    print(f"✅ Xong {seq}")

# MAIN
os.makedirs(OUT, exist_ok=True)

for split in ["train", "test"]:
    if not os.path.exists(split):
        continue
    sequences = [d for d in os.listdir(split) if os.path.isdir(os.path.join(split, d))]
    for seq in sequences:
        convert_sequence(split, seq)

print("\n🎉 DONE! Dataset đầy đủ đã được tạo tại:", OUT)
