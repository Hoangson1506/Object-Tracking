import os
import argparse
from detect.utils import convert_sequence

parser = argparse.ArgumentParser(description="convert MOT data to YOLO format.")
parser.add_argument("--out_path", type=str, default="data/YOLO", help="Path to store the converted data.")
parser.add_argument("--train_data_path", type=str, default="data/MOT20/train", help="Path to get the MOT train data")
parser.add_argument("--val_test_data_path", type=str, default="data/MOT16/train", help="Path to get the MOT val and test data")
args = parser.parse_args()


os.makedirs(args.out_path, exist_ok=True)

train_path = args.train_data_path
if not os.path.exists(train_path):
    print(f"Path not found: {train_path}")

sequences = [seq for seq in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, seq))]

for seq in sequences:
    seq_path = os.path.join(train_path, seq)
    convert_sequence(seq_path=seq_path, output_path=args.out_path, split="train", seq=seq)

val_test_path = args.val_test_data_path
if not os.path.exists(val_test_path):
    print(f"Path not found: {val_test_path}")

sequences = [seq for seq in os.listdir(val_test_path) if os.path.isdir(os.path.join(val_test_path, seq))]

seq_path = os.path.join(val_test_path, sequences[0])
convert_sequence(seq_path=seq_path, output_path=args.out_path, split="val", seq=sequences[0])

for seq in sequences[1:]:
    seq_path = os.path.join(val_test_path, seq)
    convert_sequence(seq_path=seq_path, output_path=args.out_path, split="test", seq=seq)
