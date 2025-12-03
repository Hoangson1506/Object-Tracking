from ultralytics import YOLO
from track.sort import SORT
from track.bytetrack import ByteTrack
from detect.detect import inference_video
from track.utils import ciou, iou
from utils import process_and_write_frame, parse_args_tracking, handle_video_capture, handle_result_filename, select_zones
import cv2
import numpy as np
import os
import csv


if __name__ == "__main__":
    args = parse_args_tracking()

    # Prepare output paths
    result_filename, ext = handle_result_filename(args.data_path, args.tracker)
    video_result_path = os.path.join(args.output_dir, "video", result_filename + ext)
    csv_result_path = os.path.join(args.output_dir, "csv", result_filename + ".csv")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "csv"), exist_ok=True)

    if args.tracker == 'sort':
        tracker_instance = SORT(cost_function=iou, max_age=60, min_hits=5, iou_threshold=0.5)
        conf_threshold = 0.25
    elif args.tracker == 'bytetrack':
        tracker_instance = ByteTrack(cost_function=iou, max_age=60, min_hits=5, high_conf_threshold=0.5)
        conf_threshold = 0.1
    else:
        raise ValueError(f"Unknown tracker: {args.tracker}")

    data_path = args.data_path
    model = YOLO(args.model, task='detect', verbose=True)
    device = "cuda"
    np.random.seed(42)

    # Setup VideoWriter and display Window
    FRAME_WIDTH, FRAME_HEIGHT, FPS, first_frame, ret = handle_video_capture(data_path)
    select_zones(first_frame)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_result_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    cv2.namedWindow("Tracking Results", cv2.WINDOW_AUTOSIZE)

    # Prepare detections
    dets = inference_video(
        model=model,
        data_path=data_path,
        output_path=None,
        device=device,
        stream=True,
        conf_threshold=conf_threshold,
        classes=[0]
    )
    final_results = []

    for i, result in enumerate(dets):
        frame, tracked_objs = process_and_write_frame(i, result, tracker_instance, video_writer)
        final_results.extend(tracked_objs)
        cv2.imshow("Tracking Results", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_writer.release()
    cv2.destroyAllWindows()

    # Save results to CSV
    with open(csv_result_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(final_results)
    print(f"Tracking results succesfully saved to {video_result_path} and {csv_result_path}")
    print(FRAME_WIDTH, FRAME_HEIGHT, FPS)
    print(len(final_results))