# Traffic Violation Detection System

A robust computer vision system for detecting traffic violations, specifically red light violations, using YOLOv8 for object detection and SORT/ByteTrack for object tracking.

## Features

-   **Vehicle Detection**: Detects cars, motorcycles, buses, and trucks using YOLOv8.
-   **Object Tracking**: Supports SORT and ByteTrack algorithms for robust vehicle tracking.
-   **Violation Detection**: Identifies vehicles crossing a defined line during a red light phase.
-   **Video Proof**: Generates video clips of detected violations.
-   **Data Logging**: Saves violation details (frame, coordinates, ID) to CSV.
-   **Visualizations**: Draws bounding boxes, tracking IDs, and violation zones on the output video.

## Installation

### Prerequisites

-   Python 3.8 or higher
-   CUDA-compatible GPU (recommended for real-time performance)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Object-Tracking
    ```

2.  **Install dependencies:**
    You can use the provided helper script or install manually.

    *   **Using script:**
        ```bash
        chmod +x start.sh
        ./start.sh
        ```

    *   **Manual installation:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        ```

## Configuration

The system is highly configurable via `config.yaml`.

### Detection (`detections`)
-   `conf_threshold`: Confidence threshold for YOLO detections (default: 0.25).
-   `iou_threshold`: IOU threshold for NMS (default: 0.5).
-   `classes`: List of COCO class IDs to detect (e.g., `[2, 3, 5, 7]` for vehicles).
-   `input_size`: Input image size for the model (default: 640).

### Tracking (`tracking`)
Configure parameters for `sort` or `bytetrack` (e.g., `max_age`, `min_hits`, `iou_threshold`).

### Violation (`violation`)
-   `video_proof_duration`: Duration (in seconds) of the video clip saved for each violation.

## Usage

Run the main script with the desired arguments:

```bash
python main.py --model <path_to_model> --data_path <path_to_video> [options]
```

### Arguments

-   `--model`: Path to the YOLOv8 model file (default: `yolov8n.pt`).
-   `--data_path`: Path to the input video file (required).
-   `--output_dir`: Directory to save results (default: `output`).
-   `--device`: Device to run inference on (default: `cuda:0`).
-   `--tracker`: Tracker to use: `sort` or `bytetrack` (default: `bytetrack`).
-   `--save`: Save tracking results to CSV (`True`/`False`, default: `False`).

### Example

```bash
# Run with ByteTrack and save results
python main.py --model yolov8n.pt --data_path data/traffic.mp4 --tracker bytetrack --save True
```

## Project Structure

-   `core/`: Core logic for vehicle and violation classes.
-   `detect/`: YOLOv8 inference and detection utilities.
-   `track/`: Implementation of SORT and ByteTrack algorithms.
-   `utils/`: Helper functions for drawing, IO, and configuration.
-   `main.py`: Entry point of the application.

## Output

Results are saved in the `output/` directory (or as specified by `--output_dir`):

-   `video/`: Processed video with annotations.
-   `csv/`: CSV file containing tracking data and violation flags.
