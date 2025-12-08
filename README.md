# Traffic Violation Detection System

This project is a computer vision application designed to detect traffic violations, specifically red light violations, using deep learning and object tracking techniques. It utilizes YOLO for object detection and supports SORT and ByteTrack for object tracking.

## Features

-   **Object Detection**: Real-time vehicle detection using Ultralytics YOLO (default: `yolo12n.pt`).
-   **Object Tracking**: Robust vehicle tracking using SORT or ByteTrack algorithms.
-   **Violation Detection**: Automatically identifies vehicles crossing a designated line or zone during a red light phase.
-   **Visualization**: Comprehensive visual output with bounding boxes, tracking IDs, detection zones, and status annotators using `supervision`.
-   **Data Storage**: Integration with MinIO for robust storage of video proofs and results.
-   **Evaluation**: Built-in tools to evaluate tracking performance against ground truth data using `motmetrics`.

## Project Structure

-   `core/`: Contains core logic for vehicle and violation definitions.
-   `detect/`: Modules for running object detection inference.
-   `track/`: Implementations of SORT and ByteTrack trackers.
-   `utils/`: Utility scripts for I/O, config loading, drawing, and argument parsing.
-   `benchmark/`: Metrics and evaluation tools.
-   `config.yaml`: Configuration file for detection, tracking, and violation parameters.
-   `main.py`: The entry point for the application.

## Installation

### Prerequisites

-   Python 3.8+
-   Docker and Docker Compose (for MinIO)
-   CUDA-compatible GPU (recommended for performance)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Object-Tracking
    ```

2.  **Install dependencies:**
    You can use the provided setup script or install requirements manually.

    ```bash
    bash start.sh
    ```
    *Or manually:*
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Start MinIO Service:**
    This project uses MinIO for data storage. Start it using Docker Compose:
    ```bash
    To start the entire stack (MinIO + Traffic Monitor):
    ```bash
    docker-compose up -d
    ```
    *Note: The `traffic-monitor` service is configured to expect a GPU by default.*

4.  **Docker Support (Manual Run):**

    Build the Docker image:
    ```bash
    docker build -t traffic-monitor .
    ```

    Run with GPU (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):
    ```bash
    docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output traffic-monitor python main.py --device cuda --data_path data/video.mp4 --save True
    ```

    Run with CPU:
    ```bash
    docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output traffic-monitor python main.py --device cpu --data_path data/video.mp4 --save True
    ```

## Usage

### Running the Tracker

Use `main.py` to process a video file.

```bash
python main.py --data_path data/video.mp4 --device cuda
```

**Arguments:**

-   `--data_path`: Path to the input video file (default: `data/MOT16 test video/MOT16-01-raw.mp4`).
-   `--model`: Path to the YOLO model weights (default: `yolo12n.pt`).
-   `--tracker`: Tracking algorithm to use: `sort` or `bytetrack` (default: `sort`).
-   `--output_dir`: Directory to save the resulting video and CSV (default: `output`).
-   `--save`: Save processed video: `True` or `False` (default: `True`).
-   `--device`: computation device: `cuda` or `cpu` (default: `cuda`).

**Example:**

```bash
python main.py --data_path data/traffic_cam.mp4 --tracker bytetrack --model yolo12n.pt
```

### Configuration

You can fine-tune detection, tracking, and violation parameters in `config.yaml`.

-   **`detections`**: Confidence and IoU thresholds, target classes.
-   **`tracking`**: Specific parameters for SORT (max_age, min_hits) and ByteTrack.
-   **`violation`**: Video proof duration, default FPS, padding.

### Evaluation

To evaluate the tracking performance against ground truth data (MOT format):

```bash
python evaluate.py --pred_path output/csv/result.csv --gt_path data/gt.txt
```

**Arguments:**

-   `--pred_path`: Path to the tracking output CSV.
-   `--gt_path`: Path to the ground truth text file.
-   `--metrics`: List of metrics to compute (e.g., `mota`, `idf1`).

## Results

The system outputs:
1.  **Video**: An annotated video file in `output/video/` showing tracked vehicles and violations.
2.  **CSV**: A CSV file in `output/csv/` containing tracking data and violation status for each frame.
