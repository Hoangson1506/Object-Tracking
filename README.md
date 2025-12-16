# Traffic Violation Detection System

A computer vision system designed to detect and track vehicles, identify traffic violations (specifically red light violations), and capture evidence including license plates. The system utilizes YOLOv12 for detection, SORT/ByteTrack for tracking, and integrates with MinIO for evidence storage.

## Features

-   **Vehicle Detection**: Detects cars, motorcycles, buses, and trucks using YOLOv12.
-   **Object Tracking**: Supports SORT and ByteTrack algorithms for robust real-time vehicle tracking.
-   **Violation Detection**: Detects vehicles crossing stop lines during red phases.
-   **Traffic Light Detection**: Automatically detects traffic light states (Red/Green/Yellow) using computer vision.
-   **Complex Rules**: Supports special violation lines, left/right turn exceptions, and polygon-based zones.
-   **License Plate Recognition**: Automatically crops and reads license plates of violating vehicles using YOLO (detection) and PaddleOCR (text recognition).
-   **Evidence Management**: Captures image crops and video clips of violations.
-   **MinIO Integration**: Detailed storage of proofs, labeled images, and retraining data in S3-compatible buckets.
-   **Data Logging**: Saves violation details (frame, coordinates, ID) to CSV.
-   **Visualizations**: Draws bounding boxes, tracking IDs, and violation zones on the output video.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Object-Tracking
    ```

2.  **Start the services:**
    ```bash
    docker-compose up -d
    ```

    This will start the `traffic-monitor` container and a `minio` server (console at localhost:9001)

3. **Enter the container:**
    ```bash
    docker exec -it traffic-monitor /bin/bash
    ```

## Configuration

The system is highly configurable via `config.yaml` and `zones.json`.

`config.yaml`:
Controls model paths, thresholds, and system settings.
-   `detections`: Confidence/IoU thresholds, classes to detect.
-   `tracking`: Algorithm selection (`sort` or `bytetrack`) and specific hyperparameters (max_age, min_hits).
-   `system`: Paths to model weights (`vehicle_model`, `license_model`) and input data.
-   `violation`: Video proof duration, FPS, padding for crops.

`zones.json`:
Defines the road layout. You can generate this file using the interactive drawing tools in the Web UI or CLI
-   `polygon`: The Region of Interest (ROI) where detection happens.
-   `lines`: Lines defining stop boundaries of exception lines.

## Usage

1. **Web Dashboard (Gradio)**

The most user-friendly way to interact with the system.

```bash
python app.py
```

Access the dashboard at http://localhost:7860.

**Features:**
-   **Dashboard**: View system stats (connection to MinIO).
-   **Visualization**: Start/Stop the live processing feed.
-   **Zone Drawing**: Interactively draw polygons and violation lines on a captured frame and save them to `zones.json`.
-   **Settings**: Hot-swap confidence thresholds, FPS, and model paths.

2. **CLI/Headless Mode**

Run the processing pipeline directly on a video file or RTSP stream.

```bash
python main.py --data_path data/traffic_video.avi --tracker bytetrack --save True
```

**Arguments:**
-   `--data_path`: Path to video file or RTSP URL (e.g., rtsp://localhost:8554/cam).
-   `--vehicle_model`: Path to vehicle detection weights (default: detect_gtvn.pt).
-   `--license_model`: Path to LP detection weights (default: lp_yolo11s.pt).
-   `--tracker`: `sort` or `bytetrack`.
-   `--light_detect`: `True`/`False` to enable automatic traffic light detection.

## Project Structure

-   `core/`: Core logic for the system.
-   `detect/`: YOLOv12 inference and detection utilities.
-   `track/`: Implementation of SORT and ByteTrack algorithms.
-   `utils/`: Helper functions for drawing, IO, and configuration.
-   `main.py`: Entry point of the application.

## Output

Local:
-   `output/csv/`: Tracking data logs.
MinIO (S3):
-   `proofs/`: Stores violation evidence (labeled images, video clips).
-   `retraining-data/`: Stores crops for future model retraining.

## Models
To run this system effectively, make sure you have the following weights in your root directory, the names can be changed in `config.yaml`:
-   **Vehicle Model**: A YOLO model to detect vehicles every frame.
-   **License Plate Model**: A YOLO model to detect where the license plate is.
-   **Character Model**: PaddleOCR is downloaded automatically, but you can use a method of your own.
