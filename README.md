# Crowd Management System

A Python-based crowd management system using YOLOv8 and OpenCV. It can detect and count people in real-time from IP cameras, local video files, or webcams, and provide alerts when a threshold is exceeded.

## Features
- Real-time person detection and counting.
- Support for IP camera streams (RTSP/HTTP), video files, and webcams.
- Configurable crowd threshold for alerts.
- Capability to save the processed video with annotations.
- Optional GUI display.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python ultralytics
   ```

## Usage

Run the `crowd_management.py` script with the desired options:

### Using a Webcam
```bash
python crowd_management.py --source 0 --threshold 5 --show
```

### Using an IP Camera
```bash
python crowd_management.py --source "rtsp://username:password@ip_address:port/path" --threshold 10 --show
```

### Using a Video File
```bash
python crowd_management.py --source "path/to/video.mp4" --threshold 5 --output "result.mp4"
```

### Command-line Arguments
- `--source`: Video source (IP camera URL, file path, or '0' for webcam). Default: `0`.
- `--threshold`: Crowd count threshold for alert. Default: `5`.
- `--output`: Path to save the output video. Default: `None`.
- `--model`: YOLOv8 model version (e.g., `yolov8n.pt`, `yolov8s.pt`). Default: `yolov8n.pt`.
- `--show`: Display the video window. Recommended for local use only.

## License
MIT
