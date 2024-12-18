# PoseEstimation
 A simple Python code sample to use human pose estimation and object detection (YoloV4) with OpenCV

# Skeleton and Object Detection System

This repository contains a Python-based application for real-time skeleton and object detection using OpenCV and additional custom parsing and tracking modules.

## Features

- Real-time video stream processing.
- Detection of objects with bounding boxes and confidence scores.
- Skeleton tracking for multiple bodies, including specific articulations.
- FPS counter for performance monitoring.

## Prerequisites

Before running this application, you need to install the following Python packages:

- `opencv-contrib-python` for video processing and rendering.
- `numpy` for numerical operations.
- `scipy` for system solving.

These packages can be installed using pip:

```bash
pip install opencv-contrib-python numpy scipy
```

You also need files for setting the deep learning models. The models used are available [on this Google Drive](https://drive.google.com/drive/folders/1z2JPdyjzKaJz0SWsFOwZkL8-60u12jn_)  
Download the compressed folder and decompress it at the root of the project.

## Modules Description

- **FPSCounter**: A utility class for measuring and displaying frames per second.
- **Parsers**: Contains functions to parse detection results and extract skeleton information.
- **SkeletonTracker**: Manages tracking of objects and human figures, integrates with YOLO for object detection and a body model for skeleton tracking.

## Usage

To run the application, execute the `main()` function within a Python environment. The default video source is the first connected webcam (index 0). You can change the `video_path` variable in the `main()` function to switch to a different video source or a video file.

```python
python main.py
```

## Application Flow

1. Capture video from the specified source.
2. Resize and flip each frame for consistent processing.
3. Detect objects and track human skeletons in each frame.
4. Draw bounding boxes, detection labels, and confidence scores.
5. Display the skeletal structure for detected human figures.
6. Render the processed frame with an FPS counter in the window titled "Resultat".
7. Exit the application by pressing "q" or ESC key.

## Contributing

This project has no ambition to be used for more than education purposes. However, contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

## License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for more details.
