# YOLOv8 Object Detection

This project implements real-time object detection using the YOLOv8 model from Ultralytics and OpenCV. The script captures video from a webcam, processes frames using YOLOv8, and displays detected objects with bounding boxes and labels.

## Features
- Uses **YOLOv8m** for object detection.
- Displays bounding boxes with class labels and confidence scores.
- Real-time FPS display.
- Supports CUDA (GPU) if available.
- Custom colors for different object classes.

## Requirements
Make sure you have the following dependencies installed:

```bash
pip install torch numpy opencv-python ultralytics supervision
```

## Usage

1. **Clone the Repository** (if applicable):
   ```bash
   git clone https://github.com/yourusername/yolo-object-detection.git
   cd yolo-object-detection
   ```
2. **Run the Script:**
   ```bash
   python detection.py
   ```
3. **Press `ESC` to Exit.**

## File Structure
```
project-folder/
│── detection.py      # Main script for object detection
│── README.md         # Documentation file
```

## How It Works
1. **Initialize Model:** Loads YOLOv8m and sets up device (CPU/GPU).
2. **Capture Frames:** Reads frames from the webcam.
3. **Run Predictions:** Detects objects using YOLO.
4. **Draw Bounding Boxes:** Uses OpenCV to annotate frames.
5. **Show Output:** Displays annotated video with FPS.

## Troubleshooting
- Ensure your webcam is working properly.
- If running on GPU, make sure **CUDA** is installed and compatible.
- If `supervision` module causes errors, update it:
  ```bash
  pip install --upgrade supervision
  ```

## Acknowledgments
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)

## License
This project is open-source and available under the **MIT License**.

