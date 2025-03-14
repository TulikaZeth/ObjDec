import torch 
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
import time

class ObjectDetection:

    def __init__(self, capture_index):
        print(f"Supervision version: {sv.__version__}")
        
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: " + self.device)
        
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        
        # Define a list of predefined colors for different classes
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue (OpenCV uses BGR)
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (128, 128, 0),  # Teal
            (0, 128, 128),  # Brown
            (128, 0, 128),  # Purple
            (192, 192, 192) # Silver
        ]

    def load_model(self):
        model = YOLO('yolov8m.pt')
        model.fuse()
        return model
        
    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        # Create a copy of the frame to draw on
        annotated_frame = frame.copy()
        
        # Get detections from results
        boxes = results[0].boxes
        xyxys = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        
        # Draw boxes and labels manually
        for i, (xyxy, confidence, class_id) in enumerate(zip(xyxys, confidences, class_ids)):
            # Convert bounding box to integers
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            
            # Select color based on class_id
            color_index = int(class_id) % len(self.colors)
            color = self.colors[color_index]
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label text
            class_name = self.CLASS_NAMES_DICT[class_id]
            label = f"{class_name} {confidence:.2f}"
            
            # Calculate text size
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                annotated_frame, 
                (x1, y1 - label_height - 10), 
                (x1 + label_width, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1
            )
        
        return annotated_frame
    
    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Error: Could not open video capture"
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break

            results = self.predict(frame)
            annotated_frame = self.plot_bboxes(results, frame)
            
            end_time = time.time()
            fps = 1/np.round(end_time - start_time, 2)
            
            cv2.putText(
                annotated_frame, 
                f'FPS: {int(fps)}', 
                (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )

            cv2.imshow('YOLOv8 Detection', annotated_frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Create and run detector
detector = ObjectDetection(capture_index=0)
detector()
