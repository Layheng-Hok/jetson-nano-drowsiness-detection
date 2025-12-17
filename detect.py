import cv2
import torch
import numpy as np
import threading
import time
import sys
import os
import pathlib

# --- CONFIGURATION ---
# Path to your custom model
MODEL_PATH = './yolov5/runs/train/drowsiness_yolov5/weights/best.pt' 
SOURCE = 0
ALARM_FILE = "alarm.wav" 
CONF_THRESHOLD = 0.50
NMS_THRESHOLD = 0.45 # IOU Threshold

# Class Definitions
# Ensure these match the classes your model was trained on
CLASS_NAMES = ['Alert', 'MicroSleep', 'Yawn']
ALERT_CLASSES = [1, 2] # IDs of classes that trigger the alarm

last_alert_time = 0
alert_cooldown = 3.0

# --- FIX FOR JETSON/PYTHON 3.6 PATHLIB ISSUE ---
# Older pathlib on Py3.6 can cause issues with YOLOv5 loader
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def play_alarm_sound():
    if os.path.exists(ALARM_FILE):
        try:
            os.system(f"aplay -q {ALARM_FILE}")
        except Exception as e:
            print(f"Audio Error: {e}")

def main():
    global last_alert_time

    # 1. Load Model via Torch Hub
    # We use the 'ultralytics/yolov5' repo to load the custom weights.
    # This automatically handles the model architecture.
    print(f"Loading {MODEL_PATH} on PyTorch...")
    
    try:
        # 'custom' lets us load our own weights. 
        # 'source="github"' fetches the code structure from the internet.
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have internet connection to fetch the YOLOv5 repo structure first.")
        sys.exit(1)

    # 2. Configure Model Settings
    model.conf = CONF_THRESHOLD  # Confidence threshold
    model.iou = NMS_THRESHOLD    # NMS IoU threshold
    
    # 3. Setup Device (CUDA is critical for Jetson)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if device.type == 'cuda':
        print(f">> CUDA Backend ENABLED. Model is on GPU.")
        # Half precision (fp16) runs much faster on Jetson Nano GPUs
        model.half() 
    else:
        print(">> WARNING: CUDA not found. Running on CPU (Very Slow).")

    # 4. Start Camera
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"Error: Camera {SOURCE} not found.")
        sys.exit(1)

    print("Starting Inference. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Inference
        start_time = time.time()
        
        # PyTorch YOLOv5 expects RGB images (OpenCV is BGR)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pass the image to the model
        # The model handles resizing and normalizing internally
        results = model(img_rgb)
        
        # Parse Results
        # .xyxy[0] returns a tensor of shape (N, 6): [x1, y1, x2, y2, confidence, class]
        detections = results.xyxy[0]

        alert_triggered = False

        # Loop through detections
        for *xyxy, conf, cls in detections:
            # Move data to CPU and convert to integer/float
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(conf)
            cls_id = int(cls)

            label = f"{CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else cls_id} {conf:.2f}"
            
            # Logic for Alarm/Color
            if cls_id in ALERT_CLASSES:
                color = (0, 0, 255) # Red for Alert
                alert_triggered = True
            else:
                color = (0, 255, 0) # Green for Safe

            # Draw Box & Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Alarm Trigger Logic
        if alert_triggered:
            cv2.putText(frame, "WARNING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            current_time = time.time()
            if current_time - last_alert_time > alert_cooldown:
                threading.Thread(target=play_alarm_sound, daemon=True).start()
                last_alert_time = current_time

        # FPS calculation
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Jetson PyTorch Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
