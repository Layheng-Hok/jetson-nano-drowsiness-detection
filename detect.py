import cv2
import torch
import numpy as np
import threading
import time
import sys
import os
import pathlib

# --- CONFIGURATION ---
# Path to your .pt file (Update this to your actual path)
MODEL_PATH = './yolov5/runs/train/drowsiness_yolov5/weights/best.pt' 

# Path to the cloned YOLOv5 folder (must be in the same directory as this script)
YOLO_DIR = './yolov5'

SOURCE = 0              # 0 for USB webcam, or string for video file
ALARM_FILE = "alarm.wav" 
CONF_THRESHOLD = 0.50
NMS_THRESHOLD = 0.45

# Class Names (Must match training)
CLASS_NAMES = ['Alert', 'MicroSleep', 'Yawn']
ALERT_CLASSES = [1, 2]  # Class IDs that trigger the alarm

last_alert_time = 0
alert_cooldown = 3.0

# --- PYTHON 3.6 / JETSON COMPATIBILITY PATCHES ---
# Fixes "NotImplementedError" if model was trained on Windows
pathlib.WindowsPath = pathlib.PosixPath

def play_alarm_sound():
    if os.path.exists(ALARM_FILE):
        try:
            # 'aplay' is the standard command line player for ALSA on Linux
            os.system(f"aplay -q {ALARM_FILE}")
        except Exception as e:
            print(f"Audio Error: {e}")

def main():
    global last_alert_time

    # 1. Check for YOLOv5 folder
    if not os.path.exists(YOLO_DIR):
        print(f"ERROR: '{YOLO_DIR}' folder not found.")
        print("Please run: git clone https://github.com/ultralytics/yolov5")
        sys.exit(1)

    # 2. Load Model using Local Source
    print(f"Loading {MODEL_PATH}...")
    try:
        # source='local' uses the folder on your disk, avoiding the python 3.8 dependency issue
        model = torch.hub.load(YOLO_DIR, 'custom', path=MODEL_PATH, source='local')
    except Exception as e:
        print(f"\nCRITICAL ERROR LOADING MODEL: {e}")
        print("Tip: If you get a 'requirements' error, try installing pandas/requests manually.")
        sys.exit(1)

    # 3. Configure Model Settings
    model.conf = CONF_THRESHOLD
    model.iou = NMS_THRESHOLD
    
    # 4. Setup Device (Force CUDA for Jetson)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if device.type == 'cuda':
        print(f">> CUDA Backend ENABLED. Model running on GPU.")
        # FP16 is essential for Jetson Nano performance
        model.half() 
    else:
        print(">> WARNING: CUDA not found. Running on CPU (Will be slow).")

    # 5. Initialize Camera
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"Error: Camera {SOURCE} not found.")
        sys.exit(1)

    print("Starting Inference. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Start timer for FPS
        start_time = time.time()
        
        # YOLOv5 expects RGB, OpenCV gives BGR
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = model(img_rgb)
        
        # Process Detections
        # .xyxy[0] is a tensor: [x1, y1, x2, y2, confidence, class]
        detections = results.xyxy[0]

        alert_triggered = False

        for *xyxy, conf, cls in detections:
            # Move to CPU and convert to correct types
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(conf)
            cls_id = int(cls)

            # Get Label Name
            if cls_id < len(CLASS_NAMES):
                label_name = CLASS_NAMES[cls_id]
            else:
                label_name = str(cls_id)

            label = f"{label_name} {conf:.2f}"
            
            # Determine Color and Alert Status
            if cls_id in ALERT_CLASSES:
                color = (0, 0, 255) # Red for Alert
                alert_triggered = True
            else:
                color = (0, 255, 0) # Green for Safe

            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label Background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Alarm Logic
        if alert_triggered:
            cv2.putText(frame, "WARNING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            current_time = time.time()
            if current_time - last_alert_time > alert_cooldown:
                # Run sound in separate thread to prevent freezing video
                threading.Thread(target=play_alarm_sound, daemon=True).start()
                last_alert_time = current_time

        # Calculate and Display FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show Output
        cv2.imshow("Jetson Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    