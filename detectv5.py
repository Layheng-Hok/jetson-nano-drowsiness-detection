import torch
import cv2
import numpy as np
import threading
import time
import sys
import os
import pathlib

# --- CONFIGURATION ---
YOLO_PATH = './yolov5'
MODEL_PATH = './yolov5/runs/train/drowsiness_yolov5/weights/best.pt' 
SOURCE = 0 # Camera index
ALARM_FILE = "alarm.wav" 
CONF_THRESHOLD = 0.50
ALERT_CLASSES = [1, 2] # Classes that trigger alarm
# 0=Alert, 1=MicroSleep, 2=Yawn

last_alert_time = 0
alert_cooldown = 3

temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

def play_alarm_sound():
    try:
        if os.path.exists(ALARM_FILE):
            os.system(f"aplay -q {ALARM_FILE}")
    except Exception as e:
        print(f"Error playing sound: {e}")

def main():
    global last_alert_time
    
    # --- Load Model via PyTorch ---
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # Load custom model locally
        model = torch.hub.load(YOLO_PATH, 'custom', path=MODEL_PATH, source='local')
        model.conf = CONF_THRESHOLD
        
        # Check if CUDA is available for PyTorch
        if torch.cuda.is_available():
            print("Using CUDA (GPU) for inference")
            model.cuda()
        else:
            print("WARNING: Using CPU (Slow)")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {SOURCE}")
        sys.exit(1)

    print("Starting Inference. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Inference ---
        start_time = time.time()
        
        # PyTorch handles the preprocessing/resizing internally
        results = model(frame)
        
        # Extract detections (pandas not needed, we use tensor)
        # xyxy[0] is the tensor of detections for the first image
        detections = results.xyxy[0].cpu().numpy() 
        # Format: [x1, y1, x2, y2, confidence, class_id]

        # --- Visualization ---
        status_text = "Status: Safe"
        color_status = (0, 255, 0)
        alert_triggered = False

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            
            if conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_id = int(cls_id)
                label_name = model.names[cls_id]

                # Alert Logic
                if cls_id in ALERT_CLASSES:
                    alert_triggered = True
                    color_status = (0, 0, 255)
                    status_text = f"WARNING: {label_name.upper()}"
                    box_color = (0, 0, 255)
                else:
                    box_color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"{label_name}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # --- Alarm Trigger ---
        if alert_triggered:
            current_time = time.time()
            if current_time - last_alert_time > alert_cooldown:
                threading.Thread(target=play_alarm_sound, daemon=True).start()
                last_alert_time = current_time

        # --- Info Overlay ---
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 2)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
