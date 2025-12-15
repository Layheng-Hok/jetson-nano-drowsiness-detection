import cv2
import threading
import time
from ultralytics import YOLO
from playsound import playsound
import sys

# --- CONFIGURATION ---
MODEL_PATH = './runs/detect/drowsiness_yolov8/weights/best.pt' 
SOURCE = 0
ALARM_FILE = "alarm.wav" 
CONF_THRESHOLD = 0.45

# Dataset Classes:
# 0: Alert (Safe/Normal)
# 1: MicroSleep (Danger)
# 2: Yawn (Danger)

# Trigger the alarm for MicroSleep (1) and Yawn (2)
ALERT_CLASSES = [1, 2]
last_alert_time = 0
alert_cooldown = 3

def play_alarm_sound():
    try:
        playsound(ALARM_FILE)
    except Exception as e:
        print(f"Error playing sound: {e}")

def main():
    global last_alert_time
    
    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure 'best.pt' is in the same directory.")
        sys.exit(1)

    # Open Video Source
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {SOURCE}")
        sys.exit(1)

    print("Starting Inference. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, conf=CONF_THRESHOLD, verbose=False) 
        
        # Analyze results
        annotated_frame = results[0].plot() # YOLOv8 built-in visualization
        detections = results[0].boxes
        
        status_text = "Status: Alert (Normal)"
        color_status = (0, 255, 0) # Green

        alert_triggered = False

        if detections:
            for box in detections:
                cls_id = int(box.cls[0])
                
                # Check if the detected object is in our alert list (MicroSleep or Yawn)
                if cls_id in ALERT_CLASSES:
                    alert_triggered = True
                    
                    # Update status text based on specific detection
                    class_name = model.names[cls_id]
                    status_text = f"WARNING: {class_name.upper()} DETECTED!"
                    
                    color_status = (0, 0, 255) # Red
                    break # Trigger once per frame is enough
        
        # Alarm Logic
        if alert_triggered:
            current_time = time.time()
            if current_time - last_alert_time > alert_cooldown:
                # Run sound in separate thread to prevent freezing video
                threading.Thread(target=play_alarm_sound, daemon=True).start()
                last_alert_time = current_time

        # Overlay Status Text
        # Adding a black background rectangle for better text visibility
        (w, h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(annotated_frame, (10, 5), (10 + w, 30 + h + 5), (0, 0, 0), -1)
        cv2.putText(annotated_frame, status_text, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 2)

        # Display
        cv2.imshow("Drowsiness Detection - YOLOv8", annotated_frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
