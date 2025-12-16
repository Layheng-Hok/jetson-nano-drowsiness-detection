import cv2
import numpy as np
import threading
import time
import sys
import os

# --- CONFIGURATION ---
# Update this path to where your ONNX file actually resides
MODEL_PATH = './yolov5/runs/train/drowsiness_yolov5/weights/best.onnx' 
SOURCE = 0 # Camera index
ALARM_FILE = "alarm.wav" 
CONF_THRESHOLD = 0.50
NMS_THRESHOLD = 0.45
INPUT_SIZE = (640, 640)

CLASS_NAMES = ['Alert', 'MicroSleep', 'Yawn']
ALERT_CLASSES = [1, 2] # Classes that trigger alarm

last_alert_time = 0
alert_cooldown = 3

def play_alarm_sound():
    """Uses aplay (native Linux) for lower latency than playsound"""
    try:
        # Check if file exists first
        if os.path.exists(ALARM_FILE):
            os.system(f"aplay -q {ALARM_FILE}")
        else:
            print(f"Alarm file not found: {ALARM_FILE}")
    except Exception as e:
        print(f"Error playing sound: {e}")

def main():
    global last_alert_time
    
    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    
    # Enable CUDA (Jetson Nano)
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("CUDA backend set successfully.")
    except Exception as e:
        print("WARNING: CUDA not available. Falling back to CPU (Slow).")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {SOURCE}")
        sys.exit(1)

    print("Starting Inference. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]

        # --- Preprocessing ---
        # YOLOv5 input: 1x3x640x640, normalized 0-1
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, INPUT_SIZE, swapRB=True, crop=False)
        net.setInput(blob)
        
        # --- Inference ---
        start_time = time.time()
        outputs = net.forward()
        # outputs shape: (1, 25200, 5 + Num_Classes)
        
        # --- Post-processing (Vectorized) ---
        predictions = outputs[0]
        
        # 1. Filter by Confidence (Objectness)
        # Keep only rows where objectness (index 4) > CONF_THRESHOLD
        valid_indices = predictions[:, 4] > CONF_THRESHOLD
        valid_predictions = predictions[valid_indices]
        
        boxes = []
        confidences = []
        class_ids = []

        if len(valid_predictions) > 0:
            # 2. Extract Class Scores and IDs
            # valid_predictions shape: (N, 8) -> 5 box params + 3 classes
            class_scores = valid_predictions[:, 5:]
            class_ids_arr = np.argmax(class_scores, axis=1)
            max_scores = np.max(class_scores, axis=1)
            
            # 3. Calculate Final Confidence
            final_scores = valid_predictions[:, 4] * max_scores
            
            # 4. Filter by Final Confidence
            final_indices = final_scores > CONF_THRESHOLD
            
            # Apply final filter
            valid_predictions = valid_predictions[final_indices]
            class_ids_arr = class_ids_arr[final_indices]
            final_scores = final_scores[final_indices]
            
            if len(valid_predictions) > 0:
                # 5. Convert Boxes (Center X, Center Y, W, H) -> (Left, Top, W, H)
                # Scale factors
                x_factor = img_w / INPUT_SIZE[0]
                y_factor = img_h / INPUT_SIZE[1]
                
                cx = valid_predictions[:, 0]
                cy = valid_predictions[:, 1]
                w = valid_predictions[:, 2]
                h = valid_predictions[:, 3]
                
                left = ((cx - w / 2) * x_factor).astype(int)
                top = ((cy - h / 2) * y_factor).astype(int)
                width = (w * x_factor).astype(int)
                height = (h * y_factor).astype(int)
                
                boxes = np.stack((left, top, width, height), axis=1).tolist()
                confidences = final_scores.tolist()
                class_ids = class_ids_arr.tolist()

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

        # --- Visualization ---
        status_text = "Status: Safe"
        color_status = (0, 255, 0)
        alert_triggered = False

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                cls_id = class_ids[i]
                score = confidences[i]
                
                x, y, w, h = box
                
                # Alert Logic
                if cls_id in ALERT_CLASSES:
                    alert_triggered = True
                    color_status = (0, 0, 255)
                    status_text = f"WARNING: {CLASS_NAMES[cls_id].upper()}"
                    box_color = (0, 0, 255)
                else:
                    box_color = (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

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
