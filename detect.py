import cv2
import numpy as np
import threading
import time
import sys
import os

# --- CONFIGURATION ---
MODEL_PATH = 'best.onnx' 
SOURCE = 0
ALARM_FILE = "alarm.wav" 
CONF_THRESHOLD = 0.50
NMS_THRESHOLD = 0.45
INPUT_SIZE = (640, 640)

CLASS_NAMES = ['Alert', 'MicroSleep', 'Yawn']
ALERT_CLASSES = [1, 2]

last_alert_time = 0
alert_cooldown = 3.0

def play_alarm_sound():
    if os.path.exists(ALARM_FILE):
        try:
            os.system(f"aplay -q {ALARM_FILE}")
        except Exception as e:
            print(f"Audio Error: {e}")

def main():
    global last_alert_time
    
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Did you copy 'best.onnx' from the server?")
        sys.exit(1)

    print(f"Loading {MODEL_PATH}...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    
    # Setup Backend (Try CUDA, fall back to CPU)
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print(">> CUDA Backend ENABLED (Fast)")
    except Exception as e:
        print(">> WARNING: CUDA not available/failed. Using CPU (Slow).")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"Error: Camera {SOURCE} not found.")
        sys.exit(1)

    print("Starting Inference. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        img_h, img_w = frame.shape[:2]

        # Preprocessing
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, INPUT_SIZE, swapRB=True, crop=False)
        net.setInput(blob)
        
        # Inference
        start_time = time.time()
        outputs = net.forward() # Shape: (1, 25200, 5 + Num_Classes)
        
        # Post-processing (Vectorized for Jetson Nano speed)
        predictions = outputs[0]
        
        # Filter out low confidence detections immediately to save CPU cycles
        # predictions[:, 4] is the "objectness" score
        valid_indices = predictions[:, 4] > CONF_THRESHOLD
        valid_predictions = predictions[valid_indices]

        boxes = []
        confidences = []
        class_ids = []

        if len(valid_predictions) > 0:
            # Extract class scores (columns 5 onwards)
            class_scores = valid_predictions[:, 5:]
            class_ids_arr = np.argmax(class_scores, axis=1)
            max_scores = np.max(class_scores, axis=1)
            
            # Final confidence = Objectness * Class_Score
            final_scores = valid_predictions[:, 4] * max_scores
            
            # Filter again by final confidence
            final_indices = final_scores > CONF_THRESHOLD
            
            # Apply filter
            valid_predictions = valid_predictions[final_indices]
            class_ids_arr = class_ids_arr[final_indices]
            final_scores = final_scores[final_indices]
            
            if len(valid_predictions) > 0:
                # Calculate scaling factors
                x_factor = img_w / INPUT_SIZE[0]
                y_factor = img_h / INPUT_SIZE[1]
                
                # Extract box coordinates
                cx = valid_predictions[:, 0]
                cy = valid_predictions[:, 1]
                w = valid_predictions[:, 2]
                h = valid_predictions[:, 3]
                
                # Convert (Center_x, Center_y, w, h) to (Top_left_x, Top_left_y, w, h)
                left = ((cx - w / 2) * x_factor).astype(int)
                top = ((cy - h / 2) * y_factor).astype(int)
                width = (w * x_factor).astype(int)
                height = (h * y_factor).astype(int)
                
                boxes = np.stack((left, top, width, height), axis=1).tolist()
                confidences = final_scores.tolist()
                class_ids = class_ids_arr.tolist()

        # NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

        # Visualization & Alarm
        alert_triggered = False
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cls_id = class_ids[i]
                score = confidences[i]
                
                # Check for Alert Classes
                if cls_id in ALERT_CLASSES:
                    color = (0, 0, 255) # Red
                    alert_triggered = True
                else:
                    color = (0, 255, 0) # Green

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Alarm Logic
        if alert_triggered:
            cv2.putText(frame, "WARNING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            current_time = time.time()
            if current_time - last_alert_time > alert_cooldown:
                threading.Thread(target=play_alarm_sound, daemon=True).start()
                last_alert_time = current_time

        # FPS calculation
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (img_w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Jetson Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
