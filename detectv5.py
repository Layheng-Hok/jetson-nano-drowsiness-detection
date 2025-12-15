import cv2
import numpy as np
import threading
import time
import sys
from playsound import playsound

# --- CONFIGURATION ---
MODEL_PATH = './runs/detect/drowsiness_yolov5/weights/best.onnx' 
SOURCE = 0
ALARM_FILE = "alarm.wav" 
CONF_THRESHOLD = 0.45
NMS_THRESHOLD = 0.45
INPUT_SIZE = (640, 640)

CLASS_NAMES = ['Alert', 'MicroSleep', 'Yawn']
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
    
    # --- Load Model ---
    print(f"Loading model from {MODEL_PATH}...")
    try:
        net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        
        # Try to use CUDA if available (Jetson Nano specific)
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("CUDA backend set successfully.")
        except:
            print("CUDA not available/supported in this OpenCV build. Using CPU.")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Ensure '{MODEL_PATH}' exists and is a valid ONNX file.")
        sys.exit(1)

    # --- Open Video Source ---
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {SOURCE}")
        sys.exit(1)

    print("Starting Inference. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Preprocessing for YOLOv5 ---
        # YOLOv5 expects [1, 3, 640, 640], normalized 0-1
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, INPUT_SIZE, swapRB=True, crop=False)
        net.setInput(blob)
        
        # --- Inference ---
        # Output shape is usually (1, 25200, 5 + Num_Classes)
        outputs = net.forward()
        
        # --- Post-processing ---
        # Prepare lists for NMS
        class_ids = []
        confidences = []
        boxes = []

        # The output is a large array of vectors. We need to unwrap the first dimension.
        # outputs[0] is the matrix of detections
        predictions = outputs[0]
        
        # Get image dimensions to scale boxes back to original size
        img_h, img_w = frame.shape[:2]
        x_factor = img_w / INPUT_SIZE[0]
        y_factor = img_h / INPUT_SIZE[1]

        # Iterate through detections
        # Optimization: Filter by confidence threshold first to avoid loops on weak detections
        rows = predictions.shape[0]

        for i in range(rows):
            row = predictions[i]
            confidence = row[4] # Objectness score

            if confidence >= CONF_THRESHOLD:
                # Get class scores (skip first 5 elements: x, y, w, h, obj_conf)
                classes_scores = row[5:]
                
                # Find the class with maximum score
                class_id = np.argmax(classes_scores)
                max_score = classes_scores[class_id]

                # Final confidence is objectness * class_probability
                # (Note: In some YOLOv5 exports, row[4] is already combined, but usually it's separate)
                final_score = confidence * max_score
                
                if final_score > CONF_THRESHOLD:
                    # Parse Box (Center X, Center Y, Width, Height)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    
                    # Scale to original image
                    left = int((cx - w / 2) * x_factor)
                    top = int((cy - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    class_ids.append(int(class_id))
                    confidences.append(float(final_score))
                    boxes.append([left, top, width, height])

        # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

        status_text = "Status: Alert (Normal)"
        color_status = (0, 255, 0) # Green
        alert_triggered = False

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                cls_id = class_ids[i]
                score = confidences[i]
                
                # Draw Bounding Box
                x, y, w, h = box
                
                # Check Alert Logic
                if cls_id in ALERT_CLASSES:
                    alert_triggered = True
                    color_status = (0, 0, 255) # Red
                    
                    class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                    status_text = f"WARNING: {class_name.upper()} DETECTED!"
                    
                    # Draw Red Box for danger
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    label = f"{class_name}: {score:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # Draw Green Box for safe
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Alarm Logic ---
        if alert_triggered:
            current_time = time.time()
            if current_time - last_alert_time > alert_cooldown:
                threading.Thread(target=play_alarm_sound, daemon=True).start()
                last_alert_time = current_time

        # --- Overlay Status Text ---
        (w, h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (10, 5), (10 + w, 30 + h + 5), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 2)

        # --- Display ---
        cv2.imshow("Drowsiness Detection - YOLOv5 (ONNX)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
