from ultralytics import YOLO
import torch

def train_model():
    model = YOLO("yolov8n.pt") 

    results = model.train(
        data="./dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="drowsiness_yolov8",
        device=0 if torch.cuda.is_available() else "cpu"
    )

    success = model.export(format="onnx")
    print("Training finished. Model saved and exported.")

if __name__ == '__main__':
    train_model()
