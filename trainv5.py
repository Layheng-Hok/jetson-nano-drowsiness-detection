import os
import subprocess
import torch

def train():
    # --- Configuration ---
    weights = "yolov5n.pt" 
    data_yaml = "./dataset/data.yaml"
    project_name = "drowsiness_yolov5"
    batch_size = "32"  
    workers = "8"      

    if not torch.cuda.is_available():
        print("WARNING: CUDA not found. Training will be slow on CPU!")
        device = "cpu"
    else:
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Detected GPU: {gpu_name}")

    # --- Training ---
    print(f"Starting training on {device} with batch size {batch_size}...")
    
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE'] = 'disabled'
    
    train_cmd = [
        "python", "yolov5/train.py",
        "--img", "640",
        "--batch", batch_size,
        "--epochs", "50",
        "--data", data_yaml,
        "--weights", weights,
        "--name", project_name,
        "--device", device,
        "--workers", workers
    ]

    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Training failed. Please check your dataset path and environment.")
        return

    # --- Export for Jetson Nano ---
    best_weights = f"yolov5/runs/train/{project_name}/weights/best.pt"
    
    if os.path.exists(best_weights):
        print(f"Training complete! Exporting {best_weights} to ONNX...")
        
        export_cmd = [
            "python", "yolov5/export.py",
            "--weights", best_weights,
            "--include", "onnx",
            "--opset", "11"
        ]
        subprocess.run(export_cmd, check=True)
        
        print("\nSUCCESS!")
        print(f"1. Take this file: {best_weights.replace('.pt', '.onnx')}")
        print("2. Copy it to your Jetson Nano.")
        print("3. Run it using Python 3.6 on the Nano.")
    else:
        print(f"Error: Could not find trained weights at {best_weights}")

if __name__ == '__main__':
    if not os.path.exists("yolov5"):
        print("Downloading YOLOv5 repository...")
        try:
            subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5"], check=True)
            print("Installing requirements...")
            subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"], check=True)
        except Exception as e:
            print("Network error: Could not clone/install YOLOv5. Ensure it is pre-installed.")
    
    train()
