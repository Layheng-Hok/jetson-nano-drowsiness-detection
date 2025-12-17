import os
import subprocess
import torch
import shutil

def train():
    # --- Configuration ---
    weights = "yolov5n.pt" 
    data_yaml = "./dataset/data.yaml"
    project_name = "drowsiness_yolov5"
    # Jetson Nano has limited RAM; keep batch size conservative (16 or 32)
    batch_size = "16"  
    workers = "2"      

    # --- Device Selection ---
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found. Training will be slow on CPU!")
        device = "cpu"
    else:
        device = "0"
        print(f"Detected GPU: {torch.cuda.get_device_name(0)}")

    # --- Training ---
    print(f"Starting training on {device} with batch size {batch_size}...")
    
    # Disable WANDB to prevent login prompts blocking execution
    os.environ['WANDB_DISABLED'] = 'true'
    
    # Define paths
    yolo_dir = os.path.join(os.getcwd(), "yolov5")
    train_script = os.path.join(yolo_dir, "train.py")
    export_script = os.path.join(yolo_dir, "export.py")
    
    train_cmd = [
        "python", train_script,
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
    except subprocess.CalledProcessError:
        print("Training failed. Check dataset path and memory usage.")
        return

    # --- Locate Best Weights ---
    # YOLOv5 saves to runs/train/project_name/weights/best.pt relative to where you run it
    # Since we run from root, it should be in:
    result_path = os.path.join("runs", "train", project_name, "weights", "best.pt")
    
    # Fallback search if path differs (YOLOv5 directory structure changes sometimes)
    if not os.path.exists(result_path):
        result_path = os.path.join("yolov5", "runs", "train", project_name, "weights", "best.pt")

    if os.path.exists(result_path):
        print(f"\nTraining complete! Found weights at: {result_path}")
        print("Exporting to ONNX for Jetson Nano...")
        
        # --- Export to ONNX ---
        # Opset 11 is usually safest for Jetson/TensorRT/OpenCV 4.x compatibility
        export_cmd = [
            "python", export_script,
            "--weights", result_path,
            "--include", "onnx",
            "--opset", "11" 
        ]
        
        try:
            subprocess.run(export_cmd, check=True)
            onnx_path = result_path.replace('.pt', '.onnx')
            print("\nSUCCESS!")
            print(f"ONNX Model generated: {onnx_path}")
            
            # Copy to current directory for easy access
            dest = "best_drowsiness.onnx"
            shutil.copy(onnx_path, dest)
            print(f"Copied to root folder as: {dest}")
            
        except subprocess.CalledProcessError as e:
            print(f"Export failed: {e}")
    else:
        print(f"Error: Could not find trained weights. Expected at {result_path}")

if __name__ == '__main__':
    if not os.path.exists("yolov5"):
        print("Error: 'yolov5' folder not found. Please run 'git clone' step first.")
    else:
        train()
        