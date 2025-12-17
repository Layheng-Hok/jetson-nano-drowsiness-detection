import os
import subprocess
import sys

def train_and_export():
    # --- Configuration ---
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_SILENT'] = 'true'
    if 'WANDB_API_KEY' in os.environ:
        del os.environ['WANDB_API_KEY']
    
    env = os.environ.copy()
    env['WANDB_MODE'] = 'disabled'
    env['WANDB_SILENT'] = 'true'
    env['WANDB_API_KEY'] = ''

    project_name = "drowsiness_yolov5"
    data_yaml = os.path.abspath("./dataset/data.yaml") 
    weights = "yolov5n.pt"
    batch_size = "32"
    workers = "8"
    
    yolo_dir = "./yolov5"
    train_script = os.path.join(yolo_dir, "train.py")
    export_script = os.path.join(yolo_dir, "export.py")

    # --- Sanity Checks ---
    if not os.path.exists(yolo_dir):
        print(f"ERROR: '{yolo_dir}' not found.")
        return
    if not os.path.exists(data_yaml):
        print(f"ERROR: Dataset config not found at {data_yaml}")
        return

    # --- Training ---
    print(f"Starting OFF-LINE training for {project_name}...")
    
    train_cmd = [
        sys.executable, train_script,
        "--img", "640",
        "--batch", batch_size,
        "--epochs", "50",
        "--data", data_yaml,
        "--weights", weights,
        "--name", project_name,
        "--workers", workers,
        "--exist-ok"
    ]

    try:
        subprocess.run(train_cmd, check=True, env=env)
    except subprocess.CalledProcessError:
        print("Training failed.")
        return

    # --- Export for Jetson Nano ---
    best_weights = os.path.join(yolo_dir, "runs", "train", project_name, "weights", "best.pt")
    
    if os.path.exists(best_weights):
        print(f"\nExporting {best_weights} to ONNX...")
        
        export_cmd = [
            sys.executable, export_script,
            "--weights", best_weights,
            "--include", "onnx",
            "--opset", "11"
        ]
        
        try:
            subprocess.run(export_cmd, check=True, env=env)
            onnx_file = best_weights.replace('.pt', '.onnx')
            print("\nSUCCESS! Transfer this file to Jetson:")
            print(f">> {onnx_file}")
        except subprocess.CalledProcessError:
            print("Export failed.")
    else:
        print(f"Error: Weights not found at {best_weights}")

if __name__ == '__main__':
    train_and_export()
    