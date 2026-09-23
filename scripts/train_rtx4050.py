#!/usr/bin/env python3
"""
train_rtx4050.py: Optimized local YOLOv8-seg training for NVIDIA RTX 4050 (6GB VRAM)
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
import torch

# 1. Memory and Compute Optimizations for 6GB RTX 4050
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True

def main():
    print("=" * 70)
    print("🚀 NVIDIA RTX 4050 Local YOLOv8-Seg Training Pipeline")
    print("=" * 70)

    # 2. Check GPU & CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Device:      {gpu_name} ({vram:.2f} GB VRAM)")
    else:
        print("\n⚠️ CUDA is not active in this environment!")
        print("Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    # 3. Locate Dataset
    workspace_dir = Path(__file__).resolve().parent.parent
    possible_paths = [
        Path("/home/utkarsh/Desktop/floorPlanGenerator/dataset_5000"),
        workspace_dir / "data" / "dataset_5000",
    ]
    dataset_dir = next((p for p in possible_paths if p.exists()), None)

    if not dataset_dir:
        # Check ZIP
        zip_candidates = [
            Path("/home/utkarsh/Desktop/floorPlanGenerator/dataset_5000.zip"),
            workspace_dir / "data" / "dataset_5000.zip",
        ]
        zip_f = next((z for z in zip_candidates if z.exists()), None)
        if zip_f:
            extract_to = workspace_dir / "data" / "dataset_5000"
            print(f"Extracting {zip_f} to {extract_to}...")
            with zipfile.ZipFile(zip_f, 'r') as z:
                z.extractall(extract_to)
            dataset_dir = extract_to

    if not dataset_dir:
        print("❌ Could not find dataset_5000 directory or zip.")
        sys.exit(1)

    print(f"Dataset path:    {dataset_dir}")

    # Generate verified local data.yaml
    CLASSES = ["bedroom", "living_room", "kitchen", "bathroom", "hallway", "balcony", "room", "door", "window", "wall"]
    yaml_path = dataset_dir / "data.yaml"
    train_path = dataset_dir / "images" / "train"
    val_path = dataset_dir / "images" / "val"

    yaml_lines = [
        f"path: {dataset_dir.resolve()}",
        f"train: {train_path.resolve()}",
        f"val: {val_path.resolve()}",
        "",
        "names:",
    ]
    for idx, c in enumerate(CLASSES):
        yaml_lines.append(f"  {idx}: {c}")
    yaml_lines.append("")
    yaml_path.write_text("\n".join(yaml_lines))

    # 4. Model Setup & Auto-Checkpointing Callback
    from ultralytics import YOLO

    weights_dir = workspace_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    resume_ckpt = weights_dir / "rtx4050_last.pt"
    can_resume = resume_ckpt.exists()

    if can_resume:
        print(f"🔄 Resuming training from: {resume_ckpt}")
        model = YOLO(str(resume_ckpt))
    else:
        print("📦 Starting fresh from yolov8s-seg.pt...")
        model = YOLO("yolov8s-seg.pt")

    def epoch_callback(trainer):
        ep = trainer.epoch + 1
        tot = trainer.epochs
        if trainer.best.exists():
            shutil.copy2(trainer.best, weights_dir / "rtx4050_best.pt")
        if trainer.last.exists():
            shutil.copy2(trainer.last, weights_dir / "rtx4050_last.pt")
        print(f"\n💾 [Epoch {ep}/{tot}] Saved weights to: {weights_dir / 'rtx4050_best.pt'}")

    model.add_callback("on_fit_epoch_end", epoch_callback)

    # 5. Train with RTX 4050 parameters
    print("\nStarting training loop...")
    print(f"Batch: 8 | Imgsz: 640 | Epochs: 50 | Workers: 4 | AMP: True")
    
    if can_resume:
        model.train(resume=True)
    else:
        model.train(
            data=str(yaml_path),
            epochs=50,
            batch=8,
            imgsz=640,
            device=0,
            workers=4,
            amp=True,
            retina_masks=True,
            save=True,
            verbose=True
        )

    print("\n✅ Training complete! Best weights saved to:", weights_dir / "rtx4050_best.pt")

if __name__ == "__main__":
    main()
