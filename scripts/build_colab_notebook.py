"""Generates 100% self-contained Google Colab notebook for floor plan YOLOv8 segmentation."""

import base64
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_WHL = REPO_ROOT / "dist" / "floorplan_reader-0.1.0-py3-none-any.whl"
SAMPLE_MS = REPO_ROOT / "data" / "test_samples" / "sample_master_suite.png"
SAMPLE_1BHK = REPO_ROOT / "data" / "test_samples" / "sample_1bhk.png"

if not DIST_WHL.exists():
    raise FileNotFoundError(f"Wheel file not found at {DIST_WHL}. Run 'pip wheel --no-deps -w dist .' first.")

whl_b64 = base64.b64encode(DIST_WHL.read_bytes()).decode("ascii")
ms_b64 = base64.b64encode(SAMPLE_MS.read_bytes()).decode("ascii")
b1_b64 = base64.b64encode(SAMPLE_1BHK.read_bytes()).decode("ascii")

step2_source = [
    "# ==============================================================================\n",
    "# Step 2: Automatic Self-Contained Setup (Zero Git Clone or File Upload Needed)\n",
    "# Unpacks the floorplan_reader package and test blueprints directly into Colab.\n",
    "# ==============================================================================\n",
    "import os, sys, base64, subprocess\n",
    "from pathlib import Path\n",
    "\n",
    "# 1. Unpack & Install floorplan_reader package\n",
    "try:\n",
    "    from floorplan_reader.segmentation.yolo_segmenter import YoloFloorPlanSegmenter\n",
    "    from floorplan_reader.segmentation.dataset_exporter import YoloDatasetExporter\n",
    "    from floorplan_reader.visualization.segmentation_visualizer import draw_styled_segmentation_overlay\n",
    '    print("floorplan_reader package is already installed and ready!")\n',
    "except ImportError:\n",
    '    print("Installing floorplan_reader package from embedded wheel...")\n',
    f'    WHL_B64 = "{whl_b64}"\n',
    '    whl_tmp = Path("/tmp/floorplan_reader-0.1.0-py3-none-any.whl")\n',
    "    whl_tmp.write_bytes(base64.b64decode(WHL_B64))\n",
    '    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", str(whl_tmp)])\n',
    "    \n",
    "    from floorplan_reader.segmentation.yolo_segmenter import YoloFloorPlanSegmenter\n",
    "    from floorplan_reader.segmentation.dataset_exporter import YoloDatasetExporter\n",
    "    from floorplan_reader.visualization.segmentation_visualizer import draw_styled_segmentation_overlay\n",
    '    print("Successfully installed and imported floorplan_reader!")\n',
    "\n",
    "# 2. Extract Test Blueprint Samples\n",
    'test_dir = Path("data/test_samples")\n',
    "test_dir.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    'ms_path = test_dir / "sample_master_suite.png"\n',
    "if not ms_path.exists():\n",
    f'    ms_path.write_bytes(base64.b64decode("{ms_b64}"))\n',
    '    print("Extracted: data/test_samples/sample_master_suite.png")\n',
    "\n",
    'b1_path = test_dir / "sample_1bhk.png"\n',
    "if not b1_path.exists():\n",
    f'    b1_path.write_bytes(base64.b64decode("{b1_b64}"))\n',
    '    print("Extracted: data/test_samples/sample_1bhk.png")\n',
    "\n",
    'print("Environment and test blueprints are 100% ready!")\n',
]

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Floor Plan Instance Segmentation & Dimension Analysis (YOLOv8-Seg)\n",
                "\n",
                "This notebook is **100% self-contained** and runs out-of-the-box in Google Colab:\n",
                "- **Primary Goal**: Detect all rooms and extract their real-world dimensions (length, width, area).\n",
                "- **Secondary Goal**: Detect all doors (swings/openings) and windows (wall placements).\n",
                "- **Core Technology**: YOLOv8-seg instance segmentation + Wall-snapping contour geometry + OCR dimension calibration.\n",
                "\n",
                "> **Colab Runtime Recommendation**: Go to `Runtime` -> `Change runtime type` -> select **T4 GPU**."
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Install System Dependencies\n",
                "Installs Ultralytics YOLO, Tesseract OCR, and required geometry utilities."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Install Python packages\n",
                '!pip install -q "ultralytics>=8.3.0" "pytesseract>=0.3.10" "pyclipper>=1.3.0" "shapely>=2.0.0" "cairosvg>=2.7.0"\n',
                "\n",
                "# 2. Install Tesseract OCR engine for reading blueprint dimension texts\n",
                "!apt-get update -qq > /dev/null\n",
                "!apt-get install -y -qq tesseract-ocr libtesseract-dev > /dev/null\n",
                "\n",
                "import torch\n",
                "import ultralytics\n",
                'print(f"Ultralytics version: {ultralytics.__version__}")\n',
                'print(f"CUDA Available: {torch.cuda.is_available()}")\n',
                "if torch.cuda.is_available():\n",
                '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n',
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Self-Contained Package & Data Setup\n",
                "Automatically unpacks and installs `floorplan_reader` and loads test blueprints."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": step2_source,
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Prepare YOLO Segmentation Dataset\n",
                "Formats floor plan ground truth into YOLOv8 polygon instance segmentation (`data.yaml`)."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from floorplan_reader.segmentation.dataset_exporter import YoloDatasetExporter\n",
                "from floorplan_reader.segmentation.yolo_segmenter import YoloFloorPlanSegmenter\n",
                "\n",
                'dataset_dir = Path("data/yolo_dataset")\n',
                "exporter = YoloDatasetExporter(dataset_dir)\n",
                "exporter.initialize_directories()\n",
                "yaml_path = exporter.write_data_yaml()\n",
                "\n",
                "# Export training samples\n",
                "segmenter = YoloFloorPlanSegmenter()\n",
                'for sample_name in ["sample_master_suite.png", "sample_1bhk.png"]:\n',
                '    p = Path(f"data/test_samples/{sample_name}")\n',
                "    if p.exists():\n",
                "        res = segmenter.analyze(p)\n",
                '        exporter.add_sample(p, res.to_clean_dict(), split="train", sample_id=p.stem)\n',
                '        exporter.add_sample(p, res.to_clean_dict(), split="val", sample_id=f"{p.stem}_val")\n',
                "\n",
                'print(f"YOLO segmentation dataset created at: {yaml_path}")\n',
                "print(yaml_path.read_text())\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Train YOLOv8 Instance Segmentation Model\n",
                "Trains `yolov8n-seg.pt` on the prepared floor plan dataset."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from ultralytics import YOLO\n",
                "\n",
                "# Load pretrained nano segmentation weights\n",
                'model = YOLO("yolov8n-seg.pt")\n',
                "\n",
                "# Train on dataset (15-25 epochs on Colab T4 GPU)\n",
                "epochs = 20 if torch.cuda.is_available() else 3\n",
                "results = model.train(\n",
                "    data=str(yaml_path),\n",
                "    epochs=epochs,\n",
                "    imgsz=640,\n",
                '    batch=8 if torch.cuda.is_available() else 2,\n',
                '    device=0 if torch.cuda.is_available() else "cpu",\n',
                "    plots=True,\n",
                "    verbose=True\n",
                ")\n",
                "\n",
                'best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"\n',
                'print(f"Training complete! Best weights saved to: {best_weights}")\n',
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Run Architectural Analysis (Primary & Secondary Goals)\n",
                "Executes room segmentation, dimension extraction, and door/window detection."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from PIL import Image\n",
                "import matplotlib.pyplot as plt\n",
                "from floorplan_reader.segmentation.yolo_segmenter import YoloFloorPlanSegmenter\n",
                "from floorplan_reader.visualization.segmentation_visualizer import draw_styled_segmentation_overlay\n",
                "\n",
                "# Initialize segmenter with trained weights (and automatic wall-snapping fallback)\n",
                "analyzer = YoloFloorPlanSegmenter(\n",
                "    weights_path=best_weights if best_weights.exists() else None,\n",
                "    confidence_threshold=0.25\n",
                ")\n",
                "\n",
                "test_blueprints = [\n",
                '    "data/test_samples/sample_master_suite.png",\n',
                '    "data/test_samples/sample_1bhk.png"\n',
                "]\n",
                "\n",
                "for img_file in test_blueprints:\n",
                "    if not Path(img_file).exists():\n",
                "        continue\n",
                '    print("\\n=========================================")\n',
                '    print(f"Analyzing: {img_file}")\n',
                '    print("=========================================")\n',
                "    \n",
                "    analysis = analyzer.analyze(img_file)\n",
                "    \n",
                '    print("PRIMARY GOAL (Rooms & Real Dimensions):")\n',
                "    for r in analysis.rooms:\n",
                '        dim_str = f"{r.real_length:.2f}m x {r.real_width:.2f}m" if r.real_length else "N/A"\n',
                '        area_str = f"{r.real_length * r.real_width:.2f} sq m" if r.real_length else "N/A"\n',
                '        print(f"  • {r.name.upper():16}: {dim_str} | Area: {area_str} | Box: {r.box_2d.to_list()}")\n',
                "        \n",
                '    print("\\nSECONDARY GOAL (Doors & Windows):")\n',
                '    print(f"  • Doors detected:   {len(analysis.doors)}")\n',
                "    for d in analysis.doors:\n",
                '        print(f"     - Door {d.id} ({d.door_type}) at {d.box_2d.to_list()}")\n',
                '    print(f"  • Windows detected: {len(analysis.windows)}")\n',
                "    for w in analysis.windows:\n",
                '        print(f"     - Window {w.id} ({w.window_type}) at {w.box_2d.to_list()}")\n',
                "        \n",
                "    # Render and display visual overlay\n",
                '    out_overlay_path = f"data/test_samples/{Path(img_file).stem}_result_overlay.png"\n',
                "    vis_img = draw_styled_segmentation_overlay(img_file, analysis, output_path=out_overlay_path)\n",
                "    \n",
                "    fig, ax = plt.subplots(figsize=(10, 8))\n",
                "    ax.imshow(vis_img)\n",
                '    ax.axis("off")\n',
                '    ax.set_title(f"Detected Rooms, Dimensions, Doors & Windows: {Path(img_file).name}", fontsize=14)\n',
                "    plt.show()\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Interactive Upload (Analyze Your Own Floor Plan)\n",
                "Upload your own blueprint image (.png, .jpg, or .svg) to analyze it directly."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import files\n",
                "import io\n",
                "\n",
                'print("Upload a floor plan image (PNG, JPG, or SVG):")\n',
                "uploaded = files.upload()\n",
                "\n",
                "for filename in uploaded.keys():\n",
                '    print(f"\\nProcessing uploaded drawing: {filename}...")\n',
                "    res = analyzer.analyze(filename)\n",
                "    \n",
                '    print(f"\\nDetected {len(res.rooms)} Rooms:")\n',
                "    for r in res.rooms:\n",
                '        d_str = f"{r.real_length:.2f}m x {r.real_width:.2f}m ({r.real_length * r.real_width:.2f} sq m)" if r.real_length else "N/A"\n',
                '        print(f"  • {r.name}: {d_str}")\n',
                "        \n",
                '    print(f"\\nDetected {len(res.doors)} Doors & {len(res.windows)} Windows.")\n',
                "    \n",
                '    vis = draw_styled_segmentation_overlay(filename, res, output_path=f"annotated_{filename}.png")\n',
                "    \n",
                "    fig, ax = plt.subplots(figsize=(10, 8))\n",
                "    ax.imshow(vis)\n",
                '    ax.axis("off")\n',
                '    ax.set_title(f"Analysis: {filename}", fontsize=14)\n',
                "    plt.show()\n",
            ],
        },
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 2,
}

out_nb = REPO_ROOT / "notebooks" / "04_yolo_segmentation_pipeline.ipynb"
with open(out_nb, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated self-contained Colab notebook at: {out_nb}")
print(f"File size: {out_nb.stat().st_size / 1024:.1f} KB")
