# 📐 Floor Plan Reader: Vision-Language Model Pipeline

A complete system to analyze architectural floor plans in **JPG, PNG, or SVG** vector format using fine-tuned **Qwen3-VL 8B** (or Qwen2.5-VL 7B). Outputs structured JSON with normalized bounding locations (`[ymin, xmin, ymax, xmax]`), length, width, door/window positions, and renders color-coded visual overlays.

Designed specifically to be fine-tuned on **Google Colab's Free Tier T4 GPU (15GB VRAM)** using **4-bit QLoRA** (via Unsloth / Hugging Face TRL).

---

## 🏗️ System Architecture

```
                      ┌───────────────────────────────────────────────┐
                      │    Floor Plan Input (JPG, PNG, or SVG)        │
                      └──────────────────────┬────────────────────────┘
                                             │
                                             ▼
                      ┌───────────────────────────────────────────────┐
                      │   Converters & Preprocessor                   │
                      │   - SVG Rasterization (Cairo / svglib)        │
                      │   - Dynamic resolution bounds (512 - 896 px)  │
                      └──────────────────────┬────────────────────────┘
                                             │
                                             ▼
                      ┌───────────────────────────────────────────────┐
                      │   Qwen3-VL 8B Vision-Language Model           │
                      │   - 4-bit Quantization (NormalFloat NF4)      │
                      │   - Fine-Tuned LoRA Adapter (Rank 16, a=32)   │
                      └──────────────────────┬────────────────────────┘
                                             │
                                             ▼
                      ┌───────────────────────────────────────────────┐
                      │   Resilient Postprocessor & Pydantic Schema   │
                      │   - Markdown & JSON repair                    │
                      │   - Coordinate de-normalization               │
                      │   - Room dimension & area % calculation       │
                      └──────────────────────┬────────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
         ┌───────────────────────────┐               ┌───────────────────────────┐
         │ Structured Output (JSON)  │               │ Annotated Visual Overlay  │
         │ - Rooms (L x W, Area)     │               │ - Semi-transparent masks  │
         │ - Doors & Windows         │               │ - Door / Window markers   │
         │ - Grounding [0, 1000]     │               │ - Dimension pill tags     │
         └───────────────────────────┘               └───────────────────────────┘
```

---

## ⚡ Colab Free T4 Memory Footprint (15 GB VRAM)

| Component | Precision / Configuration | VRAM Footprint |
| :--- | :--- | :--- |
| **Base Model Weights** | 4-bit NormalFloat (NF4 via Unsloth / bitsandbytes) | ~5.2 GB |
| **LoRA Trainable Adapters** | Rank 16, Alpha 32 (Language + Attention + MLP) | ~200 MB |
| **Vision Token Budget** | Resolution dynamically capped at 768–896 px | ~2.5 GB |
| **Gradient & Activations** | Gradient Checkpointing (`use_gradient_checkpointing=True`) | ~2.0 GB |
| **Optimizer States** | `paged_adamw_8bit` (offloads spikes safely) | ~1.5 GB |
| **Batch Configuration** | `batch_size = 1`, `gradient_accumulation_steps = 8` | ~0.5 GB |
| **Total VRAM Consumption** | | **~11.9 GB / 15.0 GB (~3.1 GB buffer)** |

---

## 📋 Structured Output Schema

The model generates normalized coordinates `[ymin, xmin, ymax, xmax]` in range `[0, 1000]`. The postprocessor calculates normalized length, normalized width, area percentages, and de-normalizes to original pixel coordinates:

```json
{
  "metadata": {
    "image_width": 1600,
    "image_height": 1200,
    "source_format": "svg",
    "total_rooms": 4,
    "total_doors": 2,
    "total_windows": 3
  },
  "rooms": [
    {
      "id": "room_1",
      "name": "bedroom",
      "box_2d": {
        "ymin": 100.0,
        "xmin": 80.0,
        "ymax": 480.0,
        "xmax": 480.0
      },
      "norm_length": 400.0,
      "norm_width": 380.0,
      "area_percentage": 15.2,
      "detected_label_text": "BEDROOM 4.0m x 3.8m"
    },
    {
      "id": "room_2",
      "name": "kitchen",
      "box_2d": {
        "ymin": 100.0,
        "xmin": 500.0,
        "ymax": 450.0,
        "xmax": 900.0
      },
      "norm_length": 400.0,
      "norm_width": 350.0,
      "area_percentage": 14.0
    }
  ],
  "doors": [
    {
      "id": "door_1",
      "box_2d": {
        "ymin": 470.0,
        "xmin": 200.0,
        "ymax": 510.0,
        "xmax": 250.0
      },
      "door_type": "single_swing"
    }
  ],
  "windows": [
    {
      "id": "win_1",
      "box_2d": {
        "ymin": 90.0,
        "xmin": 200.0,
        "ymax": 105.0,
        "xmax": 360.0
      },
      "window_type": "standard",
      "wall_side": "north"
    }
  ]
}
```

---

## 🚀 Quickstart & Installation

### Local Setup
```bash
git clone https://github.com/your-username/floorPlanReader.git
cd floorPlanReader
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run Tests
```bash
pytest tests/ -v
```

---

## 📓 Google Colab Notebooks

Upload these ready-to-run notebooks directly to Google Colab:

1. **[`notebooks/01_prepare_datasets.ipynb`](file:///home/utkarsh/Desktop/floorPlanReader/notebooks/01_prepare_datasets.ipynb)**:
   - Streams CubiCasa5k from Hugging Face Hub (`phungpx/cubicasa5k-coco`).
   - Ingests your synthetic floor plans (or generates mock samples).
   - Mixes datasets into `train.jsonl` and `val.jsonl` saved to Google Drive.
2. **[`notebooks/02_train_qwen_vl_lora_colab.ipynb`](file:///home/utkarsh/Desktop/floorPlanReader/notebooks/02_train_qwen_vl_lora_colab.ipynb)**:
   - Fine-tunes **Qwen3-VL 8B** with 4-bit QLoRA using Unsloth on Colab's Free T4 GPU.
   - Saves fine-tuned LoRA checkpoints directly to Google Drive.
3. **[`notebooks/03_inference_and_evaluation.ipynb`](file:///home/utkarsh/Desktop/floorPlanReader/notebooks/03_inference_and_evaluation.ipynb)**:
   - Runs inference on any JPG, PNG, or SVG floor plan.
   - Computes Mean IoU, Precision, and Recall on validation floor plans.
   - Renders side-by-side visual overlays.

---

## 💻 CLI Usage

### 1. Prepare Dataset (CubiCasa5k + Synthetic Mixture)
```bash
# Generate mock synthetic data + mix with CubiCasa
python scripts/prepare_data.py --generate_mock_synthetic 20 --cubicasa_ratio 0.7 --output_dir ./data/processed

# Or provide your floor plan generator tool's folder
python scripts/prepare_data.py --synthetic_dir /path/to/my_generator_output --cubicasa_ratio 0.7 --output_dir ./data/processed
```

### 2. Fine-Tune Model (Locally or on GPU machine)
```bash
python scripts/train.py \
  --train_file ./data/processed/train.jsonl \
  --val_file ./data/processed/val.jsonl \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --output_dir ./output/floorplan_qwen3_vl_lora \
  --epochs 3 \
  --lr 2e-4
```

### 3. Run Inference on a Floor Plan (JPG, PNG, or SVG)
```bash
# Offline demo run (tests postprocessing and visualizer)
python scripts/run_inference.py --image path/to/blueprint.svg --mock

# Run with fine-tuned LoRA model
python scripts/run_inference.py \
  --image path/to/blueprint.svg \
  --lora_path ./output/floorplan_qwen3_vl_lora \
  --output_json analysis.json \
  --output_viz overlay.png
```

---

## 🧪 Testing

The repository contains 17 unit tests verifying:
- SVG rasterization and transparent background compositing.
- Smart VLM image resizing and divisibility padding.
- Grounding coordinate normalization and de-normalization.
- JSON markdown extraction, syntax repair, and box deduplication.
- Synthetic dataset ingestion and dataset mixing.

```bash
pytest tests/ -v
```
