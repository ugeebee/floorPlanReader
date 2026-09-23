"""YOLO segmentation dataset exporter for floor plan datasets.

Converts CubiCasa5k annotations, synthetic floor plans, and hybrid detections
into standardized YOLOv8/v11 segmentation polygon formats with data.yaml.
"""

from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image

# Standard 9-class floor plan segmentation hierarchy
YOLO_FLOORPLAN_CLASSES = [
    "bedroom",       # 0
    "living_room",   # 1
    "kitchen",       # 2
    "bathroom",      # 3
    "hallway",       # 4
    "balcony",       # 5
    "room",          # 6 (office, dining, generic)
    "door",          # 7
    "window",        # 8
]

CLASS_NAME_TO_ID = {name: i for i, name in enumerate(YOLO_FLOORPLAN_CLASSES)}

# Normalization mapping from various naming conventions
ROOM_LABEL_MAP = {
    "bedroom": 0,
    "master_bedroom": 0,
    "guest_bedroom": 0,
    "living_room": 1,
    "living": 1,
    "living_and_dining": 1,
    "lounge": 1,
    "kitchen": 2,
    "kitchenette": 2,
    "bathroom": 3,
    "bath": 3,
    "wc": 3,
    "toilet": 3,
    "powder_room": 3,
    "hallway": 4,
    "hall": 4,
    "corridor": 4,
    "entry": 4,
    "balcony": 5,
    "terrace": 5,
    "patio": 5,
    "room": 6,
    "office": 6,
    "dining": 6,
    "dining_room": 6,
    "study": 6,
    "storage": 6,
    "closet": 6,
    "door": 7,
    "single_swing": 7,
    "double_swing": 7,
    "sliding_door": 7,
    "window": 8,
    "standard_window": 8,
    "sliding_window": 8,
}


class YoloDatasetExporter:
    """Exports floor plan annotations into YOLO instance segmentation format."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.classes = YOLO_FLOORPLAN_CLASSES

    def initialize_directories(self) -> Dict[str, Path]:
        """Create directory structure for train and val splits."""
        dirs = {
            "train_images": self.output_dir / "images" / "train",
            "val_images": self.output_dir / "images" / "val",
            "train_labels": self.output_dir / "labels" / "train",
            "val_labels": self.output_dir / "labels" / "val",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    def write_data_yaml(self) -> Path:
        """Write the standard YOLO data.yaml configuration file."""
        yaml_path = self.output_dir / "data.yaml"
        lines = [
            f"path: {self.output_dir.resolve()}",
            "train: images/train",
            "val: images/val",
            "",
            "names:",
        ]
        for idx, name in enumerate(self.classes):
            lines.append(f"  {idx}: {name}")
        lines.append("")

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return yaml_path

    @staticmethod
    def box_to_normalized_polygon(box_0_1000: Union[List[float], Dict[str, float]]) -> List[float]:
        """Convert normalized [ymin, xmin, ymax, xmax] (0-1000) to 4-corner polygon (0.0 - 1.0).
        
        Returns [x1, y1, x2, y2, x3, y3, x4, y4] normalized to [0, 1].
        """
        if isinstance(box_0_1000, dict):
            ymin = float(box_0_1000.get("ymin", box_0_1000.get("top", 0))) / 1000.0
            xmin = float(box_0_1000.get("xmin", box_0_1000.get("left", 0))) / 1000.0
            ymax = float(box_0_1000.get("ymax", box_0_1000.get("bottom", 1000))) / 1000.0
            xmax = float(box_0_1000.get("xmax", box_0_1000.get("right", 1000))) / 1000.0
        else:
            ymin, xmin, ymax, xmax = [float(c) / 1000.0 for c in box_0_1000]

        # Clockwise polygon: top-left, top-right, bottom-right, bottom-left
        return [
            round(xmin, 6), round(ymin, 6),
            round(xmax, 6), round(ymin, 6),
            round(xmax, 6), round(ymax, 6),
            round(xmin, 6), round(ymax, 6),
        ]

    def add_sample(
        self,
        image_path: Union[str, Path],
        annotations: Dict[str, Any],
        split: str = "train",
        sample_id: Optional[str] = None,
    ) -> bool:
        """Export a single floor plan image and its annotations.
        
        Args:
            image_path: Path to the image (PNG/JPG).
            annotations: Dict with 'rooms', 'doors', 'windows'.
            split: 'train' or 'val'.
            sample_id: Optional unique filename stem.
        """
        img_p = Path(image_path)
        if not img_p.exists():
            return False

        stem = sample_id or img_p.stem
        target_img_dir = self.output_dir / "images" / split
        target_lbl_dir = self.output_dir / "labels" / split
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lbl_dir.mkdir(parents=True, exist_ok=True)

        target_img = target_img_dir / f"{stem}.png"
        target_lbl = target_lbl_dir / f"{stem}.txt"

        # Copy or convert image
        try:
            with Image.open(img_p) as pil_img:
                pil_img.convert("RGB").save(target_img)
        except Exception:
            shutil.copy2(img_p, target_img)

        # Build YOLO segmentation annotation lines
        label_lines: List[str] = []

        # 1. Rooms (Classes 0 - 6)
        for room in annotations.get("rooms", []):
            name = str(room.get("name") or room.get("type") or "room").lower().replace(" ", "_")
            class_id = ROOM_LABEL_MAP.get(name, 6)

            # Check if explicit polygon points exist
            polygon = room.get("polygon")
            if polygon and len(polygon) >= 6:
                # Polygon given as list of [x, y] or flat floats
                coords: List[float] = []
                if isinstance(polygon[0], (list, tuple)):
                    for pt in polygon:
                        coords.extend([pt[0] / 1000.0 if pt[0] > 1.0 else pt[0], pt[1] / 1000.0 if pt[1] > 1.0 else pt[1]])
                else:
                    coords = [c / 1000.0 if c > 1.0 else c for c in polygon]
                poly_str = " ".join(f"{c:.6f}" for c in coords)
                label_lines.append(f"{class_id} {poly_str}")
            else:
                box = room.get("box_2d") or room.get("bbox")
                if box and len(box) == 4:
                    poly_coords = self.box_to_normalized_polygon(box)
                    poly_str = " ".join(f"{c:.6f}" for c in poly_coords)
                    label_lines.append(f"{class_id} {poly_str}")

        # 2. Doors (Class 7)
        for door in annotations.get("doors", []):
            box = door.get("box_2d") or door.get("bbox")
            if box and len(box) == 4:
                poly_coords = self.box_to_normalized_polygon(box)
                poly_str = " ".join(f"{c:.6f}" for c in poly_coords)
                label_lines.append(f"7 {poly_str}")

        # 3. Windows (Class 8)
        for win in annotations.get("windows", []):
            box = win.get("box_2d") or win.get("bbox")
            if box and len(box) == 4:
                poly_coords = self.box_to_normalized_polygon(box)
                poly_str = " ".join(f"{c:.6f}" for c in poly_coords)
                label_lines.append(f"8 {poly_str}")

        # Write label file
        with open(target_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + ("\n" if label_lines else ""))

        return True
