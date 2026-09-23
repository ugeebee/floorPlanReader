"""CubiCasa5k dataset parser for Qwen-VL multimodal instruction tuning.

Extracts rooms, doors, and windows from CubiCasa5k annotations (COCO format or
raw SVG hierarchies) and formats them into visual grounding Qwen-VL conversations.
"""

from __future__ import annotations
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image

from floorplan_reader.schema import BoundingBox2D
from floorplan_reader.converters.image_preprocessor import load_and_preprocess_floorplan

DEFAULT_INSTRUCTION = (
    "Analyze this architectural floor plan drawing. Detect all rooms, doors, and windows. "
    "Output a structured JSON object containing: "
    "1. 'rooms': list with id, name (e.g. bedroom, kitchen, living_room, bathroom), and normalized [ymin, xmin, ymax, xmax] box_2d (0-1000 scale). "
    "2. 'doors': list with id, type (e.g. single_swing, sliding), and normalized [ymin, xmin, ymax, xmax] box_2d. "
    "3. 'windows': list with id, type, and normalized [ymin, xmin, ymax, xmax] box_2d."
)

# Standard CubiCasa5k category mapping to simplified names
CUBICASA_ROOM_MAPPING = {
    "Living Room": "living_room",
    "Kitchen": "kitchen",
    "Bedroom": "bedroom",
    "Bathroom": "bathroom",
    "Entry": "hallway",
    "Hall": "hallway",
    "Corridor": "hallway",
    "Balcony": "balcony",
    "Closet": "storage",
    "Storage": "storage",
    "Office": "office",
    "Dining": "dining_room",
    "Garage": "garage",
    "Terrace": "balcony",
    "Room": "room",
}


CUBICASA_ID_MAPPING = {
    1: "bathroom",
    2: "bedroom",
    3: "door",
    4: "kitchen",
    5: "room",
    6: "stairs",
    7: "wall",
    8: "window",
}


class CubiCasaParser:
    """Parses CubiCasa5k samples into Qwen-VL multimodal chat format."""

    def __init__(self, instruction: str = DEFAULT_INSTRUCTION):
        self.instruction = instruction

    def parse_coco_sample(
        self,
        image_path_or_pil: Union[str, Path, Image.Image],
        annotations: Union[List[Dict[str, Any]], Dict[str, List[Any]]],
        img_width: int,
        img_height: int,
        sample_id: str = "cubicasa_sample",
        output_image_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Convert a COCO-style CubiCasa sample into a Qwen-VL chat record.

        Args:
            image_path_or_pil: Floor plan image.
            annotations: COCO annotations (list of dicts, or Hugging Face columnar dict of lists).
            img_width: Image width in pixels.
            img_height: Image height in pixels.
            sample_id: Unique identifier for the sample.
            output_image_dir: Optional directory to persist preprocessed image.

        Returns:
            Dictionary matching Hugging Face multimodal conversation schema.
        """
        # Save or resolve image path
        image_save_path: str
        if isinstance(image_path_or_pil, (str, Path)) and os.path.isfile(str(image_path_or_pil)):
            image_save_path = str(Path(image_path_or_pil).resolve())
        else:
            if output_image_dir is None:
                raise ValueError("output_image_dir must be specified when passing PIL Image")
            out_dir = Path(output_image_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            target_p = out_dir / f"{sample_id}.png"
            if isinstance(image_path_or_pil, Image.Image):
                image_path_or_pil.convert("RGB").save(target_p)
            image_save_path = str(target_p.resolve())

        # Normalize annotations if passed as Hugging Face columnar dict of lists
        ann_list: List[Dict[str, Any]] = []
        if isinstance(annotations, dict):
            bboxes = annotations.get("bbox", [])
            cat_ids = annotations.get("category_id", [])
            cat_names = annotations.get("category_name", [])
            for i in range(len(bboxes)):
                entry: Dict[str, Any] = {"bbox": bboxes[i]}
                if i < len(cat_ids):
                    entry["category_id"] = cat_ids[i]
                if i < len(cat_names):
                    entry["category_name"] = cat_names[i]
                ann_list.append(entry)
        elif isinstance(annotations, list):
            ann_list = annotations

        rooms: List[Dict[str, Any]] = []
        doors: List[Dict[str, Any]] = []
        windows: List[Dict[str, Any]] = []

        room_idx = 1
        door_idx = 1
        win_idx = 1

        for ann in ann_list:
            bbox = ann.get("bbox")  # COCO bbox: [x, y, width, height]
            if not bbox or len(bbox) != 4:
                continue

            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue

            # Convert to normalized [ymin, xmin, ymax, xmax] in scale [0, 1000]
            ymin = max(0, min(1000, int(round((y / img_height) * 1000))))
            xmin = max(0, min(1000, int(round((x / img_width) * 1000))))
            ymax = max(0, min(1000, int(round(((y + h) / img_height) * 1000))))
            xmax = max(0, min(1000, int(round(((x + w) / img_width) * 1000))))

            cat_id = ann.get("category_id")
            if cat_id is not None and cat_id in CUBICASA_ID_MAPPING:
                cat_raw = CUBICASA_ID_MAPPING[cat_id]
            else:
                cat_raw = ann.get("category_name") or ann.get("label") or "room"

            cat_lower = str(cat_raw).strip().lower()

            # Skip walls or background elements
            if cat_lower in ("wall", "stairs", "objects"):
                continue

            box_list = [ymin, xmin, ymax, xmax]

            if "door" in cat_lower:
                doors.append({
                    "id": f"door_{door_idx}",
                    "type": "single_swing" if "swing" in cat_lower or "single" in cat_lower else "door",
                    "box_2d": box_list,
                })
                door_idx += 1
            elif "window" in cat_lower:
                windows.append({
                    "id": f"window_{win_idx}",
                    "type": "standard",
                    "box_2d": box_list,
                })
                win_idx += 1
            else:
                room_name = CUBICASA_ROOM_MAPPING.get(cat_raw, cat_lower.replace(" ", "_"))
                rooms.append({
                    "id": f"room_{room_idx}",
                    "name": room_name,
                    "box_2d": box_list,
                })
                room_idx += 1

        target_output = {
            "rooms": rooms,
            "doors": doors,
            "windows": windows,
        }

        return self.create_conversation_entry(
            image_path=image_save_path,
            target_json=target_output,
            sample_id=sample_id,
        )

    def parse_svg_model(
        self,
        svg_path: Union[str, Path],
        raster_image_path: Union[str, Path],
        sample_id: str = "cubicasa_svg_sample",
    ) -> Dict[str, Any]:
        """Parse raw CubiCasa5k model.svg vector annotations file into bounding boxes."""
        tree = ET.parse(str(svg_path))
        root = tree.getroot()

        # Extract SVG viewBox or width/height
        viewbox = root.attrib.get("viewBox")
        if viewbox:
            parts = [float(p) for p in viewbox.replace(",", " ").split()]
            svg_w, svg_h = parts[2], parts[3]
        else:
            svg_w = float(root.attrib.get("width", 1000))
            svg_h = float(root.attrib.get("height", 1000))

        rooms = []
        doors = []
        windows = []

        # Build parent map so child polygons can inherit class from parent <g class="Space Bedroom">
        parent_map = {c: p for p in root.iter() for c in p}

        # CubiCasa5k SVGs group spaces under <g id="Rooms"> or class tags
        for elem in root.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

            # Handle polygon, polyline, or rect elements
            pts: List[List[float]] = []
            if tag in ("polygon", "polyline"):
                points_str = elem.attrib.get("points")
                if points_str:
                    pts = [
                        [float(coord) for coord in p.split(",")]
                        for p in points_str.strip().split()
                        if "," in p
                    ]
            elif tag == "rect":
                rx = float(elem.attrib.get("x", 0))
                ry = float(elem.attrib.get("y", 0))
                rw = float(elem.attrib.get("width", 0))
                rh = float(elem.attrib.get("height", 0))
                if rw > 0 and rh > 0:
                    pts = [[rx, ry], [rx + rw, ry], [rx + rw, ry + rh], [rx, ry + rh]]

            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                xmin = max(0, min(1000, int(round((min(xs) / svg_w) * 1000))))
                ymin = max(0, min(1000, int(round((min(ys) / svg_h) * 1000))))
                xmax = max(0, min(1000, int(round((max(xs) / svg_w) * 1000))))
                ymax = max(0, min(1000, int(round((max(ys) / svg_h) * 1000))))

                # Skip zero-area noise
                if (xmax - xmin) < 5 or (ymax - ymin) < 5:
                    continue

                box = [ymin, xmin, ymax, xmax]

                # Trace class name from elem up parent hierarchy
                class_name = elem.attrib.get("class", "")
                curr = elem
                while not class_name and curr in parent_map:
                    curr = parent_map[curr]
                    class_name = curr.attrib.get("class", "") or curr.attrib.get("id", "")

                label = (class_name or elem.attrib.get("id") or "room").strip().lower()

                # Filter out walls or structural bounds
                if "wall" in label or "exterior" in label or "background" in label:
                    continue

                if "door" in label:
                    doors.append({"id": f"door_{len(doors)+1}", "type": "single_swing", "box_2d": box})
                elif "window" in label:
                    windows.append({"id": f"window_{len(windows)+1}", "type": "standard", "box_2d": box})
                elif "space" in label or "room" in label or any(k.lower() in label for k in CUBICASA_ROOM_MAPPING):
                    # Clean room name (e.g. "Space Bedroom" -> "bedroom")
                    cleaned_name = label.replace("space", "").replace("room", "").strip() or "room"
                    cleaned_name = cleaned_name.split()[0] if cleaned_name.split() else "room"
                    rooms.append({"id": f"room_{len(rooms)+1}", "name": cleaned_name, "box_2d": box})

        target_output = {
            "rooms": rooms,
            "doors": doors,
            "windows": windows,
        }

        return self.create_conversation_entry(
            image_path=str(Path(raster_image_path).resolve()),
            target_json=target_output,
            sample_id=sample_id,
        )

    def parse_cubicasa_directory(
        self,
        cubicasa_root: Union[str, Path],
        max_samples: Optional[int] = None,
        output_image_dir: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """Recursively scan an extracted Zenodo CubiCasa5k directory and parse all samples."""
        root = Path(cubicasa_root)
        svg_files = sorted(list(root.rglob("model.svg")))

        samples: List[Dict[str, Any]] = []
        for svg_f in svg_files:
            if max_samples and len(samples) >= max_samples:
                break
            parent_d = svg_f.parent
            sample_name = f"{parent_d.parent.name}_{parent_d.name}"

            # Look for raster image
            raster_img = None
            for cand in ["F1_scaled.png", "F1_original.png", "color.png", "original.png"]:
                p = parent_d / cand
                if p.exists():
                    raster_img = p
                    break

            if not raster_img:
                from floorplan_reader.converters.svg_converter import convert_svg_to_png
                raster_img = parent_d / "color.png"
                if not raster_img.exists():
                    try:
                        convert_svg_to_png(svg_f, output_path=raster_img)
                    except Exception:
                        continue

            try:
                entry = self.parse_svg_model(
                    svg_path=svg_f,
                    raster_image_path=raster_img,
                    sample_id=sample_name,
                )
                parsed_json = json.loads(entry["messages"][1]["content"][0]["text"])
                if len(parsed_json["rooms"]) > 0:
                    samples.append(entry)
            except Exception as e:
                logger.debug(f"Skipping {svg_f}: {e}")

        return samples

    def create_conversation_entry(
        self,
        image_path: str,
        target_json: Dict[str, Any],
        sample_id: str,
    ) -> Dict[str, Any]:
        """Generate Hugging Face / Qwen3-VL conversation dictionary format."""
        target_json_str = json.dumps(target_json, indent=2)
        return {
            "id": sample_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": self.instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": target_json_str},
                    ],
                },
            ],
        }
