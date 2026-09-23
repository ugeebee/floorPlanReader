"""Semantic text and dimensional measurement associator for floor plans.

Extracts text labels, room names, area measurements, and metric dimensions from
both SVG vector text elements and raster images (via OCR), associating each
label with its enclosing room polygon.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import re
import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# Known room keywords for architectural labeling
ROOM_KEYWORDS = {
    "master_bedroom": ["master bedroom", "master bed", "master suite", "m.bedroom", "primary bedroom"],
    "bedroom": ["bedroom", "bed room", "bed", "guest room", "br"],
    "living_room": ["living & dining", "living and dining", "living room", "living", "lounge", "great room", "drawing"],
    "kitchen": ["kitchen", "kit", "cooking"],
    "bathroom": ["bathroom", "bath", "washroom", "toilet", "wc", "powder room", "ensuite"],
    "dining_room": ["dining room", "dining", "din"],
    "balcony": ["balcony", "terrace", "deck", "patio", "verandah", "porch", "hallway & balcony"],
    "hallway": ["hallway", "hall", "corridor", "passage", "foyer", "entry", "lobby"],
    "office": ["office", "study", "den", "work"],
    "storage": ["storage", "store", "closet", "utility", "laundry", "pantry"],
}


class TextAndDimensionAssociator:
    """Extracts text, areas, and dimensions, mapping them to enclosing room spaces."""

    def __init__(self, min_ocr_confidence: int = 25):
        self.min_conf = min_ocr_confidence

    def extract_text_from_svg(self, svg_content: str) -> List[Dict[str, Any]]:
        """Parse text tags directly from SVG content without OCR."""
        text_elements: List[Dict[str, Any]] = []
        # Match <text x="..." y="...">content</text>
        pattern = re.compile(r'<text[^>]*x=["\']([\d.]+)["\'][^>]*y=["\']([\d.]+)["\'][^>]*>(.*?)</text>', re.DOTALL)
        for match in pattern.finditer(svg_content):
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                raw_text = re.sub(r'<[^>]+>', '', match.group(3)).strip()
                if raw_text:
                    text_elements.append({
                        "text": raw_text,
                        "center": (x, y),
                        "source": "svg",
                    })
            except Exception:
                continue
        return text_elements

    def extract_text_from_raster(self, bgr_image: np.ndarray) -> List[Dict[str, Any]]:
        """Run OCR on image to extract text words and coordinates."""
        if not HAS_TESSERACT:
            return []

        h, w = bgr_image.shape[:2]
        
        # Scale up small images for higher OCR recognition accuracy
        scale = 1.0
        if max(w, h) < 1200:
            scale = min(3.0, 1500.0 / max(w, h))
            target_w = int(round(w * scale))
            target_h = int(round(h * scale))
            ocr_img = cv2.resize(bgr_image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            ocr_img = bgr_image

        try:
            rgb = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            # Use PSM 11 (sparse text with non-uniform orientation)
            data = pytesseract.image_to_data(pil_img, config="--psm 11", output_type=pytesseract.Output.DICT)
        except Exception:
            return []

        elements: List[Dict[str, Any]] = []
        num_boxes = len(data.get("text", []))

        # Group words into lines / blocks
        current_block = []
        for i in range(num_boxes):
            text = str(data["text"][i]).strip()
            conf = float(data["conf"][i]) if "conf" in data else 0.0

            if text and conf >= self.min_conf:
                bx = float(data["left"][i]) / scale
                by = float(data["top"][i]) / scale
                bw = float(data["width"][i]) / scale
                bh = float(data["height"][i]) / scale
                cx = bx + (bw / 2.0)
                cy = by + (bh / 2.0)
                elements.append({
                    "text": text,
                    "bbox": (bx, by, bw, bh),
                    "center": (cx, cy),
                    "conf": conf,
                    "source": "ocr",
                })

        return elements

    def classify_room_name(self, text: str) -> Optional[str]:
        """Classify a text string into a standardized room category."""
        clean = text.lower().strip()
        for cat, keywords in ROOM_KEYWORDS.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', clean) or kw in clean:
                    return cat
        return None

    def extract_dimensions_and_area(self, text_list: List[str]) -> Dict[str, Any]:
        """Extract length, width, area, and scale from a list of text strings."""
        result: Dict[str, Any] = {}
        combined = " ".join(text_list)

        # 1. Match dimensions: e.g. "5.50 m x 4.50 m" or "5.50x4.50m" or "5.00m x 4.50m"
        dim_match = re.search(r'([\d.]+)\s*m?\s*[xX×]\s*([\d.]+)\s*m', combined)
        if dim_match:
            try:
                d1 = float(dim_match.group(1))
                d2 = float(dim_match.group(2))
                result["real_length"] = max(d1, d2)
                result["real_width"] = min(d1, d2)
                result["unit"] = "m"
            except ValueError:
                pass

        # 2. Match area: e.g. "24.75 m²" or "24.75 sqm" or "266.4 sq ft"
        area_match = re.search(r'([\d.]+)\s*(?:m²|sq\s*m|sqm)', combined, re.IGNORECASE)
        if area_match:
            try:
                result["real_area"] = float(area_match.group(1))
            except ValueError:
                pass

        return result

    def associate_text_with_rooms(
        self,
        rooms: List[Dict[str, Any]],
        image_or_svg: Union[np.ndarray, str, Path],
        pixels_per_meter: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Associate extracted room text and dimensions with room bounding boxes.

        Args:
            rooms: List of room dictionaries from WallContourDetector.
            image_or_svg: Source image array or SVG file/content.
            pixels_per_meter: Optional manual or estimated metric scale factor.

        Returns:
            Updated rooms list with enriched semantic labels and dimensions.
        """
        # 1. Extract text elements
        text_elements: List[Dict[str, Any]] = []
        if isinstance(image_or_svg, (str, Path)) and str(image_or_svg).lower().endswith(".svg"):
            try:
                svg_str = Path(image_or_svg).read_text(encoding="utf-8")
                text_elements = self.extract_text_from_svg(svg_str)
            except Exception:
                pass

        if not text_elements and isinstance(image_or_svg, np.ndarray):
            text_elements = self.extract_text_from_raster(image_or_svg)
        elif not text_elements and isinstance(image_or_svg, (str, Path)):
            bgr = cv2.imread(str(image_or_svg))
            if bgr is not None:
                text_elements = self.extract_text_from_raster(bgr)

        # 2. Map each text element into its enclosing room bounding box
        for room in rooms:
            rx, ry, rw, rh = room["pixel_bbox"]
            room_texts: List[str] = []

            for elem in text_elements:
                cx, cy = elem["center"]
                # Check point containment inside room box (with small boundary margin)
                if (rx - 5) <= cx <= (rx + rw + 5) and (ry - 5) <= cy <= (ry + rh + 5):
                    room_texts.append(elem["text"])

            # Classify room name from enclosed texts
            identified_name = None
            for txt in room_texts:
                cat = self.classify_room_name(txt)
                if cat:
                    identified_name = cat
                    break

            if identified_name:
                room["name"] = identified_name
                room["detected_label_text"] = " ".join(room_texts)
            else:
                # Default generic sequential naming
                room["name"] = room.get("name", "room")

            # Extract metric dimensions if written on floor plan
            dim_info = self.extract_dimensions_and_area(room_texts)
            if "real_length" in dim_info:
                room["real_length"] = dim_info["real_length"]
                room["real_width"] = dim_info["real_width"]
                room["unit"] = dim_info.get("unit", "m")
            elif pixels_per_meter and pixels_per_meter > 0:
                room["real_length"] = round(max(rw, rh) / pixels_per_meter, 2)
                room["real_width"] = round(min(rw, rh) / pixels_per_meter, 2)
                room["unit"] = "m"

            if "real_area" in dim_info:
                room["real_area"] = dim_info["real_area"]
            elif "real_length" in room and "real_width" in room:
                room["real_area"] = round(room["real_length"] * room["real_width"], 2)

        return rooms
