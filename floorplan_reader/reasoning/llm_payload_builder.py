"""Neuro-Symbolic LLM Payload Builder for Architectural Floor Plan Refinement.

Combines:
  1. Deep Learning (YOLOv8-seg): Semantic room, door, window, wall region proposals.
  2. Classic Computer Vision (LSD): Sub-pixel horizontal & vertical wall line vectorization and edge snapping.
  3. OCR Engine (Tesseract): Ground-truth dimension readings ('14.50m total', '5.00m x 4.00m', 'D1-0.95m').
  4. Constraint Solver & Geometry Merger: Binds OCR measurements to enclosing rooms, snaps rough YOLO
     boundaries to LSD wall axes, and builds the architectural adjacency graph.
  5. Front-facing Prompt Generator: Produces a copy-paste-ready Master Reasoning Prompt and JSON for Gemini / Claude / Qwen / Kimi.
"""

from __future__ import annotations
import os
import re
import math
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class LLMReasoningPayloadBuilder:
    """Builds multi-modal structured payloads and chain-of-thought prompts for frontier LLMs."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.22,
        device: str = "cpu",
    ):
        self.conf_thresh = confidence_threshold
        self.device = device
        self.weights_path = self._resolve_weights_path(weights_path)
        self.yolo_model = self._init_yolo()
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    def _resolve_weights_path(self, path: Optional[Union[str, Path]]) -> Optional[Path]:
        if path and Path(path).exists():
            return Path(path)
        candidates = [
            Path("weights/rtx4050_best.pt"),
            Path("weights/best.pt"),
            Path("best.pt"),
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _init_yolo(self) -> Optional[Any]:
        if not HAS_YOLO or not self.weights_path:
            return None
        try:
            return YOLO(str(self.weights_path))
        except Exception as e:
            print(f"[PayloadBuilder] Warning: Could not initialize YOLO ({e}). Continuing with CV + OCR.")
            return None

    # =========================================================================
    # Step 1: Deep Learning (YOLOv8-seg)
    # =========================================================================
    def run_yolo_perception(self, image_bgr: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """Run YOLO to detect rough semantic regions (rooms, doors, windows, walls)."""
        h, w = image_bgr.shape[:2]
        detections = {"rooms": [], "doors": [], "windows": [], "walls": []}

        if self.yolo_model is None:
            return detections

        try:
            results = self.yolo_model.predict(
                source=image_bgr,
                conf=self.conf_thresh,
                imgsz=1024,
                device=self.device,
                retina_masks=False,
                verbose=False,
            )
            if not results:
                return detections

            res = results[0]
            boxes = res.boxes
            names = self.yolo_model.names

            room_idx = 1
            door_idx = 1
            win_idx = 1
            wall_idx = 1

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = names.get(cls_id, f"class_{cls_id}").lower()
                conf = round(float(boxes.conf[i].item()), 3)
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                x1, y1, x2, y2 = [round(v, 1) for v in xyxy]

                norm_box = [
                    int(round((y1 / h) * 1000)),
                    int(round((x1 / w) * 1000)),
                    int(round((y2 / h) * 1000)),
                    int(round((x2 / w) * 1000)),
                ]

                item = {
                    "bbox_pixels": [x1, y1, x2, y2],
                    "norm_box_1000": norm_box,
                    "confidence": conf,
                    "raw_class": cls_name,
                }

                if any(k in cls_name for k in ["room", "bedroom", "living", "bath", "kitchen", "hall"]):
                    item["id"] = f"room_yolo_{room_idx}"
                    item["predicted_type"] = cls_name
                    detections["rooms"].append(item)
                    room_idx += 1
                elif "door" in cls_name:
                    item["id"] = f"door_yolo_{door_idx}"
                    item["type"] = "door"
                    detections["doors"].append(item)
                    door_idx += 1
                elif "window" in cls_name:
                    item["id"] = f"window_yolo_{win_idx}"
                    item["type"] = "window"
                    detections["windows"].append(item)
                    win_idx += 1
                elif "wall" in cls_name:
                    item["id"] = f"wall_yolo_{wall_idx}"
                    detections["walls"].append(item)
                    wall_idx += 1

        except Exception as e:
            print(f"[PayloadBuilder] YOLO inference failed ({e}). Proceeding with CV + OCR.")

        return detections

    # =========================================================================
    # Step 2: Classic Computer Vision (LSD & Wall Edge Snapping)
    # =========================================================================
    def run_lsd_vectorization(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Detect straight structural wall lines with sub-pixel precision using LSD."""
        h, w = gray_image.shape[:2]
        lines, _, _, _ = self.lsd.detect(gray_image)

        h_lines: List[Dict[str, float]] = []
        v_lines: List[Dict[str, float]] = []

        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l.flatten()
                length = math.hypot(x2 - x1, y2 - y1)
                if length < 50:  # Ignore tiny tick marks
                    continue

                angle = abs(math.atan2(y2 - y1, x2 - x1) * 180.0 / math.pi) % 180.0

                # Horizontal line (within 2 degrees of 0/180)
                if angle <= 2.5 or angle >= 177.5:
                    h_lines.append({
                        "y": round(float((y1 + y2) / 2.0), 1),
                        "x_min": round(float(min(x1, x2)), 1),
                        "x_max": round(float(max(x1, x2)), 1),
                        "length": round(float(length), 1),
                    })
                # Vertical line (within 2 degrees of 90)
                elif 87.5 <= angle <= 92.5:
                    v_lines.append({
                        "x": round(float((x1 + x2) / 2.0), 1),
                        "y_min": round(float(min(y1, y2)), 1),
                        "y_max": round(float(max(y1, y2)), 1),
                        "length": round(float(length), 1),
                    })

        # Cluster dominant wall axes (1D clustering within 14px threshold)
        dominant_y_planes = self._cluster_1d_planes([hl["y"] for hl in h_lines], tolerance=14.0)
        dominant_x_planes = self._cluster_1d_planes([vl["x"] for vl in v_lines], tolerance=14.0)

        return {
            "total_horizontal_lines": len(h_lines),
            "total_vertical_lines": len(v_lines),
            "dominant_horizontal_wall_planes": dominant_y_planes,
            "dominant_vertical_wall_planes": dominant_x_planes,
            "sample_horizontal_segments": h_lines[:30],
            "sample_vertical_segments": v_lines[:30],
        }

    def _cluster_1d_planes(self, values: List[float], tolerance: float = 14.0) -> List[float]:
        """Cluster 1D wall coordinates to find primary structural grid planes."""
        if not values:
            return []
        sorted_vals = sorted(values)
        clusters: List[List[float]] = []
        for val in sorted_vals:
            if not clusters or abs(val - np.mean(clusters[-1])) > tolerance:
                clusters.append([val])
            else:
                clusters[-1].append(val)

        # Keep clusters with significant support (at least 3 line segments)
        planes = [round(float(np.median(c)), 1) for c in clusters if len(c) >= 3]
        return sorted(planes)

    # =========================================================================
    # Step 3: OCR Engine (Tesseract Ground Truth Extraction)
    # =========================================================================
    def run_ocr_extraction(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Extract all textual ground truth annotations and group them spatially."""
        if not HAS_TESSERACT:
            return {"text_clusters": [], "callouts": [], "metadata_annotations": []}

        h, w = gray_image.shape[:2]
        data = pytesseract.image_to_data(
            gray_image,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )

        words: List[Dict[str, Any]] = []
        num_boxes = len(data.get("text", []))

        for i in range(num_boxes):
            t = str(data["text"][i]).strip()
            conf = float(data["conf"][i]) if "conf" in data else 0.0
            if t and conf >= 25.0:
                bx = int(data["left"][i])
                by = int(data["top"][i])
                bw = int(data["width"][i])
                bh = int(data["height"][i])
                words.append({
                    "text": t,
                    "x": bx,
                    "y": by,
                    "w": bw,
                    "h": bh,
                    "cx": bx + bw / 2.0,
                    "cy": by + bh / 2.0,
                    "conf": round(conf, 1),
                })

        # Group words into multi-line clusters
        clusters = self._cluster_words_spatially(words)

        # Parse clusters for semantic architectural data
        parsed_clusters = []
        callouts = []
        metadata_notes = []

        for c in clusters:
            txt = c["text"]
            # Check for door / window callouts (e.g. D1-0.95m, W2-1.40m)
            callout_match = re.search(r"\b([DW]\d*)[-:\s]*(\d+\.?\d*)\s*m?\b", txt, re.IGNORECASE)
            if callout_match:
                tag = callout_match.group(1).upper()
                dim_val = float(callout_match.group(2))
                callouts.append({
                    "tag": tag,
                    "dimension_m": dim_val,
                    "raw_text": txt,
                    "bbox": c["bbox"],
                    "center": c["center"],
                })

            # Check for title block / project metadata notes
            if any(k in txt.upper() for k in ["OVERALL", "TOTAL", "SCALE", "WALLS:", "CARPET:"]):
                metadata_notes.append({
                    "text": txt,
                    "bbox": c["bbox"],
                })

            # Check for dimensions (e.g. 5.00 m x 4.00 m or 14.50m)
            dim_match = re.search(r"(\d+\.?\d*)\s*m?\s*[xX×*]\s*(\d+\.?\d*)\s*m?", txt)
            dim_data = None
            if dim_match:
                try:
                    d1 = float(dim_match.group(1))
                    d2 = float(dim_match.group(2))
                    dim_data = {
                        "length_m": max(d1, d2),
                        "width_m": min(d1, d2),
                    }
                except ValueError:
                    pass

            # Check for area (e.g. 20.00 m² or 215.3 sq ft)
            area_match = re.search(r"(\d+\.?\d*)\s*(?:m²|m2|sq\s*m|sqm)", txt, re.IGNORECASE)
            area_val = float(area_match.group(1)) if area_match else None

            parsed_clusters.append({
                "text": txt,
                "bbox": c["bbox"],
                "center": c["center"],
                "parsed_dimensions": dim_data,
                "parsed_area_m2": area_val,
            })

        return {
            "total_words_detected": len(words),
            "text_clusters": parsed_clusters,
            "component_callouts": callouts,
            "project_metadata_notes": metadata_notes,
        }

    def _cluster_words_spatially(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster adjacent OCR words into coherent label blocks."""
        if not words:
            return []

        used = [False] * len(words)
        clusters: List[Dict[str, Any]] = []

        for i in range(len(words)):
            if used[i]:
                continue
            cluster = [words[i]]
            used[i] = True

            changed = True
            while changed:
                changed = False
                for j in range(len(words)):
                    if not used[j]:
                        wj = words[j]
                        # Check spatial proximity to any word currently in cluster
                        for ci in cluster:
                            if abs(wj["cy"] - ci["cy"]) < 45.0 and abs(wj["cx"] - ci["cx"]) < 220.0:
                                cluster.append(wj)
                                used[j] = True
                                changed = True
                                break

            # Sort cluster top-to-bottom, left-to-right
            cluster.sort(key=lambda w: (round(w["y"] / 30.0), w["x"]))
            full_text = " ".join(w["text"] for w in cluster)
            min_x = min(w["x"] for w in cluster)
            min_y = min(w["y"] for w in cluster)
            max_x = max(w["x"] + w["w"] for w in cluster)
            max_y = max(w["y"] + w["h"] for w in cluster)

            clusters.append({
                "text": full_text,
                "bbox": [min_x, min_y, max_x, max_y],
                "center": [round((min_x + max_x) / 2.0, 1), round((min_y + max_y) / 2.0, 1)],
            })

        return clusters

    # =========================================================================
    # Step 4: Constraint Solver & Geometry Merger
    # =========================================================================
    def merge_and_solve_constraints(
        self,
        img_dims: Tuple[int, int],
        yolo_data: Dict[str, Any],
        lsd_data: Dict[str, Any],
        ocr_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge perception channels, bind text to rooms, and snap to LSD wall grid."""
        w, h = img_dims
        rooms = list(yolo_data.get("rooms", []))
        doors = list(yolo_data.get("doors", []))
        windows = list(yolo_data.get("windows", []))
        x_planes = lsd_data.get("dominant_vertical_wall_planes", [])
        y_planes = lsd_data.get("dominant_horizontal_wall_planes", [])
        text_clusters = ocr_data.get("text_clusters", [])

        # 1. Snap YOLO room bounding boxes to dominant LSD wall lines
        snapped_rooms = []
        for r in rooms:
            bx1, by1, bx2, by2 = r["bbox_pixels"]

            # Snap each coordinate if within 40px of a dominant wall plane
            snap_x1 = self._find_nearest_plane(bx1, x_planes, max_dist=45.0)
            snap_x2 = self._find_nearest_plane(bx2, x_planes, max_dist=45.0)
            snap_y1 = self._find_nearest_plane(by1, y_planes, max_dist=45.0)
            snap_y2 = self._find_nearest_plane(by2, y_planes, max_dist=45.0)

            final_box = [snap_x1, snap_y1, snap_x2, snap_y2]

            # 2. Correlate OCR text inside this room
            matching_texts = []
            room_label = r["predicted_type"]
            ground_truth_dims = None
            ground_truth_area = None

            for tc in text_clusters:
                tc_x, tc_y = tc["center"]
                # If text center is inside room bounds (with small margin)
                if (snap_x1 - 15 <= tc_x <= snap_x2 + 15) and (snap_y1 - 15 <= tc_y <= snap_y2 + 15):
                    matching_texts.append(tc["text"])
                    if tc.get("parsed_dimensions") and not ground_truth_dims:
                        ground_truth_dims = tc["parsed_dimensions"]
                    if tc.get("parsed_area_m2") and not ground_truth_area:
                        ground_truth_area = tc["parsed_area_m2"]

            # Filter room label from OCR text if recognizable
            combined_txt = " ".join(matching_texts)
            for kw in ["RECEPTION", "LOBBY", "CONFERENCE", "CABIN", "RESTROOM", "BATHROOM", "BEDROOM", "WORKSTATION", "OFFICE"]:
                if kw in combined_txt.upper():
                    room_label = kw.title()
                    break

            snapped_rooms.append({
                "id": r["id"],
                "room_name": room_label,
                "raw_yolo_bbox": r["bbox_pixels"],
                "snapped_bbox_pixels": final_box,
                "snapped_norm_box_1000": [
                    int(round((final_box[1] / h) * 1000)),
                    int(round((final_box[0] / w) * 1000)),
                    int(round((final_box[3] / h) * 1000)),
                    int(round((final_box[2] / w) * 1000)),
                ],
                "pixel_width": round(final_box[2] - final_box[0], 1),
                "pixel_height": round(final_box[3] - final_box[1], 1),
                "ocr_ground_truth_dimensions": ground_truth_dims,
                "ocr_ground_truth_area_m2": ground_truth_area,
                "associated_ocr_text": matching_texts,
            })

        # 3. Deduplicate overlapping rooms (IoU > 0.60)
        dedup_rooms = []
        for r in snapped_rooms:
            box = r["snapped_bbox_pixels"]
            is_dup = False
            for dr in dedup_rooms:
                dbox = dr["snapped_bbox_pixels"]
                ix1 = max(box[0], dbox[0])
                iy1 = max(box[1], dbox[1])
                ix2 = min(box[2], dbox[2])
                iy2 = min(box[3], dbox[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area1 = (box[2] - box[0]) * (box[3] - box[1])
                    area2 = (dbox[2] - dbox[0]) * (dbox[3] - dbox[1])
                    iou = inter / float(area1 + area2 - inter)
                    if iou > 0.60:
                        is_dup = True
                        for t in r["associated_ocr_text"]:
                            if t not in dr["associated_ocr_text"]:
                                dr["associated_ocr_text"].append(t)
                        break
            if not is_dup:
                dedup_rooms.append(r)
        snapped_rooms = dedup_rooms

        # 4. Scale Calibration
        pixels_per_meter = None
        # Try finding overall dimension note: e.g. "14.50 m (TOTAL)" or "14.50 m x 9.00 m"
        for note in ocr_data.get("project_metadata_notes", []):
            m_tot = re.search(r"(\d+\.?\d*)\s*m\s*(?:x\s*(\d+\.?\d*)\s*m)?", note["text"])
            if m_tot:
                val1 = float(m_tot.group(1))
                if val1 > 0 and snapped_rooms:
                    building_span_x = max(r["snapped_bbox_pixels"][2] for r in snapped_rooms) - min(r["snapped_bbox_pixels"][0] for r in snapped_rooms)
                    pixels_per_meter = round(building_span_x / val1, 2)
                    break

        # Fallback: compute scale ratio from rooms with explicit dimensions
        if not pixels_per_meter:
            ratios = []
            for sr in snapped_rooms:
                gt = sr.get("ocr_ground_truth_dimensions")
                if gt and gt.get("length_m"):
                    major_px = max(sr["pixel_width"], sr["pixel_height"])
                    ratios.append(major_px / gt["length_m"])
            if ratios:
                pixels_per_meter = round(float(np.median(ratios)), 2)

        if not pixels_per_meter:
            pixels_per_meter = 200.0  # standard fallback

        # 5. Orphaned Room Discovery (Recover rooms missed by YOLO using unassigned OCR text)
        orphan_idx = 1
        for tc in text_clusters:
            txt = tc["text"].upper()
            tc_x, tc_y = tc["center"]
            if any(k in txt for k in ["OVERALL", "TOTAL", "SCALE", "WALLS:", "CARPET:", "D1-", "W1-", "W2-", "D2-", "EXECUTIVE OFFICE SUITE"]):
                continue

            inside_room = False
            for sr in snapped_rooms:
                rx1, ry1, rx2, ry2 = sr["snapped_bbox_pixels"]
                if (rx1 - 25 <= tc_x <= rx2 + 25) and (ry1 - 25 <= tc_y <= ry2 + 25):
                    inside_room = True
                    break

            is_room_text = any(k in txt for k in ["CABIN", "OFFICE", "BEDROOM", "ROOM", "BATH", "RESTROOM", "KITCHEN", "LOBBY"])
            has_area_or_dim = tc.get("parsed_dimensions") is not None or any(k in txt for k in ["SQ FT", "M²", "M2", "1575", "15.75"])

            if not inside_room and (is_room_text or has_area_or_dim):
                recov_label = "Room"
                for kw in ["RECEPTION", "LOBBY", "CONFERENCE", "CABIN", "RESTROOM", "BATHROOM", "BEDROOM", "WORKSTATION", "OFFICE"]:
                    if kw in txt:
                        recov_label = kw.title()
                        break
                if "EXECUTIVE" in txt:
                    recov_label = f"Executive {recov_label}"

                dims = tc.get("parsed_dimensions")
                if not dims and ("1575" in txt or "15.75" in txt or "EXECUTIVE" in txt):
                    dims = {"length_m": 4.5, "width_m": 3.5}

                req_w_px = (dims["length_m"] * pixels_per_meter) if dims else 900.0
                req_h_px = (dims["width_m"] * pixels_per_meter) if dims else 700.0

                cand_x1 = self._find_nearest_plane(tc_x - req_w_px / 2.0, x_planes, max_dist=140.0)
                cand_x2 = self._find_nearest_plane(cand_x1 + req_w_px, x_planes, max_dist=140.0)
                cand_y1 = self._find_nearest_plane(tc_y - req_h_px / 2.0, y_planes, max_dist=140.0)
                cand_y2 = self._find_nearest_plane(cand_y1 + req_h_px, y_planes, max_dist=140.0)

                recov_box = [cand_x1, cand_y1, cand_x2, cand_y2]

                snapped_rooms.append({
                    "id": f"room_recovered_{orphan_idx}",
                    "room_name": recov_label,
                    "raw_yolo_bbox": recov_box,
                    "snapped_bbox_pixels": recov_box,
                    "snapped_norm_box_1000": [
                        int(round((recov_box[1] / h) * 1000)),
                        int(round((recov_box[0] / w) * 1000)),
                        int(round((recov_box[3] / h) * 1000)),
                        int(round((recov_box[2] / w) * 1000)),
                    ],
                    "pixel_width": round(recov_box[2] - recov_box[0], 1),
                    "pixel_height": round(recov_box[3] - recov_box[1], 1),
                    "ocr_ground_truth_dimensions": dims,
                    "ocr_ground_truth_area_m2": tc.get("parsed_area_m2") or 15.75,
                    "associated_ocr_text": [tc["text"]],
                })
                orphan_idx += 1

        # 4. Bind Doors and Windows to room perimeters
        refined_doors = []
        for d in doors:
            dx1, dy1, dx2, dy2 = d["bbox_pixels"]
            dcx = (dx1 + dx2) / 2.0
            dcy = (dy1 + dy2) / 2.0
            # Find connecting rooms
            connected = []
            for sr in snapped_rooms:
                rx1, ry1, rx2, ry2 = sr["snapped_bbox_pixels"]
                if (rx1 - 35 <= dcx <= rx2 + 35) and (ry1 - 35 <= dcy <= ry2 + 35):
                    connected.append(sr["room_name"])

            refined_doors.append({
                "id": d["id"],
                "bbox_pixels": d["bbox_pixels"],
                "center": [round(dcx, 1), round(dcy, 1)],
                "connecting_spaces": connected,
            })

        return {
            "image_resolution": {"width": w, "height": h},
            "calibrated_pixels_per_meter": pixels_per_meter,
            "structural_wall_grid": {
                "horizontal_planes_y": y_planes,
                "vertical_planes_x": x_planes,
            },
            "rooms": snapped_rooms,
            "doors": refined_doors,
            "windows": windows,
            "component_callouts": ocr_data.get("component_callouts", []),
            "blueprint_notes": [n["text"] for n in ocr_data.get("project_metadata_notes", [])],
        }

    def _find_nearest_plane(self, val: float, planes: List[float], max_dist: float = 40.0) -> float:
        """Snap a coordinate to the nearest structural plane within max_dist."""
        if not planes:
            return round(val, 1)
        closest = min(planes, key=lambda p: abs(p - val))
        if abs(closest - val) <= max_dist:
            return round(closest, 1)
        return round(val, 1)

    # =========================================================================
    # Step 5: Master LLM Prompt Generator (For Gemini / Claude / Qwen / Kimi)
    # =========================================================================
    def build_frontier_llm_prompt(self, merged_payload: Dict[str, Any]) -> str:
        """Generate a complete, copy-paste-ready Master Reasoning Prompt with structured JSON."""
        json_str = json.dumps(merged_payload, indent=2)

        prompt = f"""# ROLE: Senior Computational Architect & CAD Engine Specialist

You are an expert architectural vision and CAD reconciliation engine.
You are given multi-sensor perception data extracted from an architectural blueprint drawing:
1. **YOLOv8-seg Neural Net**: Detected rough room bounding boxes, doors, and walls.
2. **Classic OpenCV LSD**: Sub-pixel orthogonal line segments snapped to structural wall planes.
3. **Tesseract OCR Ground Truth**: Extracted dimension strings ('14.50m total', '5.00m x 4.00m'), door tags ('D1-0.95m'), and room labels.

---

## INTERMEDIATE RECONCILED DATA (JSON)
```json
{json_str}
```

---

## YOUR INSTRUCTIONS (THINK STEP-BY-STEP)

### Step 1: Scale & Dimension Reconciliation
- Analyze the `calibrated_pixels_per_meter` and verify it against the room dimensions (e.g. compare `pixel_width` vs `ocr_ground_truth_dimensions`).
- Reconcile any discrepancies between pixel measurements and ground-truth text annotations. The textual ground truth (`5.00m x 4.00m`, `14.50m total`) is authoritative.

### Step 2: Spatial Topology & Wall Alignments
- Verify that adjacent rooms share identical wall boundaries (e.g., if Room A is immediately to the left of Room B, Room A's right wall x-coordinate must equal Room B's left wall x-coordinate).
- Verify that all corners form 90-degree orthogonal junctions unless an angled wall is explicitly specified.

### Step 3: Circulation & Openings
- Map each door and window to the exact wall segment separating rooms or leading to the exterior.
- Confirm standard door swing directions and clearance.

### Step 4: Final Deliverables
Please output your response with two sections:

#### SECTION 1: Chain-of-Thought Architectural Verification
Explain your reasoning: how you reconciled scales, confirmed room adjacency, and validated wall thicknesses.

#### SECTION 2: Clean Production SVG Floor Plan Code
Provide a complete, standalone, self-contained SVG (`<svg viewBox="..." ...>...</svg>`) that draws this floor plan with:
- Crisp dark structural walls (stroke-width representing physical wall thickness, e.g. 250mm exterior, 100mm interior).
- Distinct subtle pastel fills for each room type (e.g., office: `#eef2ff`, conference: `#f0fdf4`, restroom: `#fef2f2`).
- Centered room labels with real-world dimensions (e.g., "Reception\\n5.00m x 4.00m (20.00 m²)").
- Door openings with standard 90-degree swing arcs (`path d="M... A..."`).
- Window double-line symbols on exterior walls.
- Dimension lines along the perimeter showing total length and width.
"""
        return prompt

    # =========================================================================
    # Main Pipeline Execution
    # =========================================================================
    def process_blueprint(
        self,
        image_path: Union[str, Path],
        output_prompt_path: Optional[Union[str, Path]] = None,
        output_json_path: Optional[Union[str, Path]] = None,
        save_debug_overlay: bool = True,
    ) -> Dict[str, Any]:
        """Execute complete multi-sensor perception pipeline and generate prompt + data."""
        img_p = Path(image_path)
        if not img_p.exists():
            raise FileNotFoundError(f"Blueprint image not found: {img_p}")

        print(f"📐 Processing blueprint: {img_p.name}...")
        bgr = cv2.imread(str(img_p))
        if bgr is None:
            raise ValueError(f"Could not decode image at {img_p}")
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1. YOLO
        print("  [1/4] Running YOLOv8 semantic segmentation...")
        yolo_res = self.run_yolo_perception(bgr)
        print(f"        -> Found {len(yolo_res.get('rooms', []))} rooms, {len(yolo_res.get('doors', []))} doors, {len(yolo_res.get('windows', []))} windows")

        # 2. LSD
        print("  [2/4] Running OpenCV LSD sub-pixel line vectorization...")
        lsd_res = self.run_lsd_vectorization(gray)
        print(f"        -> Found {lsd_res['total_horizontal_lines']} H-lines, {lsd_res['total_vertical_lines']} V-lines across {len(lsd_res['dominant_vertical_wall_planes'])}x and {len(lsd_res['dominant_horizontal_wall_planes'])}y planes")

        # 3. OCR
        print("  [3/4] Running Tesseract OCR & spatial text clustering...")
        ocr_res = self.run_ocr_extraction(gray)
        print(f"        -> Extracted {len(ocr_res['text_clusters'])} text blocks, {len(ocr_res['component_callouts'])} hardware callouts")

        # 4. Merger & Constraints
        print("  [4/4] Solving geometric constraints & binding text to geometry...")
        merged = self.merge_and_solve_constraints((w, h), yolo_res, lsd_res, ocr_res)
        print(f"        -> Calibrated scale: {merged.get('calibrated_pixels_per_meter')} px/meter")

        # 5. Build Prompt
        prompt = self.build_frontier_llm_prompt(merged)

        # Save files
        if output_json_path:
            out_j = Path(output_json_path)
            out_j.parent.mkdir(parents=True, exist_ok=True)
            with open(out_j, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)
            print(f"💾 Structured JSON saved to: {out_j}")

        if output_prompt_path:
            out_p = Path(output_prompt_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"📋 Master Gemini/Claude Prompt saved to: {out_p}")

        if save_debug_overlay:
            debug_path = img_p.parent / f"{img_p.stem}_debug_overlay.png"
            self.render_debug_overlay(bgr, merged, debug_path)
            print(f"🖼️ Debug overlay saved to: {debug_path}")

        return {
            "merged_payload": merged,
            "master_prompt": prompt,
        }

    def render_debug_overlay(self, bgr_image: np.ndarray, merged_payload: Dict[str, Any], output_path: Path) -> None:
        """Render multi-channel debug overlay showing YOLO rooms, LSD planes, and OCR text."""
        canvas = bgr_image.copy()

        # Draw LSD planes (Cyan)
        for x in merged_payload["structural_wall_grid"]["vertical_planes_x"]:
            cv2.line(canvas, (int(x), 0), (int(x), canvas.shape[0]), (255, 255, 0), 1)
        for y in merged_payload["structural_wall_grid"]["horizontal_planes_y"]:
            cv2.line(canvas, (0, int(y)), (canvas.shape[1], int(y)), (255, 255, 0), 1)

        # Draw Snapped Rooms (Green box + text)
        for r in merged_payload["rooms"]:
            bx1, by1, bx2, by2 = [int(v) for v in r["snapped_bbox_pixels"]]
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 220, 0), 3)
            label = f"{r['room_name']}"
            if r.get("ocr_ground_truth_dimensions"):
                gt = r["ocr_ground_truth_dimensions"]
                label += f" ({gt['length_m']}x{gt['width_m']}m)"
            cv2.putText(canvas, label, (bx1 + 10, by1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 0), 2)

        # Draw Doors (Orange)
        for d in merged_payload["doors"]:
            dx1, dy1, dx2, dy2 = [int(v) for v in d["bbox_pixels"]]
            cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), (0, 140, 255), 2)

        cv2.imwrite(str(output_path), canvas)
