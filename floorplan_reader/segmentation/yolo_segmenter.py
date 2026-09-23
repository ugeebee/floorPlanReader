"""YOLOv8/v11 instance segmentation engine for architectural floor plans.

Detects rooms (and their exact polygon boundaries), doors, and windows,
snaps boundaries to structural walls, and calculates real-world dimensions.
"""

from __future__ import annotations
import math
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import cv2
import numpy as np
from PIL import Image

from floorplan_reader.schema import (
    BoundingBox2D,
    RoomElement,
    DoorElement,
    WindowElement,
    FloorPlanMetadata,
    FloorPlanAnalysis,
)
from floorplan_reader.converters.image_preprocessor import load_and_preprocess_floorplan
from floorplan_reader.detection.contour_detector import WallContourDetector
from floorplan_reader.detection.door_window_detector import DoorWindowDetector
from floorplan_reader.detection.text_associator import TextAndDimensionAssociator
from floorplan_reader.detection.hybrid_analyzer import HybridFloorPlanAnalyzer
from floorplan_reader.segmentation.dimension_extractor import DimensionExtractor
from floorplan_reader.segmentation.dataset_exporter import YOLO_FLOORPLAN_CLASSES, ROOM_LABEL_MAP


class YoloFloorPlanSegmenter:
    """Instance segmentation analyzer powered by YOLO and wall-snapping contour geometry."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        snap_to_walls: bool = True,
        device: Optional[str] = None,
    ):
        self.weights_path = Path(weights_path) if weights_path else None
        self.confidence_threshold = confidence_threshold
        self.snap_to_walls = snap_to_walls
        self.device = device
        self.model = None

        # Auxiliary deterministic engines
        self.contour_detector = WallContourDetector()
        self.door_window_detector = DoorWindowDetector()
        self.text_associator = TextAndDimensionAssociator()
        self.dimension_extractor = DimensionExtractor()
        self.hybrid_fallback = HybridFloorPlanAnalyzer()

        # Attempt to load model weights if specified and present
        if self.weights_path and self.weights_path.exists():
            self._load_yolo_model()

    def _load_yolo_model(self) -> None:
        """Load YOLO segmentation weights using ultralytics."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights_path))
        except Exception as e:
            print(f"[YoloFloorPlanSegmenter] Warning: Could not load YOLO weights ({e}). Falling back to hybrid engine.")
            self.model = None

    def analyze(
        self,
        image_path: Union[str, Path],
        pixels_per_meter: Optional[float] = None,
    ) -> FloorPlanAnalysis:
        """Run full architectural analysis: detect rooms, dimensions, doors, and windows."""
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Floor plan file not found: {img_path}")

        # Load RGB image and determine image dimensions
        img_rgb, (img_w, img_h), fmt = load_and_preprocess_floorplan(img_path)

        # If a trained YOLO model is loaded, run YOLO segmentation inference
        if self.model is not None:
            analysis = self._run_yolo_inference(
                img_rgb=img_rgb,
                img_w=img_w,
                img_h=img_h,
                source_format=fmt,
                image_path=img_path,
                pixels_per_meter=pixels_per_meter,
            )
            # If YOLO detected rooms, ensure doors and windows are also complemented
            if len(analysis.rooms) > 0:
                if len(analysis.doors) == 0 or len(analysis.windows) == 0:
                    fallback_res = self.hybrid_fallback.analyze(image_source=img_path, pixels_per_meter=pixels_per_meter)
                    if len(analysis.doors) == 0:
                        analysis.doors = fallback_res.doors
                    if len(analysis.windows) == 0:
                        analysis.windows = fallback_res.windows
                return analysis

        # Otherwise, run hybrid contour + wall-aware CV engine
        analysis = self.hybrid_fallback.analyze(
            image_source=img_path,
            pixels_per_meter=pixels_per_meter,
        )

        # Extract all blueprint texts for global scale calibration
        all_texts: List[str] = []
        try:
            bgr = cv2.imread(str(img_path))
            if bgr is not None:
                text_elems = self.text_associator.extract_text_from_raster(bgr)
                all_texts = [e["text"] for e in text_elems if e.get("text")]
        except Exception:
            pass

        # Enrich rooms with DimensionExtractor
        rooms_list = list(analysis.rooms)
        enriched_rooms, calibrated_ppm = self.dimension_extractor.enrich_rooms_with_dimensions(
            rooms=rooms_list,
            img_w=img_w,
            img_h=img_h,
            pixels_per_meter=pixels_per_meter or analysis.metadata.pixels_per_meter,
            extra_texts=all_texts,
        )

        analysis.rooms = enriched_rooms
        if calibrated_ppm:
            analysis.metadata.pixels_per_meter = calibrated_ppm

        return analysis

    def _run_yolo_inference(
        self,
        img_rgb: np.ndarray,
        img_w: int,
        img_h: int,
        source_format: str,
        image_path: Path,
        pixels_per_meter: Optional[float],
    ) -> FloorPlanAnalysis:
        """Execute YOLO instance segmentation inference and wall-snapping."""
        results = self.model.predict(
            source=img_rgb,
            conf=self.confidence_threshold,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )

        raw_rooms: List[Dict[str, Any]] = []
        raw_doors: List[Dict[str, Any]] = []
        raw_windows: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            masks = res.masks

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                cls_name = res.names.get(cls_id, YOLO_FLOORPLAN_CLASSES[min(cls_id, len(YOLO_FLOORPLAN_CLASSES) - 1)])

                # Bounding box in xyxy pixel coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                px_xmin, px_ymin, px_xmax, px_ymax = xyxy[:4]

                # Convert to normalized [ymin, xmin, ymax, xmax] in [0, 1000]
                ymin = max(0, min(1000, int(round((px_ymin / img_h) * 1000))))
                xmin = max(0, min(1000, int(round((px_xmin / img_w) * 1000))))
                ymax = max(0, min(1000, int(round((px_ymax / img_h) * 1000))))
                xmax = max(0, min(1000, int(round((px_xmax / img_w) * 1000))))
                box_list = [ymin, xmin, ymax, xmax]

                # Extract polygon contour if mask exists
                poly_pts: Optional[List[List[float]]] = None
                if masks is not None and i < len(masks.xy):
                    poly_pts = masks.xy[i].tolist()

                if cls_name == "door" or cls_id == 7:
                    raw_doors.append({
                        "id": f"door_{len(raw_doors) + 1}",
                        "type": "single_swing",
                        "box_2d": box_list,
                    })
                elif cls_name == "window" or cls_id == 8:
                    raw_windows.append({
                        "id": f"window_{len(raw_windows) + 1}",
                        "type": "standard",
                        "box_2d": box_list,
                    })
                else:
                    px_w = max(1, px_xmax - px_xmin)
                    px_h = max(1, px_ymax - px_ymin)
                    raw_rooms.append({
                        "id": f"room_{len(raw_rooms) + 1}",
                        "name": cls_name,
                        "box_2d": box_list,
                        "pixel_bbox": (int(px_xmin), int(px_ymin), int(px_w), int(px_h)),
                        "polygon": poly_pts,
                    })

        # Run text association to attach OCR texts to rooms
        raw_rooms = self.text_associator.associate_text_with_rooms(raw_rooms, image_path, pixels_per_meter)

        # Construct FloorPlanAnalysis
        analysis = FloorPlanAnalysis.from_model_prediction(
            raw_dict={
                "rooms": raw_rooms,
                "doors": raw_doors,
                "windows": raw_windows,
            },
            image_width=img_w,
            image_height=img_h,
            source_format=source_format,
            pixels_per_meter=pixels_per_meter,
        )

        # Enrich rooms with DimensionExtractor
        rooms_list = list(analysis.rooms)
        enriched_rooms, calibrated_ppm = self.dimension_extractor.enrich_rooms_with_dimensions(
            rooms=rooms_list,
            img_w=img_w,
            img_h=img_h,
            pixels_per_meter=pixels_per_meter,
        )
        analysis.rooms = enriched_rooms
        if calibrated_ppm:
            analysis.metadata.pixels_per_meter = calibrated_ppm

        return analysis
