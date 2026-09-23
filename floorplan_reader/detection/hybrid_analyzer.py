"""Master Hybrid Floor Plan Analyzer combining Computer Vision and Text Association.

Provides deterministic, pixel-perfect boundary detection for rooms, doors, and windows,
producing structured Pydantic FloorPlanAnalysis outputs with zero hallucination.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List
from pathlib import Path
import json
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


class HybridFloorPlanAnalyzer:
    """End-to-end deterministic analyzer for architectural floor plans."""

    def __init__(
        self,
        wall_thresh: int = 85,
        gap_close_px: int = 45,
        min_room_area_ratio: float = 0.015,
        min_ocr_confidence: int = 25,
    ):
        """Initialize hybrid analyzer modules.

        Args:
            wall_thresh: Grayscale cutoff for wall strokes (0-255).
            gap_close_px: Morphological closing span to bridge openings in walls.
            min_room_area_ratio: Minimum area percentage for valid rooms.
            min_ocr_confidence: Minimum confidence score for OCR text.
        """
        self.contour_detector = WallContourDetector(
            wall_intensity_threshold=wall_thresh,
            gap_close_px=gap_close_px,
            min_room_area_ratio=min_room_area_ratio,
        )
        self.symbol_detector = DoorWindowDetector()
        self.text_associator = TextAndDimensionAssociator(
            min_ocr_confidence=min_ocr_confidence
        )

    def analyze(
        self,
        image_source: Union[str, Path, Image.Image, np.ndarray],
        pixels_per_meter: Optional[float] = None,
    ) -> FloorPlanAnalysis:
        """Run full deterministic floor plan analysis.

        Args:
            image_source: Filepath, PIL Image, or numpy BGR array.
            pixels_per_meter: Optional metric scale calibration (pixels / meter).

        Returns:
            Validated FloorPlanAnalysis Pydantic instance.
        """
        # 1. Load image and standardize to BGR numpy array
        if isinstance(image_source, np.ndarray):
            bgr = image_source.copy()
            orig_w, orig_h = bgr.shape[1], bgr.shape[0]
            pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        elif isinstance(image_source, Image.Image):
            pil_img = image_source.convert("RGB")
            orig_w, orig_h = pil_img.size
            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            pil_img, (orig_w, orig_h), _ = load_and_preprocess_floorplan(image_source)
            bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

        h, w = bgr.shape[:2]

        # 2. Detect physical room contours & wall mask
        raw_rooms, wall_mask, dims = self.contour_detector.detect_rooms(bgr)

        # 3. Associate text, room categories, and metric dimensions
        enriched_rooms = self.text_associator.associate_text_with_rooms(
            raw_rooms, image_source, pixels_per_meter=pixels_per_meter
        )

        # 4. Detect doors and windows along room and wall boundaries
        raw_doors, raw_windows = self.symbol_detector.detect_symbols(
            bgr, enriched_rooms, wall_mask
        )

        # 5. Format into standardized Pydantic models
        rooms: List[RoomElement] = []
        for r in enriched_rooms:
            rooms.append(
                RoomElement(
                    id=r["id"],
                    name=r.get("name", "room"),
                    box_2d=BoundingBox2D(
                        ymin=r["box_2d"][0],
                        xmin=r["box_2d"][1],
                        ymax=r["box_2d"][2],
                        xmax=r["box_2d"][3],
                    ),
                    detected_label_text=r.get("detected_label_text"),
                    norm_length=float(r.get("norm_length", 0.0)),
                    norm_width=float(r.get("norm_width", 0.0)),
                    real_length=r.get("real_length"),
                    real_width=r.get("real_width"),
                    real_area=r.get("real_area"),
                    unit=r.get("unit", "m"),
                    area_percentage=r.get("area_percentage"),
                )
            )

        doors: List[DoorElement] = []
        for d in raw_doors:
            doors.append(
                DoorElement(
                    id=d["id"],
                    door_type=d.get("type", "single_swing"),
                    box_2d=BoundingBox2D(
                        ymin=d["box_2d"][0],
                        xmin=d["box_2d"][1],
                        ymax=d["box_2d"][2],
                        xmax=d["box_2d"][3],
                    ),
                )
            )

        windows: List[WindowElement] = []
        for win in raw_windows:
            windows.append(
                WindowElement(
                    id=win["id"],
                    window_type=win.get("type", "standard"),
                    box_2d=BoundingBox2D(
                        ymin=win["box_2d"][0],
                        xmin=win["box_2d"][1],
                        ymax=win["box_2d"][2],
                        xmax=win["box_2d"][3],
                    ),
                )
            )

        meta = FloorPlanMetadata(
            image_width=w,
            image_height=h,
            source_format="png",
            total_rooms=len(rooms),
            total_doors=len(doors),
            total_windows=len(windows),
            pixels_per_meter=pixels_per_meter,
        )

        return FloorPlanAnalysis(
            metadata=meta,
            rooms=rooms,
            doors=doors,
            windows=windows,
        )
