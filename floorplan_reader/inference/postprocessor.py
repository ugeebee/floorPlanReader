"""Post-processor for VLM floor plan generations.

Extracts JSON payloads from model text, handles markdown fencing, repairs malformed
syntax (trailing commas, quotes), deduplicates overlapping detections, and constructs
validated FloorPlanAnalysis instances with pixel-space coordinates.
"""

from __future__ import annotations
import json
import re
import logging
from typing import Dict, Any, Optional, List

from floorplan_reader.schema import FloorPlanAnalysis, BoundingBox2D, RoomElement

logger = logging.getLogger(__name__)


class FloorPlanPostprocessor:
    """Robust parser and validator for model responses."""

    def __init__(self, iou_dedup_threshold: float = 0.85):
        self.iou_dedup_threshold = iou_dedup_threshold

    def extract_json_string(self, raw_text: str) -> str:
        """Extract valid JSON substring from markdown blocks or conversational text."""
        text = raw_text.strip()

        # Check for ```json ... ``` blocks
        json_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if json_block_match:
            candidate = json_block_match.group(1).strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate

        # Check for outermost { ... }
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            return text[brace_start : brace_end + 1].strip()

        return text

    def repair_json_string(self, text: str) -> str:
        """Repair common small syntax quirks in generated JSON (trailing commas, comments)."""
        repaired = text.strip()
        # Remove trailing commas before closing braces or brackets: , } -> } and , ] -> ]
        repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
        # Fix unquoted keys if any
        repaired = re.sub(r"([{,]\s*)([a-zA-Z0-9_]+)(\s*:)", r'\1"\2"\3', repaired)
        return repaired

    def parse_to_dict(self, raw_text: str) -> Dict[str, Any]:
        """Extract and parse dictionary from raw model text output."""
        extracted = self.extract_json_string(raw_text)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            repaired = self.repair_json_string(extracted)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as err:
                logger.error(f"Failed to decode JSON from model output: {err}\nRaw text:\n{raw_text}")
                # Fallback to an empty structure rather than crashing
                return {"rooms": [], "doors": [], "windows": []}

    def deduplicate_rooms(self, rooms: List[RoomElement]) -> List[RoomElement]:
        """Filter out duplicate room detections that share significant IoU and the same class."""
        if len(rooms) <= 1:
            return rooms

        filtered: List[RoomElement] = []
        for r in rooms:
            is_dup = False
            for existing in filtered:
                if existing.name == r.name and existing.box_2d.iou(r.box_2d) > self.iou_dedup_threshold:
                    is_dup = True
                    break
            if not is_dup:
                filtered.append(r)
        return filtered

    def process_generation(
        self,
        raw_text: str,
        image_width: int,
        image_height: int,
        source_format: str = "png",
        pixels_per_meter: Optional[float] = None,
    ) -> FloorPlanAnalysis:
        """Complete pipeline: extract, repair, validate, calculate dimensions and return analysis.

        Args:
            raw_text: Model generation output string.
            image_width: Original image width.
            image_height: Original image height.
            source_format: Source file format ('jpg', 'png', 'svg').
            pixels_per_meter: Optional scaling ratio for real-world meters.

        Returns:
            Validated FloorPlanAnalysis instance.
        """
        parsed_dict = self.parse_to_dict(raw_text)
        analysis = FloorPlanAnalysis.from_model_prediction(
            raw_dict=parsed_dict,
            image_width=image_width,
            image_height=image_height,
            source_format=source_format,
            pixels_per_meter=pixels_per_meter,
        )

        # Apply deduplication on rooms
        analysis.rooms = self.deduplicate_rooms(analysis.rooms)
        analysis.metadata.total_rooms = len(analysis.rooms)

        return analysis
