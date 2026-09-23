"""Dimension and geometry extractor for floor plan room segments.

Extracts real-world dimensions (length, width, area) from blueprint text annotations,
dimension lines, and automatically calibrates pixel-to-meter scale ratios across rooms.
"""

from __future__ import annotations
import math
import re
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from floorplan_reader.schema import BoundingBox2D, RoomElement


class ExtractedDimension(BaseModel):
    """Structured dimension extracted from blueprint room annotations."""
    length_m: Optional[float] = None
    width_m: Optional[float] = None
    area_m2: Optional[float] = None
    raw_text: Optional[str] = None
    unit: str = "m"
    confidence: float = 1.0


class DimensionExtractor:
    """Extracts, parses, and propagates real-world room dimensions."""

    # Regex patterns for dimension strings like "5.50m x 4.50m", "5.50 x 4.50", "12'4\" x 14'6\""
    DIMENSION_PATTERNS = [
        # Metric with unit: 5.50m x 4.50m, 5.5m X 4.5m, 5.50 x 4.50 m
        re.compile(
            r"(\d+(?:\.\d+)?)\s*(?:m|meter|mtr)?\s*(?:x|X|×|\*)\s*(\d+(?:\.\d+)?)\s*(m|meter|mtr)?",
            re.IGNORECASE,
        ),
        # Imperial: 12'4" x 14'6", 12' x 14', 12'4" * 14'6"
        re.compile(
            r"(\d+)(?:'|ft|\s*feet)?(?:\s*(\d+)(?:\"|in|\s*inch)?)?\s*(?:x|X|×|\*)\s*(\d+)(?:'|ft|\s*feet)?(?:\s*(\d+)(?:\"|in|\s*inch)?)?",
            re.IGNORECASE,
        ),
        # Area patterns: 24.75 m², 24.75 sq m, 24.75 sqm, 250 sq ft
        re.compile(
            r"(\d+(?:\.\d+)?)\s*(?:m²|m2|sq\s*m|sqm|sq\.m\.)",
            re.IGNORECASE,
        ),
    ]

    AREA_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?)\s*(?:m²|m2|sq\s*m|sqm|sq\.m\.)",
        re.IGNORECASE,
    )

    IMPERIAL_AREA_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?)\s*(?:sq\s*ft|sqft|sq\.ft\.)",
        re.IGNORECASE,
    )

    def __init__(self, default_unit: str = "m", default_pixels_per_meter: Optional[float] = None):
        self.default_unit = default_unit
        self.default_pixels_per_meter = default_pixels_per_meter

    def parse_text_for_dimensions(self, text: str, room_w_px: float = 1.0, room_h_px: float = 1.0) -> Optional[ExtractedDimension]:
        """Parse raw text string for geometric dimensions or area."""
        if not text or not text.strip():
            return None

        clean = text.strip()

        # 1. Check metric dimension pair: e.g. "5.50m x 4.50m" or "5.50 x 4.50"
        m_dim = re.search(r"(\d+(?:\.\d+)?)\s*(?:m)?\s*(?:x|X|×|\*)\s*(\d+(?:\.\d+)?)\s*(?:m)?", clean)
        if m_dim:
            val1 = float(m_dim.group(1))
            val2 = float(m_dim.group(2))
            length = max(val1, val2)
            width = min(val1, val2)
            area = round(length * width, 2)
            return ExtractedDimension(
                length_m=length,
                width_m=width,
                area_m2=area,
                raw_text=clean,
                unit="m",
                confidence=0.95,
            )

        # 2. Check imperial dimension pair: e.g. "12'4\" x 14'6\"" or "12' x 14'"
        m_imp = re.search(
            r"(\d+)(?:'|ft)?(?:\s*(\d+)(?:\"|in)?)?\s*(?:x|X|×|\*)\s*(\d+)(?:'|ft)?(?:\s*(\d+)(?:\"|in)?)?",
            clean,
        )
        if m_imp and ("'" in clean or "ft" in clean or "\"" in clean):
            f1 = float(m_imp.group(1))
            i1 = float(m_imp.group(2) or 0)
            f2 = float(m_imp.group(3))
            i2 = float(m_imp.group(4) or 0)
            d1_m = round(f1 * 0.3048 + i1 * 0.0254, 2)
            d2_m = round(f2 * 0.3048 + i2 * 0.0254, 2)
            length = max(d1_m, d2_m)
            width = min(d1_m, d2_m)
            area = round(length * width, 2)
            return ExtractedDimension(
                length_m=length,
                width_m=width,
                area_m2=area,
                raw_text=clean,
                unit="m",
                confidence=0.90,
            )

        # 3. Check explicit metric area: e.g. "24.75 m²", "24.75 m 266.4 ft", "24.75 sqm"
        m_area = re.search(r"(\d+(?:\.\d+)?)\s*(?:m²|m2|sq\s*m|sqm|sq\.m\.|m\s+\d)", clean, re.IGNORECASE)
        if not m_area:
            # Fallback for OCR "24.75 m"
            m_area = re.search(r"(\d{1,3}\.\d{1,2})\s*m\b", clean, re.IGNORECASE)

        if m_area:
            area_val = float(m_area.group(1))
            if 2.0 <= area_val <= 250.0:
                # Compute dimensions from room aspect ratio
                r = (room_w_px / room_h_px) if room_h_px > 0 else 1.0
                calc_w = math.sqrt(area_val * r)
                calc_h = math.sqrt(area_val / r)
                length = round(max(calc_w, calc_h), 2)
                width = round(min(calc_w, calc_h), 2)
                return ExtractedDimension(
                    length_m=length,
                    width_m=width,
                    area_m2=area_val,
                    raw_text=clean,
                    unit="m",
                    confidence=0.92,
                )

        return None

    def detect_global_scale_from_texts(
        self,
        text_elements: List[str],
        rooms: List[RoomElement],
        img_w: int,
        img_h: int,
    ) -> Optional[float]:
        """Detect blueprint total dimension annotations (e.g. '9.00 m (TOTAL)') to calibrate scale."""
        combined = " ".join(text_elements)

        # Look for total dimension like "9.00 m (TOTAL)" or "9.00 m"
        total_m = re.search(r"(\d+(?:\.\d+)?)\s*m\s*(?:\(TOTAL\)|\(TOT\)|TOTAL)?", combined, re.IGNORECASE)
        if total_m:
            try:
                total_val = float(total_m.group(1))
                if 4.0 <= total_val <= 30.0 and rooms:
                    # Calculate bounding envelope of all rooms
                    all_xmin = min(r.box_2d.xmin for r in rooms)
                    all_xmax = max(r.box_2d.xmax for r in rooms)
                    envelope_width_norm = max(1.0, all_xmax - all_xmin)
                    envelope_w_px = (envelope_width_norm / 1000.0) * img_w
                    calibrated_ppm = envelope_w_px / total_val
                    return round(calibrated_ppm, 2)
            except Exception:
                pass

        return None

    def calibrate_scale(
        self,
        rooms: List[RoomElement],
        img_w: int,
        img_h: int,
    ) -> Optional[float]:
        """Calibrate pixels-per-meter scale ratio from rooms with explicit dimension text."""
        scales: List[float] = []

        for r in rooms:
            if not r.real_length or not r.real_width:
                continue
            if r.real_length <= 0 or r.real_width <= 0:
                continue

            # Pixel dimensions
            xmin, ymin, pw, ph = r.box_2d.to_pixel_xywh(img_w, img_h)
            pixel_major = max(pw, ph)
            pixel_minor = min(pw, ph)

            scale_major = pixel_major / r.real_length
            scale_minor = pixel_minor / r.real_width

            if 15.0 <= scale_major <= 500.0:
                scales.append(scale_major)
            if 15.0 <= scale_minor <= 500.0:
                scales.append(scale_minor)

        if scales:
            scales.sort()
            median_scale = scales[len(scales) // 2]
            return round(median_scale, 2)

        return self.default_pixels_per_meter

    def enrich_rooms_with_dimensions(
        self,
        rooms: List[RoomElement],
        img_w: int,
        img_h: int,
        pixels_per_meter: Optional[float] = None,
        extra_texts: Optional[List[str]] = None,
    ) -> Tuple[List[RoomElement], Optional[float]]:
        """Populate real_length, real_width, and area across all rooms.
        
        1. Applies direct OCR text matches if present in detected_label_text.
        2. Detects global dimension annotations (e.g. 9.00m total).
        3. Calibrates pixels_per_meter ratio from resolved rooms.
        4. Fills in missing dimensions for all other rooms using calibrated scale.
        """
        # Step 1: Direct text extraction per room
        for r in rooms:
            if r.detected_label_text:
                _, _, pw, ph = r.box_2d.to_pixel_xywh(img_w, img_h)
                dim = self.parse_text_for_dimensions(r.detected_label_text, room_w_px=pw, room_h_px=ph)
                if dim:
                    if dim.length_m and dim.width_m:
                        r.real_length = dim.length_m
                        r.real_width = dim.width_m
                        r.unit = dim.unit

        # Step 2: Calibrate scale from resolved rooms
        active_ppm = pixels_per_meter or self.calibrate_scale(rooms, img_w, img_h)

        # Step 3: Check global texts if still no scale
        if not active_ppm and extra_texts:
            active_ppm = self.detect_global_scale_from_texts(extra_texts, rooms, img_w, img_h)

        # Step 4: Fallback calibration if typical residential apartment (~8.5m envelope)
        if not active_ppm and rooms:
            all_xmin = min(r.box_2d.xmin for r in rooms)
            all_xmax = max(r.box_2d.xmax for r in rooms)
            envelope_w_px = ((all_xmax - all_xmin) / 1000.0) * img_w
            # Standard residential footprint width ~ 8.5 meters
            active_ppm = round(envelope_w_px / 8.5, 2)

        # Step 5: Populate remaining rooms without real dimensions
        if active_ppm and active_ppm > 0:
            for r in rooms:
                xmin, ymin, pw, ph = r.box_2d.to_pixel_xywh(img_w, img_h)
                major_px = max(pw, ph)
                minor_px = min(pw, ph)

                if r.real_length is None:
                    r.real_length = round(major_px / active_ppm, 2)
                if r.real_width is None:
                    r.real_width = round(minor_px / active_ppm, 2)
                r.unit = "m"

        return rooms, active_ppm
