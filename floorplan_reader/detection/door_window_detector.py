"""Detector for door openings and window symbols along architectural wall boundaries.

Combines color symbol extraction (for modern/CAD blue/cyan symbols) and geometric
wall gap analysis (for B&W and architectural blueprints) to identify doors and windows.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class DoorWindowDetector:
    """Detects doors and windows along the perimeter boundaries of rooms."""

    def __init__(
        self,
        min_symbol_area: int = 15,
        wall_search_margin: int = 15,
    ):
        self.min_area = min_symbol_area
        self.margin = wall_search_margin

    def detect_symbols(
        self,
        bgr_image: np.ndarray,
        rooms: List[Dict[str, Any]],
        wall_mask: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect doors and windows along the perimeter boundaries of rooms.

        Args:
            bgr_image: Original floor plan image in BGR.
            rooms: Detected room structures from WallContourDetector.
            wall_mask: Binary mask of detected dark structural walls.

        Returns:
            Tuple of (doors_list, windows_list).
        """
        h, w = bgr_image.shape[:2]
        doors: List[Dict[str, Any]] = []
        windows: List[Dict[str, Any]] = []

        # 1. Color-based detection for CAD blue/cyan door arcs and window panes
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # Bright blue / cyan architectural symbols (V > 100 avoids dark navy walls)
        lower_blue = np.array([90, 30, 95])
        upper_blue = np.array([135, 255, 255])
        symbol_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Merge adjacent thin strokes (e.g. window triple lines or dashed door arc)
        dilated = cv2.dilate(
            symbol_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1
        )
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        door_idx = 1
        win_idx = 1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            
            # Exclude blueprint border zones:
            # 1. Legend / title box at bottom right
            if by + bh > h * 0.80 and bx + bw > w * 0.60:
                continue
            # 2. Scale bar at bottom left
            if by + bh > h * 0.85 and bx < w * 0.35:
                continue
            # 3. North arrow at top right
            if by < h * 0.20 and bx > w * 0.80:
                continue

            # Verify proximity to structural walls (must touch or lie within 12px of a wall)
            sub_y1 = max(0, by - 12)
            sub_y2 = min(h, by + bh + 12)
            sub_x1 = max(0, bx - 12)
            sub_x2 = min(w, bx + bw + 12)
            if np.sum(wall_mask[sub_y1:sub_y2, sub_x1:sub_x2] > 0) < 15:
                continue  # Not on or near a structural wall (dimension ticks or isolated badges)

            # A door or window cannot span across the entire building
            max_elem_span = max(w, h) * 0.25
            if max(bw, bh) > max_elem_span:
                continue

            # Convert to [ymin, xmin, ymax, xmax] in 0-1000 scale
            ymin = max(0, min(1000, int(round((by / h) * 1000))))
            xmin = max(0, min(1000, int(round((bx / w) * 1000))))
            ymax = max(0, min(1000, int(round(((by + bh) / h) * 1000))))
            xmax = max(0, min(1000, int(round(((bx + bw) / w) * 1000))))

            aspect_ratio = max(bw, bh) / max(1, min(bw, bh))

            # Distinguish doors vs windows:
            # Windows are long rectangular blocks along the wall (aspect ratio >= 2.0 and thickness <= 20px)
            # Door swings are square or arc-shaped (min dimension >= 14px, area >= 40px)
            if aspect_ratio >= 2.0 and max(bw, bh) >= 20 and min(bw, bh) <= 22:
                windows.append({
                    "id": f"win_{win_idx}",
                    "type": "standard",
                    "box_2d": [ymin, xmin, ymax, xmax],
                    "pixel_bbox": (bx, by, bw, bh),
                    "norm_length": max(xmax - xmin, ymax - ymin),
                    "norm_width": min(xmax - xmin, ymax - ymin),
                })
                win_idx += 1
            elif (area >= 35 and min(bw, bh) >= 10 and max(bw, bh) <= 120):
                doors.append({
                    "id": f"door_{door_idx}",
                    "type": "single_swing",
                    "box_2d": [ymin, xmin, ymax, xmax],
                    "pixel_bbox": (bx, by, bw, bh),
                    "norm_length": max(xmax - xmin, ymax - ymin),
                    "norm_width": min(xmax - xmin, ymax - ymin),
                })
                door_idx += 1

        # 2. Geometric wall gap detection fallback:
        # If no colored symbols found (e.g. grayscale blueprint or black & white CubiCasa),
        # scan room perimeters for wall gaps
        if len(doors) == 0 and len(windows) == 0 and len(rooms) > 0:
            bw_doors, bw_windows = self._detect_wall_gaps(rooms, wall_mask, (w, h))
            doors.extend(bw_doors)
            windows.extend(bw_windows)

        return doors, windows

    def _detect_wall_gaps(
        self, rooms: List[Dict[str, Any]], wall_mask: np.ndarray, dimensions: Tuple[int, int]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect wall openings along room perimeters in black-and-white drawings."""
        w, h = dimensions
        doors: List[Dict[str, Any]] = []
        windows: List[Dict[str, Any]] = []

        d_count = 1
        w_count = 1

        for r in rooms:
            rx, ry, rw, rh = r["pixel_bbox"]
            # Scan top and bottom edges for gaps
            for edge_y in [ry, ry + rh]:
                if 0 <= edge_y < h:
                    strip = wall_mask[max(0, edge_y - 3):min(h, edge_y + 4), rx:rx + rw]
                    col_has_wall = np.any(strip > 0, axis=0)
                    gaps = self._find_gaps_in_strip(col_has_wall, rx, edge_y, is_horizontal=True)
                    for gx, gy, gw, gh in gaps:
                        ymin = max(0, min(1000, int(round((gy / h) * 1000))))
                        xmin = max(0, min(1000, int(round((gx / w) * 1000))))
                        ymax = max(0, min(1000, int(round(((gy + gh) / h) * 1000))))
                        xmax = max(0, min(1000, int(round(((gx + gw) / w) * 1000))))
                        if gw >= 40:
                            windows.append({
                                "id": f"win_{w_count}",
                                "type": "standard",
                                "box_2d": [ymin, xmin, ymax, xmax],
                                "pixel_bbox": (gx, gy, gw, gh),
                            })
                            w_count += 1
                        else:
                            doors.append({
                                "id": f"door_{d_count}",
                                "type": "single_swing",
                                "box_2d": [ymin, xmin, ymax, xmax],
                                "pixel_bbox": (gx, gy, gw, gh),
                            })
                            d_count += 1

        return doors, windows

    def _find_gaps_in_strip(
        self, has_wall: np.ndarray, offset_x: int, offset_y: int, is_horizontal: bool
    ) -> List[Tuple[int, int, int, int]]:
        """Find opening gaps of length between 15px and 120px."""
        gaps: List[Tuple[int, int, int, int]] = []
        diffs = np.diff(np.pad(has_wall.astype(int), (1, 1), 'constant'))
        starts = np.where(diffs == -1)[0]
        ends = np.where(diffs == 1)[0]
        for s, e in zip(starts, ends):
            length = e - s
            if 15 <= length <= 120:
                if is_horizontal:
                    gaps.append((offset_x + s, offset_y - 5, length, 10))
                else:
                    gaps.append((offset_x - 5, offset_y + s, 10, length))
        return gaps
