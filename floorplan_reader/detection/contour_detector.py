"""Computer Vision wall & room contour detector for architectural floor plans.

Extracts pixel-perfect room boundaries by segmenting structural wall lines,
filtering out thin text strokes, closing door/window wall gaps, and isolating
enclosed room spaces from the exterior background.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Union, Optional
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from floorplan_reader.converters.image_preprocessor import load_and_preprocess_floorplan


class WallContourDetector:
    """Detects physical room boundaries using computer vision wall segmentation."""

    def __init__(
        self,
        wall_intensity_threshold: int = 85,
        min_wall_span_px: int = 25,
        gap_close_px: int = 35,
        min_room_area_ratio: float = 0.015,
        max_room_area_ratio: float = 0.85,
    ):
        """Initialize detector.

        Args:
            wall_intensity_threshold: Grayscale cutoff for dark wall strokes (0-255).
            min_wall_span_px: Minimum width/height for a connected dark component to be
                treated as a wall (removes thin text characters).
            gap_close_px: Morphological closing span to bridge door/window openings in walls.
            min_room_area_ratio: Minimum area percentage (filters tiny noise/text boxes).
            max_room_area_ratio: Maximum area percentage (filters outer blueprint borders).
        """
        self.wall_thresh = wall_intensity_threshold
        self.min_wall_span = min_wall_span_px
        self.gap_close = gap_close_px
        self.min_area_ratio = min_room_area_ratio
        self.max_area_ratio = max_room_area_ratio

    def extract_wall_mask(self, bgr_image: np.ndarray) -> np.ndarray:
        """Segment structural walls and filter out text characters."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # 1. Threshold dark strokes
        _, dark_mask = cv2.threshold(gray, self.wall_thresh, 255, cv2.THRESH_BINARY_INV)

        # 2. Filter connected components: walls have long span (length >= min_wall_span)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        wall_mask = np.zeros_like(dark_mask)

        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            # Keep components that span across wall dimensions
            if max(w, h) >= self.min_wall_span:
                wall_mask[labels == i] = 255

        return wall_mask

    def detect_rooms(
        self,
        image_source: Union[str, Path, Image.Image, np.ndarray],
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, Tuple[int, int]]:
        """Extract exact room boundaries from floor plan.

        Args:
            image_source: Filepath, PIL Image, or numpy array.

        Returns:
            Tuple of (list_of_room_dicts, wall_mask, (width, height)).
        """
        # 1. Convert to OpenCV BGR numpy array
        if isinstance(image_source, np.ndarray):
            bgr = image_source.copy()
        elif isinstance(image_source, Image.Image):
            rgb = np.array(image_source.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            pil_img, _, _ = load_and_preprocess_floorplan(image_source)
            rgb = np.array(pil_img.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        h, w = bgr.shape[:2]
        total_pixels = float(w * h)

        # 2. Segment structural walls
        wall_mask = self.extract_wall_mask(bgr)

        # 3. Incorporate blue/cyan door and window symbols to help seal wall openings
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower_symbol = np.array([90, 20, 40])
        upper_symbol = np.array([135, 255, 255])
        symbol_mask = cv2.inRange(hsv, lower_symbol, upper_symbol)

        symbol_dilated = cv2.dilate(
            symbol_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
        )
        combined_walls = cv2.bitwise_or(wall_mask, symbol_dilated)

        # 4. Multi-directional 1D wall bridging to seal collinear door and window openings
        h_open = cv2.morphologyEx(combined_walls, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
        v_open = cv2.morphologyEx(combined_walls, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))

        h_close_len = max(95, self.gap_close)
        v_close_len = max(75, self.gap_close)
        h_bridged = cv2.morphologyEx(h_open, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (h_close_len, 1)))
        v_bridged = cv2.morphologyEx(v_open, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_close_len)))

        sealed = cv2.bitwise_or(h_bridged, v_bridged)
        # Fuse 90-degree corner joints
        closed_walls = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

        # 5. Invert closed wall mask to obtain room interior spaces
        rooms_mask = cv2.bitwise_not(closed_walls)

        # 6. Flood-fill from borders to eliminate exterior background
        flood = rooms_mask.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Fill four corners
        cv2.floodFill(flood, flood_mask, (0, 0), 0)
        cv2.floodFill(flood, flood_mask, (w - 1, 0), 0)
        cv2.floodFill(flood, flood_mask, (0, h - 1), 0)
        cv2.floodFill(flood, flood_mask, (w - 1, h - 1), 0)

        # Fill along outer margins (in case padding or borders isolate corners)
        for y in range(0, h, max(1, h // 20)):
            if flood[y, 0] == 255:
                cv2.floodFill(flood, flood_mask, (0, y), 0)
            if flood[y, w - 1] == 255:
                cv2.floodFill(flood, flood_mask, (w - 1, y), 0)
        for x in range(0, w, max(1, w // 20)):
            if flood[0, x] == 255:
                cv2.floodFill(flood, flood_mask, (x, 0), 0)
            if flood[h - 1, x] == 255:
                cv2.floodFill(flood, flood_mask, (x, h - 1), 0)

        # 7. Clean up interior room spaces (fill small text holes)
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_rooms = cv2.morphologyEx(flood, cv2.MORPH_OPEN, clean_kernel)

        # 8. Extract raw room contours
        contours, _ = cv2.findContours(cleaned_rooms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_rooms: List[Dict[str, Any]] = []
        if contours is not None:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                area_ratio = area / total_pixels

                # Filter out tiny noise (raw fragments >= 0.5% area)
                if 0.005 <= area_ratio <= self.max_area_ratio:
                    x, y, rw, rh = cv2.boundingRect(cnt)
                    
                    # Exclude title/legend box at bottom right
                    if y + rh > h * 0.82 and x + rw > w * 0.60 and rh < h * 0.18:
                        continue

                    raw_rooms.append({
                        "pixel_bbox": (x, y, rw, rh),
                        "contour": cnt.tolist(),
                        "area": area,
                    })

        # 9. Merge adjacent sub-fragments separated only by interior text (where no wall exists)
        rooms: List[Dict[str, Any]] = []
        used = set()
        for i in range(len(raw_rooms)):
            if i in used:
                continue
            r1 = raw_rooms[i]
            x1, y1, w1, h1 = r1["pixel_bbox"]
            cb = [x1, y1, x1 + w1, y1 + h1]
            c_area = r1["area"]
            
            for j in range(i + 1, len(raw_rooms)):
                if j in used:
                    continue
                r2 = raw_rooms[j]
                x2, y2, w2, h2 = r2["pixel_bbox"]
                
                # Check vertical stacking
                x_ov = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                min_w = min(w1, w2)
                if min_w > 0 and (x_ov / min_w) > 0.75:
                    y_top = min(y1 + h1, y2 + h2)
                    y_bot = max(y1, y2)
                    if 0 <= (y_bot - y_top) < 30:
                        xs = max(x1, x2)
                        xe = min(x1 + w1, x2 + w2)
                        wall_count = np.sum(wall_mask[y_top:y_bot, xs:xe] > 0)
                        if wall_count < 20:  # No structural wall between fragments
                            cb[0] = min(cb[0], x2)
                            cb[1] = min(cb[1], y2)
                            cb[2] = max(cb[2], x2 + w2)
                            cb[3] = max(cb[3], y2 + h2)
                            c_area += r2["area"]
                            used.add(j)

            used.add(i)
            area_pct = round((c_area / total_pixels) * 100, 2)
            if (c_area / total_pixels) < self.min_area_ratio:
                continue

            mx = cb[0]
            my = cb[1]
            mw = cb[2] - mx
            mh = cb[3] - my
            
            ymin = max(0, min(1000, int(round((my / h) * 1000))))
            xmin = max(0, min(1000, int(round((mx / w) * 1000))))
            ymax = max(0, min(1000, int(round(((my + mh) / h) * 1000))))
            xmax = max(0, min(1000, int(round(((mx + mw) / w) * 1000))))

            rooms.append({
                "id": f"room_{len(rooms) + 1}",
                "name": "room",
                "pixel_bbox": (mx, my, mw, mh),
                "box_2d": [ymin, xmin, ymax, xmax],
                "contour": r1["contour"],
                "norm_length": max(xmax - xmin, ymax - ymin),
                "norm_width": min(xmax - xmin, ymax - ymin),
                "area_percentage": area_pct,
            })

        # Fallback: if flood fill erased rooms due to a wide unsealed opening,
        # use wall grid partition or contour bounding of the largest inner region
        if len(rooms) == 0:
            rooms = self._fallback_wall_partition(wall_mask, (w, h))

        # Sort rooms from top-left to bottom-right
        rooms.sort(key=lambda r: (r["box_2d"][0] // 100, r["box_2d"][1]))
        for idx, r in enumerate(rooms):
            r["id"] = f"room_{idx + 1}"

        return rooms, wall_mask, (w, h)

    def _fallback_wall_partition(
        self, wall_mask: np.ndarray, dimensions: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Fallback wall bounding box detection if morphological closing had leaks."""
        w, h = dimensions
        y_indices, x_indices = np.where(wall_mask > 0)
        if len(x_indices) == 0:
            return []
        
        xmin_px, xmax_px = np.min(x_indices), np.max(x_indices)
        ymin_px, ymax_px = np.min(y_indices), np.max(y_indices)
        
        rw = xmax_px - xmin_px
        rh = ymax_px - ymin_px
        area_ratio = (rw * rh) / float(w * h)

        ymin = max(0, min(1000, int(round((ymin_px / h) * 1000))))
        xmin = max(0, min(1000, int(round((xmin_px / w) * 1000))))
        ymax = max(0, min(1000, int(round((ymax_px / h) * 1000))))
        xmax = max(0, min(1000, int(round((xmax_px / w) * 1000))))

        return [{
            "id": "room_1",
            "name": "room",
            "pixel_bbox": (int(xmin_px), int(ymin_px), int(rw), int(rh)),
            "box_2d": [ymin, xmin, ymax, xmax],
            "contour": [],
            "norm_length": max(xmax - xmin, ymax - ymin),
            "norm_width": min(xmax - xmin, ymax - ymin),
            "area_percentage": round(area_ratio * 100, 2),
        }]
