"""Pydantic schemas and geometry helpers for structured floor plan analysis."""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class BoundingBox2D(BaseModel):
    """Normalized 2D Bounding Box in Qwen-VL grounding format [ymin, xmin, ymax, xmax].
    
    Coordinates are normalized to range [0, 1000] where:
    - ymin, ymax: 0 (top) to 1000 (bottom)
    - xmin, xmax: 0 (left) to 1000 (right)
    """
    ymin: float = Field(..., ge=0, le=1000, description="Top edge coordinate in range 0-1000")
    xmin: float = Field(..., ge=0, le=1000, description="Left edge coordinate in range 0-1000")
    ymax: float = Field(..., ge=0, le=1000, description="Bottom edge coordinate in range 0-1000")
    xmax: float = Field(..., ge=0, le=1000, description="Right edge coordinate in range 0-1000")

    @model_validator(mode="before")
    @classmethod
    def _preprocess_input(cls, data: Any) -> Any:
        """Allow initializing from [ymin, xmin, ymax, xmax] list or dict."""
        if isinstance(data, (list, tuple)):
            if len(data) != 4:
                raise ValueError(f"Expected 4 coordinates [ymin, xmin, ymax, xmax], got {len(data)}")
            return {
                "ymin": float(data[0]),
                "xmin": float(data[1]),
                "ymax": float(data[2]),
                "xmax": float(data[3]),
            }
        return data

    @classmethod
    def parse_from_list_or_dict(cls, data: Any) -> BoundingBox2D:
        """Convenience factory method returning a validated BoundingBox2D instance."""
        if isinstance(data, cls):
            return data
        return cls.model_validate(data)

    @model_validator(mode="after")
    def validate_bounds_order(self) -> BoundingBox2D:
        """Ensure ymin <= ymax and xmin <= xmax, automatically reordering if inverted."""
        if self.ymin > self.ymax:
            self.ymin, self.ymax = self.ymax, self.ymin
        if self.xmin > self.xmax:
            self.xmin, self.xmax = self.xmax, self.xmin
        return self

    @property
    def norm_width(self) -> float:
        """Normalized width in range [0, 1000]."""
        return max(0.0, self.xmax - self.xmin)

    @property
    def norm_height(self) -> float:
        """Normalized height in range [0, 1000]."""
        return max(0.0, self.ymax - self.ymin)

    @property
    def norm_area(self) -> float:
        """Normalized area relative to full floor plan (scale 0 to 1,000,000)."""
        return self.norm_width * self.norm_height

    def to_list(self) -> List[int]:
        """Return integer [ymin, xmin, ymax, xmax] for Qwen grounding representation."""
        return [int(round(self.ymin)), int(round(self.xmin)), int(round(self.ymax)), int(round(self.xmax))]

    def to_pixel_xyxy(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized [0, 1000] to absolute pixel coordinates (xmin, ymin, xmax, ymax)."""
        px_xmin = int(round((self.xmin / 1000.0) * img_width))
        px_ymin = int(round((self.ymin / 1000.0) * img_height))
        px_xmax = int(round((self.xmax / 1000.0) * img_width))
        px_ymax = int(round((self.ymax / 1000.0) * img_height))
        return px_xmin, px_ymin, px_xmax, px_ymax

    def to_pixel_xywh(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized [0, 1000] to absolute pixel bounding box (x, y, w, h)."""
        xmin, ymin, xmax, ymax = self.to_pixel_xyxy(img_width, img_height)
        return xmin, ymin, max(1, xmax - xmin), max(1, ymax - ymin)

    def iou(self, other: BoundingBox2D) -> float:
        """Calculate Intersection over Union (IoU) with another bounding box."""
        inter_ymin = max(self.ymin, other.ymin)
        inter_xmin = max(self.xmin, other.xmin)
        inter_ymax = min(self.ymax, other.ymax)
        inter_xmax = min(self.xmax, other.xmax)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        union_area = self.norm_area + other.norm_area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area


class RoomElement(BaseModel):
    """Represents an identified room or functional architectural zone."""
    id: str = Field(..., description="Unique room identifier (e.g. room_1)")
    name: str = Field(..., description="Room type: bedroom, living_room, kitchen, bathroom, hallway, etc.")
    box_2d: BoundingBox2D = Field(..., description="Normalized bounding box [ymin, xmin, ymax, xmax]")
    
    # Relative normalized dimension calculations (derived automatically if omitted)
    norm_length: Optional[float] = Field(None, description="Longer dimension in normalized units [0, 1000]")
    norm_width: Optional[float] = Field(None, description="Shorter dimension in normalized units [0, 1000]")
    area_percentage: Optional[float] = Field(None, description="Percentage of total floor plan area (0-100%)")
    
    # Text extracted from floor plan blueprints if visible
    detected_label_text: Optional[str] = Field(None, description="Blueprint text found inside the room (e.g. '12 x 14')")
    
    # Calibrated real-world dimensions (if scale ratio or generator metadata is provided)
    real_length: Optional[float] = Field(None, description="Estimated real-world length")
    real_width: Optional[float] = Field(None, description="Estimated real-world width")
    unit: Optional[str] = Field("m", description="Dimension unit ('m' or 'ft')")

    @model_validator(mode="after")
    def calculate_dimensions(self) -> RoomElement:
        """Automatically calculate norm_length, norm_width, and area_percentage from box_2d."""
        w = self.box_2d.norm_width
        h = self.box_2d.norm_height
        if self.norm_length is None:
            self.norm_length = round(max(w, h), 1)
        if self.norm_width is None:
            self.norm_width = round(min(w, h), 1)
        if self.area_percentage is None:
            # Full layout area is 1000 * 1000 = 1,000,000
            self.area_percentage = round((self.box_2d.norm_area / 1_000_000.0) * 100.0, 2)
        return self


class DoorElement(BaseModel):
    """Represents an identified door opening or swing symbol."""
    id: str = Field(..., description="Unique door identifier (e.g. door_1)")
    box_2d: BoundingBox2D = Field(..., description="Normalized bounding box [ymin, xmin, ymax, xmax]")
    door_type: str = Field("single_swing", description="Door classification: single_swing, double_swing, sliding, opening")
    connects: Optional[List[str]] = Field(default_factory=list, description="IDs of rooms connected by this door")


class WindowElement(BaseModel):
    """Represents an identified window element."""
    id: str = Field(..., description="Unique window identifier (e.g. window_1)")
    box_2d: BoundingBox2D = Field(..., description="Normalized bounding box [ymin, xmin, ymax, xmax]")
    window_type: str = Field("standard", description="Window classification: standard, bay, sliding")
    wall_side: Optional[str] = Field(None, description="Orientation on plan: north, south, east, west, exterior")


class FloorPlanMetadata(BaseModel):
    """Metadata regarding the input floor plan drawing."""
    image_width: int = Field(..., description="Original image width in pixels")
    image_height: int = Field(..., description="Original image height in pixels")
    source_format: str = Field("png", description="Original file format: jpg, png, or svg")
    total_rooms: int = Field(0, description="Total detected rooms")
    total_doors: int = Field(0, description="Total detected doors")
    total_windows: int = Field(0, description="Total detected windows")
    pixels_per_meter: Optional[float] = Field(None, description="Calibration scale ratio if available")


class FloorPlanAnalysis(BaseModel):
    """Top-level structured output of the floor plan reader analysis."""
    metadata: FloorPlanMetadata
    rooms: List[RoomElement] = Field(default_factory=list)
    doors: List[DoorElement] = Field(default_factory=list)
    windows: List[WindowElement] = Field(default_factory=list)

    @classmethod
    def from_model_prediction(
        cls,
        raw_dict: Dict[str, Any],
        image_width: int,
        image_height: int,
        source_format: str = "png",
        pixels_per_meter: Optional[float] = None,
    ) -> FloorPlanAnalysis:
        """Instantiate and enrich FloorPlanAnalysis from model JSON dictionary."""
        rooms_data = raw_dict.get("rooms", [])
        doors_data = raw_dict.get("doors", [])
        windows_data = raw_dict.get("windows", [])

        parsed_rooms: List[RoomElement] = []
        for idx, r in enumerate(rooms_data):
            r_id = r.get("id", f"room_{idx + 1}")
            box = r.get("box_2d") or r.get("bbox") or r.get("coordinates")
            if not box:
                continue
            name = r.get("name") or r.get("type") or r.get("label", "room")
            label_text = r.get("detected_label_text") or r.get("text")
            
            # Real world scaling if pixels_per_meter is provided
            real_l, real_w = None, None
            if pixels_per_meter and pixels_per_meter > 0:
                box_obj = BoundingBox2D.parse_from_list_or_dict(box)
                _, _, pw, ph = box_obj.to_pixel_xywh(image_width, image_height)
                real_l = round(max(pw, ph) / pixels_per_meter, 2)
                real_w = round(min(pw, ph) / pixels_per_meter, 2)
            else:
                real_l = r.get("length") or r.get("real_length")
                real_w = r.get("width") or r.get("real_width")

            parsed_rooms.append(
                RoomElement(
                    id=r_id,
                    name=name.lower().replace(" ", "_"),
                    box_2d=box,
                    detected_label_text=label_text,
                    real_length=real_l,
                    real_width=real_w,
                    unit=r.get("unit", "m"),
                )
            )

        parsed_doors: List[DoorElement] = []
        for idx, d in enumerate(doors_data):
            d_id = d.get("id", f"door_{idx + 1}")
            box = d.get("box_2d") or d.get("bbox") or d.get("coordinates")
            if not box:
                continue
            door_type = d.get("door_type") or d.get("type", "single_swing")
            connects = d.get("connects", [])
            parsed_doors.append(DoorElement(id=d_id, box_2d=box, door_type=door_type, connects=connects))

        parsed_windows: List[WindowElement] = []
        for idx, w in enumerate(windows_data):
            w_id = w.get("id", f"window_{idx + 1}")
            box = w.get("box_2d") or w.get("bbox") or w.get("coordinates")
            if not box:
                continue
            win_type = w.get("window_type") or w.get("type", "standard")
            wall = w.get("wall_side") or w.get("wall")
            parsed_windows.append(WindowElement(id=w_id, box_2d=box, window_type=win_type, wall_side=wall))

        metadata = FloorPlanMetadata(
            image_width=image_width,
            image_height=image_height,
            source_format=source_format,
            total_rooms=len(parsed_rooms),
            total_doors=len(parsed_doors),
            total_windows=len(parsed_windows),
            pixels_per_meter=pixels_per_meter,
        )

        return cls(metadata=metadata, rooms=parsed_rooms, doors=parsed_doors, windows=parsed_windows)

    def to_clean_dict(self) -> Dict[str, Any]:
        """Serialize into clean dictionary for API or downstream export."""
        return self.model_dump(exclude_none=True)
