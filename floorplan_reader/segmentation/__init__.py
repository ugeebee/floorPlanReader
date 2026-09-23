"""Instance segmentation package for floor plan analysis.

Provides dataset conversion, YOLOv8/v11 segmentation models, boundary snapping,
and room dimension extraction.
"""

from floorplan_reader.segmentation.dimension_extractor import DimensionExtractor
from floorplan_reader.segmentation.dataset_exporter import YoloDatasetExporter
from floorplan_reader.segmentation.yolo_segmenter import YoloFloorPlanSegmenter

__all__ = [
    "DimensionExtractor",
    "YoloDatasetExporter",
    "YoloFloorPlanSegmenter",
]
