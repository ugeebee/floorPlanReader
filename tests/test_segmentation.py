"""Unit and integration tests for segmentation, dimension extraction, and YOLO pipeline."""

import pytest
import shutil
from pathlib import Path
from PIL import Image

from floorplan_reader.schema import BoundingBox2D, RoomElement
from floorplan_reader.segmentation.dimension_extractor import DimensionExtractor
from floorplan_reader.segmentation.dataset_exporter import YoloDatasetExporter
from floorplan_reader.segmentation.yolo_segmenter import YoloFloorPlanSegmenter
from floorplan_reader.visualization.segmentation_visualizer import draw_styled_segmentation_overlay


SAMPLE_MS_PATH = Path("data/test_samples/sample_master_suite.png")
SAMPLE_1BHK_PATH = Path("data/test_samples/sample_1bhk.png")


def test_dimension_extractor_metric_and_imperial():
    """Verify regex parsing of metric and imperial dimension strings."""
    extractor = DimensionExtractor()

    # Metric pair
    res_m = extractor.parse_text_for_dimensions("Master Bedroom 5.50m x 4.50m")
    assert res_m is not None
    assert res_m.length_m == 5.50
    assert res_m.width_m == 4.50
    assert res_m.area_m2 == 24.75
    assert res_m.unit == "m"

    # Imperial pair
    res_imp = extractor.parse_text_for_dimensions("12' x 14'")
    assert res_imp is not None
    assert res_imp.length_m is not None
    assert res_imp.width_m is not None
    assert res_imp.area_m2 > 10.0


def test_dimension_extractor_area_derivation():
    """Verify area parsing and dimension derivation via aspect ratio."""
    extractor = DimensionExtractor()

    # Room is 558px wide by 496px high with area 24.75 m²
    res_area = extractor.parse_text_for_dimensions("Suite 24.75 m²", room_w_px=558, room_h_px=496)
    assert res_area is not None
    assert res_area.area_m2 == 24.75
    assert 5.2 <= res_area.length_m <= 5.6
    assert 4.3 <= res_area.width_m <= 4.7


def test_dimension_extractor_global_scale():
    """Verify global dimension calibration from annotations like '9.00 m (TOTAL)'."""
    extractor = DimensionExtractor()
    rooms = [
        RoomElement(id="r1", name="living", box_2d=BoundingBox2D(ymin=200, xmin=170, ymax=600, xmax=500)),
        RoomElement(id="r2", name="bedroom", box_2d=BoundingBox2D(ymin=200, xmin=500, ymax=600, xmax=839)),
    ]
    # Total width is 839 - 170 = 669 norm units. At 1000px width, 669px.
    # 9.00m gives ppm = 669 / 9 = 74.33
    ppm = extractor.detect_global_scale_from_texts(["Original Plan", "9.00 m (TOTAL)"], rooms, 1000, 1000)
    assert ppm is not None
    assert 70.0 <= ppm <= 80.0


def test_yolo_dataset_exporter(tmp_path):
    """Verify YOLO dataset export structure, yaml, and label formats."""
    exp_dir = tmp_path / "test_yolo_export"
    exporter = YoloDatasetExporter(exp_dir)
    exporter.initialize_directories()
    yaml_file = exporter.write_data_yaml()

    assert yaml_file.exists()
    yaml_text = yaml_file.read_text()
    assert "0: bedroom" in yaml_text
    assert "7: door" in yaml_text
    assert "8: window" in yaml_text

    # Test adding a sample
    mock_annotations = {
        "rooms": [{"name": "bedroom", "box_2d": [200, 200, 600, 600]}],
        "doors": [{"box_2d": [180, 200, 220, 250]}],
        "windows": [{"box_2d": [590, 300, 610, 400]}],
    }

    success = exporter.add_sample(SAMPLE_MS_PATH, mock_annotations, split="train", sample_id="sample_test")
    assert success is True

    lbl_file = exp_dir / "labels" / "train" / "sample_test.txt"
    assert lbl_file.exists()
    lines = lbl_file.read_text().strip().split("\n")
    assert len(lines) == 3  # 1 room + 1 door + 1 window


def test_yolo_segmenter_primary_and_secondary_goals():
    """Verify YoloFloorPlanSegmenter fulfills primary (rooms + dimensions) and secondary (doors/windows) goals."""
    segmenter = YoloFloorPlanSegmenter()

    # 1. Test on Master Suite
    analysis_ms = segmenter.analyze(SAMPLE_MS_PATH)
    assert len(analysis_ms.rooms) >= 1
    # Check primary goal: rooms & dimensions
    ms_room = analysis_ms.rooms[0]
    assert ms_room.real_length is not None and ms_room.real_length > 0
    assert ms_room.real_width is not None and ms_room.real_width > 0
    # Expected master suite is roughly 5.5m x 4.5m
    assert 5.0 <= ms_room.real_length <= 6.0
    assert 4.0 <= ms_room.real_width <= 5.0
    # Check secondary goal: doors & windows
    assert len(analysis_ms.doors) >= 1
    assert len(analysis_ms.windows) >= 1

    # 2. Test on 1BHK Multi-Room
    analysis_1bhk = segmenter.analyze(SAMPLE_1BHK_PATH)
    # Check primary goal: 5 rooms detected
    assert len(analysis_1bhk.rooms) == 5
    for r in analysis_1bhk.rooms:
        assert r.real_length is not None and r.real_length > 0
        assert r.real_width is not None and r.real_width > 0
    # Check secondary goal: doors & windows
    assert len(analysis_1bhk.doors) >= 1
    assert len(analysis_1bhk.windows) >= 4


def test_segmentation_visualizer(tmp_path):
    """Verify draw_styled_segmentation_overlay renders cleanly."""
    segmenter = YoloFloorPlanSegmenter()
    analysis = segmenter.analyze(SAMPLE_MS_PATH)

    out_png = tmp_path / "test_overlay.png"
    img = draw_styled_segmentation_overlay(SAMPLE_MS_PATH, analysis, output_path=out_png)

    assert out_png.exists()
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0 and img.size[1] > 0
