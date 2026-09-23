"""Unit tests for Hybrid Computer Vision floor plan detection modules."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from floorplan_reader.detection.contour_detector import WallContourDetector
from floorplan_reader.detection.door_window_detector import DoorWindowDetector
from floorplan_reader.detection.text_associator import TextAndDimensionAssociator
from floorplan_reader.detection.hybrid_analyzer import HybridFloorPlanAnalyzer
from floorplan_reader.schema import FloorPlanAnalysis


@pytest.fixture
def master_suite_path():
    path = Path("data/test_samples/sample_master_suite.png")
    if not path.exists():
        pytest.skip(f"Test sample not found: {path}")
    return path


@pytest.fixture
def sample_1bhk_path():
    path = Path("data/test_samples/sample_1bhk.png")
    if not path.exists():
        pytest.skip(f"Test sample not found: {path}")
    return path


def test_wall_contour_detector_single_room(master_suite_path):
    detector = WallContourDetector()
    rooms, wall_mask, dims = detector.detect_rooms(master_suite_path)
    
    assert len(rooms) >= 1
    r = rooms[0]
    assert "box_2d" in r
    assert len(r["box_2d"]) == 4
    for coord in r["box_2d"]:
        assert 0 <= coord <= 1000
    assert r["area_percentage"] > 15.0


def test_wall_contour_detector_multi_room(sample_1bhk_path):
    detector = WallContourDetector()
    rooms, wall_mask, dims = detector.detect_rooms(sample_1bhk_path)
    
    # Assert all 5 rooms of 1-BHK residence are extracted
    assert len(rooms) == 5
    for r in rooms:
        assert len(r["box_2d"]) == 4
        for coord in r["box_2d"]:
            assert 0 <= coord <= 1000
        assert r["area_percentage"] > 1.5


def test_door_window_detector(master_suite_path):
    detector = WallContourDetector()
    rooms, wall_mask, dims = detector.detect_rooms(master_suite_path)
    
    dw_detector = DoorWindowDetector()
    bgr = cv2.imread(str(master_suite_path))
    doors, windows = dw_detector.detect_symbols(bgr, rooms, wall_mask)
    
    assert len(windows) >= 1
    for win in windows:
        assert "box_2d" in win
        assert len(win["box_2d"]) == 4


def test_text_associator_classification():
    associator = TextAndDimensionAssociator()
    assert associator.classify_room_name("Master Bedroom Suite") == "master_bedroom"
    assert associator.classify_room_name("Living & Dining Area") == "living_room"
    assert associator.classify_room_name("Kitchen") == "kitchen"
    assert associator.classify_room_name("Powder Room / Bath") == "bathroom"
    assert associator.classify_room_name("Hallway & Balcony") == "balcony"


def test_text_associator_dimensions():
    associator = TextAndDimensionAssociator()
    dims = associator.extract_dimensions_and_area(["DIMENSIONS: 5.50 m x 4.50 m", "AREA: 24.75 m²"])
    assert dims.get("real_length") == 5.50
    assert dims.get("real_width") == 4.50
    assert dims.get("real_area") == 24.75


def test_hybrid_floorplan_analyzer_end_to_end(sample_1bhk_path):
    analyzer = HybridFloorPlanAnalyzer()
    analysis = analyzer.analyze(sample_1bhk_path)
    
    assert isinstance(analysis, FloorPlanAnalysis)
    assert len(analysis.rooms) == 5
    assert len(analysis.windows) >= 1
    
    # Check JSON export
    clean_dict = analysis.to_clean_dict()
    assert "metadata" in clean_dict
    assert "rooms" in clean_dict
    assert "doors" in clean_dict
    assert "windows" in clean_dict
    assert clean_dict["metadata"]["total_rooms"] == 5
