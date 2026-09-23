"""Unit tests for floorplan_reader schema and coordinate geometry."""

import pytest
from floorplan_reader.schema import (
    BoundingBox2D,
    RoomElement,
    DoorElement,
    WindowElement,
    FloorPlanAnalysis,
)


def test_bounding_box_init_and_properties():
    # Test init from list [ymin, xmin, ymax, xmax]
    box = BoundingBox2D.parse_from_list_or_dict([100, 200, 400, 700])
    assert box.ymin == 100
    assert box.xmin == 200
    assert box.ymax == 400
    assert box.xmax == 700
    assert box.norm_width == 500
    assert box.norm_height == 300
    assert box.to_list() == [100, 200, 400, 700]


def test_bounding_box_inverted_auto_fix():
    # If model outputs inverted coords (e.g. ymax < ymin), it should automatically fix
    box = BoundingBox2D.parse_from_list_or_dict([500, 800, 200, 100])
    assert box.ymin == 200
    assert box.ymax == 500
    assert box.xmin == 100
    assert box.xmax == 800


def test_bounding_box_pixel_denormalization():
    box = BoundingBox2D.parse_from_list_or_dict([100, 200, 500, 800])
    # For a 1000x2000 image (width=2000, height=1000)
    # ymin=100 -> 0.1 * 1000 = 100
    # xmin=200 -> 0.2 * 2000 = 400
    # ymax=500 -> 0.5 * 1000 = 500
    # xmax=800 -> 0.8 * 2000 = 1600
    px_xmin, px_ymin, px_xmax, px_ymax = box.to_pixel_xyxy(img_width=2000, img_height=1000)
    assert (px_xmin, px_ymin, px_xmax, px_ymax) == (400, 100, 1600, 500)

    x, y, w, h = box.to_pixel_xywh(img_width=2000, img_height=1000)
    assert (x, y, w, h) == (400, 100, 1200, 400)


def test_bounding_box_iou():
    box1 = BoundingBox2D.parse_from_list_or_dict([0, 0, 100, 100])
    box2 = BoundingBox2D.parse_from_list_or_dict([0, 0, 100, 100])
    assert box1.iou(box2) == pytest.approx(1.0)

    box3 = BoundingBox2D.parse_from_list_or_dict([200, 200, 300, 300])
    assert box1.iou(box3) == 0.0

    # 50% overlap horizontally
    box4 = BoundingBox2D.parse_from_list_or_dict([0, 50, 100, 150])
    # Inter area: 100 * 50 = 5000
    # Union area: 10000 + 10000 - 5000 = 15000
    assert box1.iou(box4) == pytest.approx(5000 / 15000)


def test_room_element_dimension_calculation():
    room = RoomElement(
        id="room_1",
        name="master_bedroom",
        box_2d=BoundingBox2D.parse_from_list_or_dict([100, 200, 600, 800]),
    )
    # Width = 600, Height = 500
    assert room.norm_length == 600.0
    assert room.norm_width == 500.0
    # Area = 600 * 500 = 300,000 / 1,000,000 = 30%
    assert room.area_percentage == 30.0


def test_floorplan_analysis_from_model_prediction():
    mock_prediction = {
        "rooms": [
            {
                "id": "r1",
                "name": "Kitchen",
                "box_2d": [100, 100, 400, 400],
            },
            {
                "id": "r2",
                "name": "Living Room",
                "box_2d": [100, 450, 800, 900],
            },
        ],
        "doors": [
            {
                "id": "d1",
                "box_2d": [200, 410, 280, 440],
                "type": "single_swing",
                "connects": ["r1", "r2"],
            }
        ],
        "windows": [
            {
                "id": "w1",
                "box_2d": [90, 150, 105, 350],
                "type": "standard",
                "wall": "north",
            }
        ],
    }

    analysis = FloorPlanAnalysis.from_model_prediction(
        raw_dict=mock_prediction,
        image_width=1200,
        image_height=800,
        source_format="png",
        pixels_per_meter=100.0,
    )

    assert analysis.metadata.total_rooms == 2
    assert analysis.metadata.total_doors == 1
    assert analysis.metadata.total_windows == 1
    assert analysis.rooms[0].name == "kitchen"
    assert analysis.doors[0].connects == ["r1", "r2"]
    # With pixels_per_meter = 100, check real length calculated
    assert analysis.rooms[0].real_length is not None
    clean = analysis.to_clean_dict()
    assert "rooms" in clean
    assert len(clean["rooms"]) == 2
