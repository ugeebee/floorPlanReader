"""Unit tests for postprocessor, predictor, and visualizer."""

import json
from pathlib import Path
from PIL import Image
import pytest

from floorplan_reader.inference.postprocessor import FloorPlanPostprocessor
from floorplan_reader.inference.predictor import FloorPlanPredictor
from floorplan_reader.visualization.visualizer import visualize_floorplan
from floorplan_reader.dataset.synthetic_adapter import generate_mock_synthetic_sample


def test_postprocessor_markdown_json_extraction():
    post = FloorPlanPostprocessor()
    raw_response = """
    Here is your structured architectural output:
    ```json
    {
      "rooms": [
        {"id": "r1", "name": "bedroom", "box_2d": [100, 100, 500, 500]},
        {"id": "r2", "name": "kitchen", "box_2d": [100, 520, 500, 900]}
      ],
      "doors": [
        {"id": "d1", "type": "single_swing", "box_2d": [300, 500, 350, 520]}
      ],
      "windows": []
    }
    ```
    I hope this helps!
    """

    analysis = post.process_generation(
        raw_text=raw_response,
        image_width=1000,
        image_height=800,
        source_format="png",
    )

    assert analysis.metadata.total_rooms == 2
    assert analysis.metadata.total_doors == 1
    assert analysis.rooms[0].name == "bedroom"
    assert analysis.rooms[1].name == "kitchen"


def test_postprocessor_syntax_repair():
    post = FloorPlanPostprocessor()
    # Notice trailing comma after the last room, which breaks standard json.loads
    malformed_json = """
    {
      "rooms": [
        {"id": "r1", "name": "bedroom", "box_2d": [100, 100, 400, 400]},
      ],
      "doors": [],
      "windows": [],
    }
    """
    analysis = post.process_generation(
        raw_text=malformed_json,
        image_width=500,
        image_height=500,
    )
    assert analysis.metadata.total_rooms == 1


def test_postprocessor_deduplication():
    post = FloorPlanPostprocessor(iou_dedup_threshold=0.8)
    duplicate_json = """
    {
      "rooms": [
        {"id": "r1", "name": "living_room", "box_2d": [100, 100, 600, 600]},
        {"id": "r2", "name": "living_room", "box_2d": [102, 101, 601, 599]}
      ],
      "doors": [],
      "windows": []
    }
    """
    analysis = post.process_generation(
        raw_text=duplicate_json,
        image_width=1000,
        image_height=1000,
    )
    # The duplicate living room with ~0.98 IoU should be merged/deduped
    assert analysis.metadata.total_rooms == 1


def test_predictor_and_visualizer_pipeline(tmp_path):
    # Generate mock floor plan sample
    img_p, _ = generate_mock_synthetic_sample(tmp_path, sample_id="test_plan", format="png")

    # Define mock model response
    mock_prediction = json.dumps({
        "rooms": [
            {"id": "r1", "name": "bedroom", "box_2d": [66, 50, 500, 500]},
            {"id": "r2", "name": "kitchen", "box_2d": [500, 500, 933, 950]},
        ],
        "doors": [
            {"id": "d1", "type": "single_swing", "box_2d": [475, 112, 525, 150]}
        ],
        "windows": [
            {"id": "w1", "type": "standard", "box_2d": [58, 150, 75, 250]}
        ],
    })

    predictor = FloorPlanPredictor(
        mock_generator_fn=lambda img, prompt: f"```json\n{mock_prediction}\n```"
    )

    analysis = predictor.predict(img_p, pixels_per_meter=50.0)
    assert analysis.metadata.total_rooms == 2
    assert analysis.metadata.total_doors == 1
    assert analysis.metadata.total_windows == 1
    assert analysis.rooms[0].real_length is not None

    # Test visualization
    out_viz = tmp_path / "overlay.png"
    viz_img = visualize_floorplan(img_p, analysis, output_path=out_viz)
    assert out_viz.exists()
    assert isinstance(viz_img, Image.Image)
