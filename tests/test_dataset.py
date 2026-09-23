"""Unit tests for dataset parsing, synthetic adapter, and dataset mixing."""

import json
from pathlib import Path
import pytest
from PIL import Image

from floorplan_reader.dataset.cubicasa_parser import CubiCasaParser
from floorplan_reader.dataset.synthetic_adapter import (
    SyntheticFloorPlanAdapter,
    generate_mock_synthetic_sample,
    generate_mock_synthetic_dataset,
)
from floorplan_reader.dataset.dataset_mixer import DatasetMixer


def test_cubicasa_parser_coco_sample(tmp_path):
    parser = CubiCasaParser()
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    annotations = [
        {"bbox": [100, 50, 300, 200], "category_name": "Bedroom"},
        {"bbox": [450, 50, 250, 400], "category_name": "Living Room"},
        {"bbox": [390, 100, 20, 80], "category_name": "Door"},
        {"bbox": [150, 45, 100, 10], "category_name": "Window"},
    ]

    record = parser.parse_coco_sample(
        image_path_or_pil=img,
        annotations=annotations,
        img_width=800,
        img_height=600,
        sample_id="test_sample_01",
        output_image_dir=tmp_path / "images",
    )

    assert record["id"] == "test_sample_01"
    assert len(record["messages"]) == 2
    user_msg = record["messages"][0]["content"]
    assert user_msg[0]["type"] == "image"
    assert Path(user_msg[0]["image"]).exists()

    assistant_msg = record["messages"][1]["content"][0]["text"]
    parsed_json = json.loads(assistant_msg)
    assert len(parsed_json["rooms"]) == 2
    assert len(parsed_json["doors"]) == 1
    assert len(parsed_json["windows"]) == 1
    assert parsed_json["rooms"][0]["name"] == "bedroom"


def test_synthetic_adapter_mock_generator(tmp_path):
    # Generate mock synthetic sample
    img_p, json_p = generate_mock_synthetic_sample(tmp_path, sample_id="synth_01", format="svg")
    assert img_p.exists()
    assert json_p.exists()

    adapter = SyntheticFloorPlanAdapter()
    samples = adapter.ingest_paired_directory(tmp_path, output_images_dir=tmp_path / "vlm_cache")

    assert len(samples) == 1
    record = samples[0]
    assert record["id"] == "synth_01"
    parsed_json = json.loads(record["messages"][1]["content"][0]["text"])
    assert len(parsed_json["rooms"]) == 4
    assert len(parsed_json["doors"]) == 3
    assert len(parsed_json["windows"]) == 3


def test_dataset_mixer(tmp_path):
    # Create 10 mock synthetic samples
    mock_samples = generate_mock_synthetic_dataset(tmp_path / "synth", num_samples=10, format="svg")
    adapter = SyntheticFloorPlanAdapter()
    synth_records = adapter.ingest_paired_directory(tmp_path / "synth", output_images_dir=tmp_path / "cache")

    # Mock some cubicasa records
    parser = CubiCasaParser()
    cubi_records = []
    for i in range(10):
        img = Image.new("RGB", (500, 500), color=(250, 250, 250))
        rec = parser.parse_coco_sample(
            image_path_or_pil=img,
            annotations=[{"bbox": [50, 50, 200, 200], "category_name": "Kitchen"}],
            img_width=500,
            img_height=500,
            sample_id=f"cubi_{i}",
            output_image_dir=tmp_path / "cubi_cache",
        )
        cubi_records.append(rec)

    mixer = DatasetMixer(cubicasa_ratio=0.7, val_split_ratio=0.2, seed=123)
    summary = mixer.build_and_export(
        cubicasa_samples=cubi_records,
        synthetic_samples=synth_records,
        output_dir=tmp_path / "mixed_dataset",
        target_total_samples=15,
    )

    assert summary["total_samples"] == 15
    assert summary["val_samples"] == 3
    assert summary["train_samples"] == 12
    assert Path(summary["train_file"]).exists()
    assert Path(summary["val_file"]).exists()
