"""Unit tests for SVG converter and image preprocessing."""

import io
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from floorplan_reader.converters.svg_converter import convert_svg_to_png, is_svg_file
from floorplan_reader.converters.image_preprocessor import (
    smart_resize_for_vlm,
    load_and_preprocess_floorplan,
)

SAMPLE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="500" height="500">
  <rect width="500" height="500" fill="#ffffff"/>
  <!-- Room 1 -->
  <rect x="50" y="50" width="180" height="180" fill="none" stroke="#000000" stroke-width="4"/>
  <text x="70" y="100" font-family="sans-serif" font-size="14">Bedroom</text>
  <!-- Room 2 -->
  <rect x="250" y="50" width="200" height="350" fill="none" stroke="#000000" stroke-width="4"/>
  <text x="280" y="100" font-family="sans-serif" font-size="14">Living Room</text>
  <!-- Door -->
  <line x1="230" y1="120" x2="250" y2="120" stroke="#ff0000" stroke-width="6"/>
  <!-- Window -->
  <line x1="100" y1="50" x2="160" y2="50" stroke="#0000ff" stroke-width="6"/>
</svg>"""


def test_is_svg_file():
    assert is_svg_file(SAMPLE_SVG)
    assert is_svg_file(SAMPLE_SVG.encode("utf-8"))
    assert not is_svg_file("some_photo.png")


def test_convert_svg_to_png(tmp_path):
    svg_path = tmp_path / "sample_plan.svg"
    svg_path.write_text(SAMPLE_SVG, encoding="utf-8")

    out_png = tmp_path / "rasterized.png"
    img = convert_svg_to_png(svg_path, output_path=out_png)

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert out_png.exists()
    assert img.size[0] > 0 and img.size[1] > 0


def test_smart_resize_for_vlm():
    # Large 4000x3000 floor plan
    large_img = Image.new("RGB", (4000, 3000), color=(255, 255, 255))
    resized, orig_size = smart_resize_for_vlm(
        large_img, min_pixels=512 * 512, max_pixels=896 * 896, factor=28
    )

    assert orig_size == (4000, 3000)
    # Both dimensions must be divisible by 28
    assert resized.size[0] % 28 == 0
    assert resized.size[1] % 28 == 0
    # Total pixels must not exceed max_pixels (+ tiny round off tolerance)
    assert resized.size[0] * resized.size[1] <= 896 * 896 + (28 * 896 * 2)


def test_load_and_preprocess_floorplan_from_svg(tmp_path):
    svg_file = tmp_path / "blueprint.svg"
    svg_file.write_text(SAMPLE_SVG, encoding="utf-8")

    vlm_img, orig_size, src_fmt = load_and_preprocess_floorplan(svg_file)
    assert src_fmt == "svg"
    assert vlm_img.size[0] % 28 == 0
    assert vlm_img.size[1] % 28 == 0
