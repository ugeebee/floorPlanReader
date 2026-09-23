"""Image preprocessor for floor plan visual inputs.

Supports JPG, PNG, and SVG inputs. Provides intelligent aspect-ratio preserving
scaling tuned for Qwen-VL tokenization and T4 GPU memory safety.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Union, Tuple, Optional
from PIL import Image

from floorplan_reader.converters.svg_converter import convert_svg_to_png, is_svg_file


# Default pixel bounds tuned for Colab Free T4 (15GB VRAM)
DEFAULT_MIN_PIXELS = 512 * 512   # 262,144 pixels (~330 visual tokens)
DEFAULT_MAX_PIXELS = 896 * 896   # 802,816 pixels (~1,024 visual tokens)
IMAGE_FACTOR = 28                # Standard patch dimension divisor for Qwen-VL models


def smart_resize_for_vlm(
    image: Image.Image,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    factor: int = IMAGE_FACTOR,
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Resize image to fit within VLM token budgets while preserving exact aspect ratio.

    Ensures both width and height are multiples of `factor` (28 for Qwen-VL vision transformer),
    preventing padding artifacts and token misalignment.

    Args:
        image: Source PIL Image.
        min_pixels: Minimum total pixel count (upscales tiny crops).
        max_pixels: Maximum total pixel count (downscales 4K blueprints to prevent T4 OOM).
        factor: Divisibility factor (28 for Qwen2.5-VL / Qwen3-VL patch grid).

    Returns:
        Tuple of (resized_image, original_dimensions (width, height)).
    """
    orig_w, orig_h = image.size
    total_pixels = orig_w * orig_h

    scale = 1.0
    if total_pixels > max_pixels:
        scale = math.sqrt(max_pixels / total_pixels)
    elif total_pixels < min_pixels:
        scale = math.sqrt(min_pixels / total_pixels)

    new_w = max(factor, int(round(orig_w * scale / factor)) * factor)
    new_h = max(factor, int(round(orig_h * scale / factor)) * factor)

    if (new_w, new_h) == (orig_w, orig_h):
        return image.copy(), (orig_w, orig_h)

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, (orig_w, orig_h)


def load_and_preprocess_floorplan(
    source: Union[str, Path, bytes, Image.Image],
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> Tuple[Image.Image, Tuple[int, int], str]:
    """Load floor plan from path, bytes, or Image, handling SVG conversion automatically.

    Args:
        source: Filepath (jpg, png, svg), bytes, or PIL Image.
        min_pixels: Minimum pixel area bound.
        max_pixels: Maximum pixel area bound.

    Returns:
        Tuple of (preprocessed_pil_image, original_size (w, h), source_format).
    """
    source_format = "png"

    if isinstance(source, Image.Image):
        img = source.convert("RGB")
        orig_size = img.size
    elif is_svg_file(source):
        source_format = "svg"
        img = convert_svg_to_png(source)
        orig_size = img.size
    elif isinstance(source, (str, Path)):
        p = Path(source)
        suffix = p.suffix.lower().replace(".", "")
        source_format = suffix if suffix in ("jpg", "jpeg", "png", "svg") else "png"
        if suffix == "svg":
            img = convert_svg_to_png(p)
        else:
            raw = Image.open(p)
            img = raw.convert("RGB")
        orig_size = img.size
    elif isinstance(source, bytes):
        raw = Image.open(io.BytesIO(source))
        img = raw.convert("RGB")
        orig_size = img.size
    else:
        raise ValueError(f"Unsupported floor plan input type: {type(source)}")

    # Smart resize for VLM
    vlm_ready_img, _ = smart_resize_for_vlm(
        img, min_pixels=min_pixels, max_pixels=max_pixels, factor=IMAGE_FACTOR
    )

    return vlm_ready_img, orig_size, source_format


def prepare_image_for_vlm(
    source: Union[str, Path, Image.Image],
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Tuple[int, int], str]:
    """Preprocess image and ensure it exists as a clean RGB PNG on disk for training or inference.

    Args:
        source: Image path or PIL Image.
        output_dir: Directory where temporary processed PNG should be saved.

    Returns:
        Tuple of (saved_file_path, original_size, source_format).
    """
    img, orig_size, src_fmt = load_and_preprocess_floorplan(source)

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(source, (str, Path)):
            base_name = Path(source).stem
        else:
            base_name = "floorplan_processed"
        target_path = out_dir / f"{base_name}.png"
        img.save(target_path, format="PNG")
        return target_path, orig_size, src_fmt

    # If no output dir, save alongside source or return existing path if PNG/JPG
    if isinstance(source, (str, Path)) and not is_svg_file(source):
        return Path(source), orig_size, src_fmt

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")
    return Path(tmp.name), orig_size, src_fmt
