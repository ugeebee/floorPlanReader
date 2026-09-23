"""SVG to raster (PNG/RGB) converter for architectural floor plans.

Handles transparent backgrounds by compositing onto a crisp white background
and upscales vector paths to ensure door swings, window mullions, and blueprint
text remain sharp for VLM tokenization.
"""

from __future__ import annotations
import os
import io
import logging
from pathlib import Path
from typing import Union, Optional
from PIL import Image

logger = logging.getLogger(__name__)


def is_svg_file(path_or_content: Union[str, Path, bytes]) -> bool:
    """Check if the given input is an SVG file path or raw SVG content."""
    if isinstance(path_or_content, (str, Path)):
        p_str = str(path_or_content).strip()
        if p_str.lower().endswith(".svg") and os.path.exists(p_str):
            return True
        if p_str.startswith("<svg") or "xmlns=\"http://www.w3.org/2000/svg\"" in p_str:
            return True
    elif isinstance(path_or_content, bytes):
        prefix = path_or_content[:200].decode("utf-8", errors="ignore")
        if "<svg" in prefix or "xmlns" in prefix:
            return True
    return False


def _composite_on_white(img: Image.Image) -> Image.Image:
    """Ensure the image is in RGB format composited over a solid white background."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha_img = img.convert("RGBA")
        background = Image.new("RGBA", alpha_img.size, (255, 255, 255, 255))
        composed = Image.alpha_composite(background, alpha_img)
        return composed.convert("RGB")
    return img.convert("RGB")


def convert_svg_to_png(
    svg_input: Union[str, Path, bytes],
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    target_scale: float = 2.0,
    background_color: str = "white",
) -> Image.Image:
    """Convert SVG floor plan vector graphics into a high-definition raster PIL Image.

    Tries cairosvg first for high-performance Cairo vector rendering,
    with a fallback to svglib + reportlab if libcairo is not found.

    Args:
        svg_input: Path to .svg file, SVG string, or SVG bytes.
        output_path: Optional path to write PNG file on disk.
        dpi: Target dots-per-inch (default 300 for crisp architectural lines).
        target_scale: Scaling factor for rasterization (2.0 renders at 2x resolution).
        background_color: Background color for transparent SVGs (default 'white').

    Returns:
        PIL.Image.Image in RGB mode.
    """
    svg_bytes: bytes
    if isinstance(svg_input, (str, Path)) and os.path.isfile(str(svg_input)):
        with open(str(svg_input), "rb") as f:
            svg_bytes = f.read()
    elif isinstance(svg_input, str):
        svg_bytes = svg_input.encode("utf-8")
    elif isinstance(svg_input, bytes):
        svg_bytes = svg_input
    else:
        raise ValueError(f"Unsupported svg_input type: {type(svg_input)}")

    pil_image: Optional[Image.Image] = None

    # Method 1: Try CairoSVG
    try:
        import cairosvg  # type: ignore

        png_bytes = cairosvg.svg2png(
            bytestring=svg_bytes,
            dpi=dpi,
            scale=target_scale,
            background_color=background_color,
        )
        pil_image = Image.open(io.BytesIO(png_bytes))
        pil_image.load()
    except Exception as e_cairo:
        logger.debug(f"CairoSVG conversion not available or failed: {e_cairo}. Trying svglib...")

    # Method 2: Fallback to svglib + reportlab
    if pil_image is None:
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM

            temp_stream = io.BytesIO(svg_bytes)
            drawing = svg2rlg(temp_stream)
            if drawing is None:
                raise ValueError("svglib could not parse the SVG content.")

            # Scale drawing
            drawing.width = drawing.width * target_scale
            drawing.height = drawing.height * target_scale
            drawing.scale(target_scale, target_scale)

            img_buf = io.BytesIO()
            renderPM.drawToFile(drawing, img_buf, fmt="PNG", dpi=dpi)
            img_buf.seek(0)
            pil_image = Image.open(img_buf)
            pil_image.load()
        except Exception as e_svglib:
            raise RuntimeError(
                f"Failed to rasterize SVG floor plan using both CairoSVG and svglib. Error: {e_svglib}"
            )

    pil_image = _composite_on_white(pil_image)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pil_image.save(out_p, format="PNG")

    return pil_image
