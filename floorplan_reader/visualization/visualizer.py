"""Visualizer for detected rooms, doors, windows, and dimensional annotations."""

from __future__ import annotations
from pathlib import Path
from typing import Union, Optional
from PIL import Image, ImageDraw, ImageFont

from floorplan_reader.schema import FloorPlanAnalysis
from floorplan_reader.converters.image_preprocessor import load_and_preprocess_floorplan

# Distinct curated color palette for rooms
ROOM_COLORS = {
    "bedroom": (52, 152, 219, 70),       # Soft Blue
    "master_bedroom": (41, 128, 185, 70),
    "living_room": (46, 204, 113, 70),   # Emerald Green
    "kitchen": (241, 196, 15, 70),       # Warm Yellow
    "bathroom": (155, 89, 182, 70),      # Amethyst Purple
    "hallway": (149, 165, 166, 70),      # Neutral Grey
    "balcony": (26, 188, 156, 70),       # Turquoise
    "dining_room": (230, 126, 34, 70),   # Orange
    "office": (52, 73, 94, 70),          # Slate
    "storage": (127, 140, 141, 70),
    "default": (100, 100, 100, 60),
}

ROOM_BORDER_COLORS = {
    "bedroom": (41, 128, 185, 220),
    "master_bedroom": (31, 97, 141, 220),
    "living_room": (39, 174, 96, 220),
    "kitchen": (212, 172, 13, 220),
    "bathroom": (142, 68, 173, 220),
    "hallway": (127, 140, 141, 220),
    "balcony": (22, 160, 133, 220),
    "dining_room": (211, 84, 0, 220),
    "office": (44, 62, 80, 220),
    "storage": (100, 100, 100, 220),
    "default": (60, 60, 60, 220),
}

DOOR_COLOR = (231, 76, 60, 220)    # Vivid Coral Red
WINDOW_COLOR = (52, 152, 219, 220)  # Bright Cyan


def visualize_floorplan(
    image_source: Union[str, Path, Image.Image],
    analysis: FloorPlanAnalysis,
    output_path: Optional[Union[str, Path]] = None,
    show_labels: bool = True,
    show_dimensions: bool = True,
) -> Image.Image:
    """Generate visual overlay showing detected rooms, doors, and windows with labels.

    Args:
        image_source: Filepath or PIL Image.
        analysis: FloorPlanAnalysis containing detected elements.
        output_path: Optional destination file path to save PNG.
        show_labels: Whether to render room text labels.
        show_dimensions: Whether to include room dimension measurements.

    Returns:
        PIL Image with semi-transparent overlays and text.
    """
    if isinstance(image_source, Image.Image):
        base_img = image_source.convert("RGBA")
    else:
        vlm_img, _, _ = load_and_preprocess_floorplan(image_source)
        base_img = vlm_img.convert("RGBA")

    img_w, img_h = base_img.size

    # Layer for semi-transparent boxes
    overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Layer for crisp text and borders
    annotation_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw_annot = ImageDraw.Draw(annotation_layer)

    # 1. Draw Rooms
    for r in analysis.rooms:
        xmin, ymin, xmax, ymax = r.box_2d.to_pixel_xyxy(img_w, img_h)
        room_key = r.name.lower().replace(" ", "_")
        fill_color = ROOM_COLORS.get(room_key, ROOM_COLORS["default"])
        border_color = ROOM_BORDER_COLORS.get(room_key, ROOM_BORDER_COLORS["default"])

        # Fill semi-transparent room box
        draw_overlay.rectangle([xmin, ymin, xmax, ymax], fill=fill_color)
        # Outline
        draw_annot.rectangle([xmin, ymin, xmax, ymax], outline=border_color, width=2)

        # Room label text
        if show_labels:
            label_text = r.name.replace("_", " ").title()
            if show_dimensions:
                if r.real_length and r.real_width:
                    dim_str = f"{r.real_length}x{r.real_width}{r.unit}"
                else:
                    dim_str = f"{r.norm_length:.0f}x{r.norm_width:.0f} (norm)"
                if r.area_percentage:
                    dim_str += f" • {r.area_percentage:.1f}%"
                full_text = f"{label_text}\n{dim_str}"
            else:
                full_text = label_text

            # Position in center of room
            text_x = xmin + 8
            text_y = ymin + 8
            # Dark pill background behind text for readability
            draw_annot.rectangle([text_x - 3, text_y - 2, text_x + 140, text_y + 28], fill=(30, 30, 30, 180))
            draw_annot.text((text_x, text_y), full_text, fill=(255, 255, 255, 255))

    # 2. Draw Doors
    for d in analysis.doors:
        xmin, ymin, xmax, ymax = d.box_2d.to_pixel_xyxy(img_w, img_h)
        # Ensure minimum visible thickness (at least 4px)
        if (xmax - xmin) < 6:
            mid = (xmin + xmax) // 2
            xmin, xmax = mid - 3, mid + 3
        if (ymax - ymin) < 6:
            mid = (ymin + ymax) // 2
            ymin, ymax = mid - 3, mid + 3
        draw_annot.rectangle([xmin, ymin, xmax, ymax], fill=DOOR_COLOR, outline=(192, 57, 43, 255), width=1)

    # 3. Draw Windows
    for w in analysis.windows:
        xmin, ymin, xmax, ymax = w.box_2d.to_pixel_xyxy(img_w, img_h)
        if (xmax - xmin) < 6:
            mid = (xmin + xmax) // 2
            xmin, xmax = mid - 3, mid + 3
        if (ymax - ymin) < 6:
            mid = (ymin + ymax) // 2
            ymin, ymax = mid - 3, mid + 3
        draw_annot.rectangle([xmin, ymin, xmax, ymax], fill=WINDOW_COLOR, outline=(41, 128, 185, 255), width=1)

    # Composite layers: base + overlay + annotations
    composed = Image.alpha_composite(base_img, overlay)
    composed = Image.alpha_composite(composed, annotation_layer)
    final_img = composed.convert("RGB")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        final_img.save(out_p, format="PNG")

    return final_img
