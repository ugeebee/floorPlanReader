"""Segmentation and architectural overlay visualizer.

Renders polygon mask overlays, dimension annotations (length, width, area),
door swings, and window openings on floor plans.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union, Optional, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from floorplan_reader.schema import FloorPlanAnalysis, RoomElement
from floorplan_reader.converters.image_preprocessor import load_and_preprocess_floorplan
from floorplan_reader.visualization.visualizer import ROOM_COLORS, ROOM_BORDER_COLORS, DOOR_COLOR, WINDOW_COLOR


def draw_styled_segmentation_overlay(
    image_source: Union[str, Path, Image.Image],
    analysis: FloorPlanAnalysis,
    output_path: Optional[Union[str, Path]] = None,
    show_labels: bool = True,
    show_dimensions: bool = True,
    show_doors_windows: bool = True,
) -> Image.Image:
    """Generate high-contrast, publication-grade architectural floor plan visualization."""
    if isinstance(image_source, Image.Image):
        base_img = image_source.convert("RGBA")
    else:
        vlm_img, _, _ = load_and_preprocess_floorplan(image_source)
        base_img = vlm_img.convert("RGBA")

    img_w, img_h = base_img.size

    # Layer for alpha-blended polygon fills
    mask_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw_mask = ImageDraw.Draw(mask_layer)

    # Layer for outlines and text
    annot_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw_annot = ImageDraw.Draw(annot_layer)

    # 1. Render Rooms
    for r in analysis.rooms:
        xmin, ymin, xmax, ymax = r.box_2d.to_pixel_xyxy(img_w, img_h)
        room_key = r.name.lower().replace(" ", "_")
        fill_color = ROOM_COLORS.get(room_key, ROOM_COLORS["default"])
        border_color = ROOM_BORDER_COLORS.get(room_key, ROOM_BORDER_COLORS["default"])

        # Check if room has custom polygon contour
        has_poly = hasattr(r, "polygon") and getattr(r, "polygon", None)
        if has_poly:
            poly_pts = getattr(r, "polygon")
            draw_mask.polygon(poly_pts, fill=fill_color)
            draw_annot.polygon(poly_pts, outline=border_color, width=3)
        else:
            draw_mask.rectangle([xmin, ymin, xmax, ymax], fill=fill_color)
            draw_annot.rectangle([xmin, ymin, xmax, ymax], outline=border_color, width=3)

        # Room label & dimension badge
        if show_labels:
            title = r.name.replace("_", " ").title()
            
            # Format dimension text
            dim_text = ""
            if show_dimensions:
                if r.real_length and r.real_width:
                    area_calc = round(r.real_length * r.real_width, 2)
                    dim_text = f"{r.real_length:.2f}m x {r.real_width:.2f}m\n({area_calc} sq m)"
                elif r.detected_label_text:
                    dim_text = r.detected_label_text
                else:
                    dim_text = f"{r.norm_length:.0f} x {r.norm_width:.0f} (norm)"

            label_content = f"{title}\n{dim_text}".strip() if dim_text else title
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            # Badge background
            lines = label_content.split("\n")
            line_height = 14
            max_w = max(len(l) for l in lines) * 7 + 14
            total_h = len(lines) * line_height + 10
            bx0 = cx - max_w // 2
            by0 = cy - total_h // 2
            bx1 = cx + max_w // 2
            by1 = cy + total_h // 2

            draw_annot.rounded_rectangle([bx0, by0, bx1, by1], radius=4, fill=(255, 255, 255, 230), outline=border_color, width=1)
            curr_y = by0 + 5
            for idx, line in enumerate(lines):
                font_color = (20, 20, 20, 255) if idx == 0 else (70, 70, 70, 255)
                lw = len(line) * 7
                draw_annot.text((cx - lw // 2 + 5, curr_y), line, fill=font_color)
                curr_y += line_height

    # 2. Render Doors (Secondary Goal)
    if show_doors_windows:
        for d in analysis.doors:
            xmin, ymin, xmax, ymax = d.box_2d.to_pixel_xyxy(img_w, img_h)
            # Vivid green/coral door marker
            draw_annot.rectangle([xmin, ymin, xmax, ymax], fill=(46, 204, 113, 140), outline=(39, 174, 96, 255), width=2)
            # Draw door tag
            dcx = (xmin + xmax) // 2
            dcy = (ymin + ymax) // 2
            draw_annot.text((dcx - 12, dcy - 5), "DOOR", fill=(255, 255, 255, 255))

        # 3. Render Windows (Secondary Goal)
        for w in analysis.windows:
            xmin, ymin, xmax, ymax = w.box_2d.to_pixel_xyxy(img_w, img_h)
            # Bright cyan/blue window marker
            draw_annot.rectangle([xmin, ymin, xmax, ymax], fill=(52, 152, 219, 140), outline=(41, 128, 185, 255), width=2)
            # Center line representing double glazing
            if (xmax - xmin) > (ymax - ymin):
                mid_y = (ymin + ymax) // 2
                draw_annot.line([(xmin, mid_y), (xmax, mid_y)], fill=(255, 255, 255, 255), width=2)
            else:
                mid_x = (xmin + xmax) // 2
                draw_annot.line([(mid_x, ymin), (mid_x, ymax)], fill=(255, 255, 255, 255), width=2)

    # 4. Summary Watermark / Legend in Top-Left
    legend_lines = [
        f"Detected Rooms: {len(analysis.rooms)}",
        f"Detected Doors: {len(analysis.doors)}",
        f"Detected Windows: {len(analysis.windows)}",
    ]
    if analysis.metadata.pixels_per_meter:
        legend_lines.append(f"Scale: {analysis.metadata.pixels_per_meter:.1f} px/m")

    leg_w = 200
    leg_h = len(legend_lines) * 16 + 14
    draw_annot.rounded_rectangle([15, 15, 15 + leg_w, 15 + leg_h], radius=6, fill=(20, 24, 33, 220), outline=(52, 152, 219, 200), width=1)
    for i, line in enumerate(legend_lines):
        draw_annot.text((25, 22 + i * 16), line, fill=(240, 240, 240, 255))

    # Composite layers
    combined = Image.alpha_composite(base_img, mask_layer)
    final_img = Image.alpha_composite(combined, annot_layer).convert("RGB")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        final_img.save(out_p)

    return final_img
