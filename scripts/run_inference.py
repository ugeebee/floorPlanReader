"""CLI tool to run floor plan analysis on JPG, PNG, or SVG drawings."""

import argparse
import json
import sys
from pathlib import Path

from floorplan_reader.inference.predictor import FloorPlanPredictor
from floorplan_reader.visualization.visualizer import visualize_floorplan


def parse_args():
    parser = argparse.ArgumentParser(description="Run floor plan analysis on JPG, PNG, or SVG drawings.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to floor plan drawing (.jpg, .png, or .svg).",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to fine-tuned LoRA weights directory.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Base HuggingFace model ID.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save output JSON analysis.",
    )
    parser.add_argument(
        "--output_viz",
        type=str,
        default=None,
        help="Path to save visualized overlay PNG.",
    )
    parser.add_argument(
        "--pixels_per_meter",
        type=float,
        default=None,
        help="Scale calibration (pixels per real-world meter) for dimension calculation.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline mock demonstration without requiring GPU/weights.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.image)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Processing floor plan: {input_path} (Format: {input_path.suffix})...")

    if args.mock or (not args.lora_path and not args.mock):
        if not args.mock:
            print("Note: No --lora_path specified. Defaulting to mock demo mode.")

        # Create demo mock prediction
        demo_json = json.dumps({
            "rooms": [
                {"id": "room_1", "name": "bedroom", "box_2d": [100, 80, 480, 480], "detected_label_text": "BEDROOM 4.0m x 3.8m"},
                {"id": "room_2", "name": "kitchen", "box_2d": [100, 500, 450, 900], "detected_label_text": "KITCHEN 3.5m x 4.0m"},
                {"id": "room_3", "name": "living_room", "box_2d": [500, 80, 900, 550], "detected_label_text": "LIVING ROOM 4.2m x 4.7m"},
                {"id": "room_4", "name": "bathroom", "box_2d": [500, 570, 900, 900], "detected_label_text": "BATHROOM 2.2m x 3.3m"},
            ],
            "doors": [
                {"id": "door_1", "type": "single_swing", "box_2d": [470, 200, 510, 250]},
                {"id": "door_2", "type": "single_swing", "box_2d": [470, 650, 510, 700]},
            ],
            "windows": [
                {"id": "win_1", "type": "standard", "box_2d": [90, 200, 105, 360], "wall_side": "north"},
                {"id": "win_2", "type": "standard", "box_2d": [90, 600, 105, 780], "wall_side": "north"},
                {"id": "win_3", "type": "standard", "box_2d": [895, 200, 905, 360], "wall_side": "south"},
            ],
        })

        predictor = FloorPlanPredictor(
            mock_generator_fn=lambda img, prompt: f"```json\n{demo_json}\n```"
        )
    else:
        print(f"Loading model {args.model_id} with LoRA adapter from {args.lora_path}...")
        predictor = FloorPlanPredictor.from_pretrained_lora(
            base_model_id=args.model_id,
            lora_weights_path=args.lora_path,
            load_in_4bit=True,
        )

    # Run Prediction
    analysis = predictor.predict(
        source=input_path,
        pixels_per_meter=args.pixels_per_meter,
    )

    # Display Summary
    print("\n--- Structured Floor Plan Analysis ---")
    print(f"Image Resolution: {analysis.metadata.image_width} x {analysis.metadata.image_height}")
    print(f"Total Rooms:      {analysis.metadata.total_rooms}")
    print(f"Total Doors:      {analysis.metadata.total_doors}")
    print(f"Total Windows:    {analysis.metadata.total_windows}")

    print("\nRooms Detected:")
    for r in analysis.rooms:
        dim_info = f"Norm Dim: {r.norm_length:.0f}x{r.norm_width:.0f} ({r.area_percentage:.1f}% area)"
        if r.real_length and r.real_width:
            dim_info += f" | Real: {r.real_length}x{r.real_width}{r.unit}"
        print(f" - [{r.id}] {r.name.upper()}: Box {r.box_2d.to_list()} | {dim_info}")

    print("\nDoors Detected:")
    for d in analysis.doors:
        print(f" - [{d.id}] {d.door_type}: Box {d.box_2d.to_list()}")

    print("\nWindows Detected:")
    for w in analysis.windows:
        wall_side = f" (Wall: {w.wall_side})" if w.wall_side else ""
        print(f" - [{w.id}] {w.window_type}: Box {w.box_2d.to_list()}{wall_side}")

    # Save Output JSON
    output_json_path = args.output_json or str(input_path.with_name(f"{input_path.stem}_analysis.json"))
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis.to_clean_dict(), f, indent=2)
    print(f"\nStructured JSON saved to: {output_json_path}")

    # Generate and Save Visual Overlay
    output_viz_path = args.output_viz or str(input_path.with_name(f"{input_path.stem}_overlay.png"))
    visualize_floorplan(input_path, analysis, output_path=output_viz_path)
    print(f"Visual overlay saved to:    {output_viz_path}")


if __name__ == "__main__":
    main()
