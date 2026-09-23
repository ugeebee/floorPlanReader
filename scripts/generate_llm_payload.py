#!/usr/bin/env python3
"""CLI utility to generate Neuro-Symbolic LLM prompts and structured JSON data.

Usage:
    python scripts/generate_llm_payload.py --image executive-office-suite-2d.png
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from floorplan_reader.reasoning.llm_payload_builder import LLMReasoningPayloadBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-sensor pipeline (YOLO + LSD + Tesseract) and generate LLM reasoning prompt."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="executive-office-suite-2d.png",
        help="Path to floor plan / blueprint image (PNG/JPG).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/rtx4050_best.pt",
        help="Path to YOLO segmentation weights.",
    )
    parser.add_argument(
        "--output-prompt",
        type=str,
        default="gemini_reasoning_prompt.md",
        help="Markdown output file for copy-pasting to Gemini/Claude/Qwen.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="extracted_blueprint_data.json",
        help="JSON output file for raw intermediate data.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for YOLO inference (cpu, cuda:0, auto).",
    )

    args = parser.parse_args()

    builder = LLMReasoningPayloadBuilder(
        weights_path=args.weights,
        device=args.device,
    )

    result = builder.process_blueprint(
        image_path=args.image,
        output_prompt_path=args.output_prompt,
        output_json_path=args.output_json,
        save_debug_overlay=True,
    )

    print("\n" + "=" * 65)
    print("✨ SUCCESS: Multi-Sensor Pipeline Complete!")
    print(f"📄 Master Prompt: file://{Path(args.output_prompt).resolve()}")
    print(f"📊 Structured Data: file://{Path(args.output_json).resolve()}")
    print("=" * 65)
    print("\n👉 You can now open 'gemini_reasoning_prompt.md', copy the entire prompt,")
    print("   and paste it into Google Gemini, Claude, or Kimi to get a refined CAD/SVG output!\n")


if __name__ == "__main__":
    main()
