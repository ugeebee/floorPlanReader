"""Grounding and architectural prompt templates for Qwen3-VL / Qwen2.5-VL."""

FLOORPLAN_SYSTEM_PROMPT = (
    "You are an expert architectural vision AI specialized in reading floor plans, blueprints, "
    "and architectural drawings. Your task is to detect all rooms, spaces, doors, and windows, "
    "and provide their exact 2D bounding boxes in normalized coordinates [ymin, xmin, ymax, xmax] "
    "on a scale of 0 to 1000, along with their labels and dimensions."
)

FLOORPLAN_USER_INSTRUCTION = (
    "Analyze this architectural floor plan drawing. Detect all rooms, doors, and windows.\n"
    "Output a valid JSON object with the following schema:\n"
    "{\n"
    '  "rooms": [\n'
    '    {"id": "room_1", "name": "bedroom", "box_2d": [ymin, xmin, ymax, xmax], "detected_label_text": "optional text"},\n'
    '    {"id": "room_2", "name": "kitchen", "box_2d": [ymin, xmin, ymax, xmax]}\n'
    "  ],\n"
    '  "doors": [\n'
    '    {"id": "door_1", "type": "single_swing", "box_2d": [ymin, xmin, ymax, xmax]}\n'
    "  ],\n"
    '  "windows": [\n'
    '    {"id": "win_1", "type": "standard", "box_2d": [ymin, xmin, ymax, xmax]}\n'
    "  ]\n"
    "}\n"
    "Use normalized bounding coordinates [ymin, xmin, ymax, xmax] scaled from 0 to 1000, where:\n"
    "- ymin, ymax: 0 (top of image) to 1000 (bottom of image)\n"
    "- xmin, xmax: 0 (left of image) to 1000 (right of image)\n"
    "Return ONLY the structured JSON inside markdown ```json ... ``` blocks."
)


def build_vlm_prompt(custom_instruction: str = None) -> str:
    """Return the complete prompt string for floor plan grounding."""
    if custom_instruction:
        return custom_instruction
    return FLOORPLAN_USER_INSTRUCTION
