"""Predictor pipeline for floor plan analysis.

Accepts JPG, PNG, or SVG inputs, executes VLM inference, and returns structured
FloorPlanAnalysis with validated room dimensions, doors, and windows.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Union, Optional, Any, Callable
from PIL import Image

from floorplan_reader.converters.image_preprocessor import load_and_preprocess_floorplan
from floorplan_reader.models.prompt_templates import build_vlm_prompt, FLOORPLAN_SYSTEM_PROMPT
from floorplan_reader.inference.postprocessor import FloorPlanPostprocessor
from floorplan_reader.schema import FloorPlanAnalysis

logger = logging.getLogger(__name__)


class FloorPlanPredictor:
    """High-level floor plan analysis inference engine."""

    def __init__(
        self,
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
        postprocessor: Optional[FloorPlanPostprocessor] = None,
        mock_generator_fn: Optional[Callable[[Image.Image, str], str]] = None,
    ):
        """Initialize the predictor.

        Args:
            model: Fine-tuned Qwen-VL model instance (or None if mock_generator_fn provided).
            processor: AutoProcessor corresponding to the model.
            postprocessor: FloorPlanPostprocessor instance.
            mock_generator_fn: Optional mock function for offline testing without GPU.
        """
        self.model = model
        self.processor = processor
        self.postprocessor = postprocessor or FloorPlanPostprocessor()
        self.mock_generator_fn = mock_generator_fn

    @classmethod
    def from_pretrained_lora(
        cls,
        base_model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        lora_weights_path: str = "./lora_adapter",
        load_in_4bit: bool = True,
        use_unsloth: bool = True,
    ) -> FloorPlanPredictor:
        """Convenience factory method to load a base model with fine-tuned LoRA weights."""
        from floorplan_reader.models.qwen_vl_wrapper import load_qwen_vl_model
        from peft import PeftModel

        model, processor = load_qwen_vl_model(
            model_id=base_model_id,
            load_in_4bit=load_in_4bit,
            use_unsloth=use_unsloth,
        )

        logger.info(f"Loading LoRA weights from {lora_weights_path}...")
        if use_unsloth and hasattr(model, "load_lora"):
            model = model.load_lora(lora_weights_path)
        else:
            model = PeftModel.from_pretrained(model, lora_weights_path)

        model.eval()
        return cls(model=model, processor=processor)

    def predict(
        self,
        source: Union[str, Path, bytes, Image.Image],
        pixels_per_meter: Optional[float] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> FloorPlanAnalysis:
        """Run complete floor plan analysis on an input image or vector drawing.

        Args:
            source: Path to .jpg, .png, or .svg, or raw bytes, or PIL Image.
            pixels_per_meter: Optional scale calibration (e.g. 100 pixels = 1 meter).
            max_new_tokens: Maximum tokens for generation.
            temperature: Sampling temperature (low temperature = deterministic coordinates).

        Returns:
            Structured FloorPlanAnalysis instance.
        """
        # Step 1: Preprocess and detect format / dimensions
        vlm_img, (orig_w, orig_h), source_format = load_and_preprocess_floorplan(source)

        # Step 2: Run model inference (or mock generator if offline)
        prompt_text = build_vlm_prompt()

        if self.mock_generator_fn is not None:
            raw_output = self.mock_generator_fn(vlm_img, prompt_text)
        elif self.model is not None and self.processor is not None:
            raw_output = self._run_vlm_inference(
                vlm_img, prompt_text, max_new_tokens=max_new_tokens, temperature=temperature
            )
        else:
            raise RuntimeError(
                "Neither model+processor nor mock_generator_fn is configured on FloorPlanPredictor."
            )

        # Step 3: Resilient postprocessing & schema creation
        analysis = self.postprocessor.process_generation(
            raw_text=raw_output,
            image_width=orig_w,
            image_height=orig_h,
            source_format=source_format,
            pixels_per_meter=pixels_per_meter,
        )

        return analysis

    def _run_vlm_inference(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Execute Qwen-VL model forward pass."""
        import torch

        messages = [
            {"role": "system", "content": FLOORPLAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        # Move tensors to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.processor.tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode generated response
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        return response_text
