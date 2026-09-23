"""Model loader and LoRA PEFT configuration wrapper for Qwen3-VL / Qwen2.5-VL.

Supports both Unsloth FastVisionModel (optimal for Google Colab Free T4 15GB VRAM)
and Hugging Face transformers + bitsandbytes 4-bit QLoRA.
"""

from __future__ import annotations
import logging
from typing import Tuple, Any, Optional, List, Dict

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
FALLBACK_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Target linear layers for QLoRA fine-tuning
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_qwen_vl_model(
    model_id: str = DEFAULT_MODEL_ID,
    load_in_4bit: bool = True,
    use_unsloth: bool = True,
    device_map: str = "auto",
) -> Tuple[Any, Any]:
    """Load Qwen-VL model and processor with 4-bit quantization for T4 GPU compatibility.

    Args:
        model_id: HuggingFace model repo ID (e.g. Qwen/Qwen3-VL-8B-Instruct).
        load_in_4bit: Whether to quantize weights to 4-bit NormalFloat (NF4).
        use_unsloth: If True, attempts to use unsloth.FastVisionModel for 2x speed and lowest VRAM.
        device_map: Device mapping strategy ('auto' or specific device).

    Returns:
        Tuple of (model, processor_or_tokenizer).
    """
    model = None
    processor = None

    # Option 1: Try Unsloth (fastest, optimized kernels for Colab T4)
    if use_unsloth:
        try:
            from unsloth import FastVisionModel  # type: ignore

            logger.info(f"Loading {model_id} via Unsloth FastVisionModel (4-bit={load_in_4bit})...")
            model, processor = FastVisionModel.from_pretrained(
                model_name=model_id,
                load_in_4bit=load_in_4bit,
                use_gradient_checkpointing="unsloth",
            )
            return model, processor
        except (ImportError, Exception) as e_unsloth:
            logger.info(f"Unsloth loader not available ({e_unsloth}). Falling back to Hugging Face transformers...")

    # Option 2: Standard Hugging Face transformers + bitsandbytes
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

    quant_config = None
    if load_in_4bit:
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    logger.info(f"Loading {model_id} via transformers AutoModelForVision2Seq...")
    try:
        processor = AutoProcessor.from_pretrained(model_id, min_pixels=512*512, max_pixels=896*896)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map=device_map,
            torch_dtype="auto",
        )
    except Exception as e_load:
        # If Qwen3 is not yet cached or model ID fallback is needed
        logger.warning(f"Error loading {model_id}: {e_load}. Attempting fallback {FALLBACK_MODEL_ID}...")
        processor = AutoProcessor.from_pretrained(FALLBACK_MODEL_ID, min_pixels=512*512, max_pixels=896*896)
        model = AutoModelForVision2Seq.from_pretrained(
            FALLBACK_MODEL_ID,
            quantization_config=quant_config,
            device_map=device_map,
            torch_dtype="auto",
        )

    return model, processor


def apply_lora_to_model(
    model: Any,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_unsloth: bool = True,
) -> Any:
    """Apply Low-Rank Adaptation (LoRA) adapters to the vision-language model.

    Args:
        model: Loaded model instance.
        r: LoRA attention dimension / rank (default 16).
        lora_alpha: LoRA scaling alpha parameter (default 32).
        lora_dropout: Dropout probability for LoRA layers (default 0.05).
        target_modules: Module names to attach LoRA adapters to.
        use_unsloth: Whether to use Unsloth's get_peft_model if available.

    Returns:
        Model with LoRA adapters injected.
    """
    if target_modules is None:
        target_modules = LORA_TARGET_MODULES

    # Try Unsloth PEFT first
    if use_unsloth and hasattr(model, "get_peft_model"):
        from unsloth import FastVisionModel  # type: ignore

        logger.info(f"Applying Unsloth LoRA (r={r}, alpha={lora_alpha})...")
        peft_model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,  # Keep vision encoder frozen to save VRAM on T4
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            random_state=42,
        )
        return peft_model

    # Standard PEFT fallback
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    logger.info(f"Applying PEFT LoraConfig (r={r}, alpha={lora_alpha})...")
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)
