"""Hugging Face instruction-tuned LLM wrapper with caching."""

from __future__ import annotations

from typing import List, Optional

from src.config import (
    DEVICE,
    FALLBACK_MODEL_NAME,
    LOAD_IN_4BIT,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
    TOP_P,
)
from src.logging_utils import get_logger

logger = get_logger(__name__)

_tokenizer = None
_model = None
_active_model_name: Optional[str] = None
_active_device: Optional[str] = None


def _resolve_device(device: Optional[str] = None) -> str:
    device = (device or DEVICE or "auto").lower()
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def get_llm(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    load_in_4bit: Optional[bool] = None,
):
    """
    Load and cache tokenizer + causal LM.

    Prefers 4-bit quantization on CUDA when LOAD_IN_4BIT is true to fit
    Qwen2.5-3B on ~6GB GPUs. Falls back to the configured smaller model
    if the primary model fails to load.
    """
    global _tokenizer, _model, _active_model_name, _active_device

    name = model_name or MODEL_NAME
    resolved = _resolve_device(device)
    use_4bit = LOAD_IN_4BIT if load_in_4bit is None else load_in_4bit

    if _model is not None and _active_model_name == name:
        return _tokenizer, _model

    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _load(target: str):
        logger.info(
            "Loading LLM: %s (device=%s, load_in_4bit=%s)",
            target,
            resolved,
            use_4bit and resolved == "cuda",
        )
        tok = AutoTokenizer.from_pretrained(target, trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        if resolved == "cuda" and use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import torch

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                kwargs["device_map"] = "auto"
            except Exception as exc:
                logger.warning("4-bit load unavailable (%s); using float16.", exc)
                import torch

                kwargs["torch_dtype"] = torch.float16
                kwargs["device_map"] = "auto"
        elif resolved == "cuda":
            import torch

            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        else:
            import torch

            kwargs["torch_dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(target, **kwargs)
        model.eval()
        return tok, model

    try:
        _tokenizer, _model = _load(name)
        _active_model_name = name
    except Exception as primary_exc:
        logger.error("Failed to load %s: %s", name, primary_exc)
        if name != FALLBACK_MODEL_NAME:
            logger.info("Attempting fallback model: %s", FALLBACK_MODEL_NAME)
            try:
                _tokenizer, _model = _load(FALLBACK_MODEL_NAME)
                _active_model_name = FALLBACK_MODEL_NAME
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Model loading failed for {name} and fallback "
                    f"{FALLBACK_MODEL_NAME}: {fallback_exc}"
                ) from fallback_exc
        else:
            raise RuntimeError(f"Model loading failed: {primary_exc}") from primary_exc

    _active_device = resolved
    logger.info("LLM ready: %s", _active_model_name)
    return _tokenizer, _model


def active_model_name() -> str:
    return _active_model_name or MODEL_NAME


def generate_answer(
    messages: List[dict],
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> str:
    """Generate a completion from chat messages."""
    import torch

    tokenizer, model = get_llm()
    max_new_tokens = MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens
    temperature = TEMPERATURE if temperature is None else temperature
    top_p = TOP_P if top_p is None else top_p

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Generic fallback if chat template is missing
        parts = []
        for m in messages:
            parts.append(f"{m.get('role', 'user').upper()}: {m.get('content', '')}")
        parts.append("ASSISTANT:")
        prompt = "\n\n".join(parts)

    inputs = tokenizer(prompt, return_tensors="pt")
    # Move tensors to model device
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    do_sample = temperature is not None and temperature > 0

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5) if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("Empty generation")
        return text
    except Exception as exc:
        logger.error("LLM generation failure: %s", exc)
        raise RuntimeError(f"LLM generation failed: {exc}") from exc
