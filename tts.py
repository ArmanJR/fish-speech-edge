# /// script
# requires-python = ">=3.10"
# ///
"""
Fish Speech TTS — single-script text-to-speech for Jetson.

Loads the model, runs inference, saves audio, then frees all GPU memory.

Usage:
    source .venv/bin/activate
    python tts.py "Hello, world!"
    python tts.py "Hello, world!" --output /tmp/hello.wav
    python tts.py "Hello, world!" --seed 123 --temperature 0.8
"""

import argparse
import gc
import os
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from loguru import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

FISH_SPEECH_DIR = Path(__file__).resolve().parent

# Max sequence length for KV cache. The default (32768) wastes ~2.1GB on
# Jetson's unified memory for positions that are never used during TTS.
# generate_long reserves 2048 positions as minimum headroom, so the floor
# is ~2100. 4096 gives ample room while cutting KV cache from 2.25GB to 288MB.
MAX_SEQ_LEN = 4096

# Default cap on generated tokens. Prevents the progress bar from showing
# 32K iterations. Enough for ~5 minutes of speech.
DEFAULT_MAX_NEW_TOKENS = 1024


def load_text2semantic_model(
    checkpoint_path: Path, device: str, precision: torch.dtype
):
    """Load the DualAR text-to-semantic model onto the given device."""
    from fish_speech.models.text2semantic.llama import DualARTransformer
    from fish_speech.models.text2semantic.inference import decode_one_token_ar

    model = DualARTransformer.from_pretrained(
        str(checkpoint_path), load_weights=True, max_length=MAX_SEQ_LEN
    )

    # Move to device and set dtype, preserving int8 quantized weight dtypes
    model._apply(
        lambda t: t.to(device=device)
        if t.dtype == torch.int8
        else t.to(device=device, dtype=precision)
    )
    model.eval()

    logger.info(
        f"Text2Semantic model loaded on {device} ({precision}), "
        f"max_seq_len={model.config.max_seq_len}"
    )
    return model, decode_one_token_ar


def load_codec_model(codec_path: Path, device: str, precision: torch.dtype):
    """Load the DAC codec model for decoding VQ codes to audio."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_path = FISH_SPEECH_DIR / "fish_speech" / "configs" / "modded_dac_vq.yaml"
    cfg = OmegaConf.load(str(config_path))
    codec = instantiate(cfg)

    state_dict = torch.load(str(codec_path), map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    codec.load_state_dict(state_dict, strict=False)
    codec.eval()
    codec.to(device=device, dtype=precision)

    logger.info("Codec model loaded")
    return codec


def free_gpu(*models):
    """Delete models and aggressively free all GPU memory."""
    for m in models:
        if m is not None:
            del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("GPU memory freed")


@torch.inference_mode()
def synthesize(
    text: str,
    output_path: Path,
    checkpoint_path: Path,
    device: str = "cuda",
    precision: torch.dtype = torch.float16,
    seed: int = 42,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 30,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    iterative_prompt: bool = True,
    chunk_length: int = 300,
) -> Path:
    """
    Full TTS pipeline: text -> semantic tokens -> audio file.

    Returns the absolute path to the generated .wav file.
    """
    from fish_speech.models.text2semantic.inference import generate_long

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    t_start = time.time()
    codec = None

    try:
        # --- Stage 1: Load text2semantic model and generate codes ---
        logger.info("Stage 1: Loading text2semantic model...")
        model, decode_one_token = load_text2semantic_model(
            checkpoint_path, device, precision
        )

        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info(f"Model loaded in {time.time() - t_start:.1f}s")

        # Generate semantic tokens
        t_gen = time.time()
        generator = generate_long(
            model=model,
            device=device,
            decode_one_token=decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            iterative_prompt=iterative_prompt,
            chunk_length=chunk_length,
        )

        all_codes = []
        for response in generator:
            if response.action == "sample":
                all_codes.append(response.codes.cpu())
                logger.info(f"Generated codes for: {response.text}")
            elif response.action == "next":
                break

        if not all_codes:
            raise RuntimeError("No codes generated — model produced empty output")

        merged_codes = torch.cat(all_codes, dim=1)
        gen_time = time.time() - t_gen
        logger.info(
            f"Semantic generation done in {gen_time:.1f}s, "
            f"codes shape: {merged_codes.shape}"
        )

        # Free the text2semantic model before loading codec
        free_gpu(model)

        # --- Stage 2: Decode codes to audio ---
        logger.info("Stage 2: Loading codec model...")
        codec_path = checkpoint_path / "codec.pth"
        codec = load_codec_model(codec_path, device, precision)

        codes_gpu = merged_codes.to(device).long()
        if codes_gpu.ndim == 2:
            codes_gpu = codes_gpu.unsqueeze(0)

        audio = codec.from_indices(codes_gpu)
        audio_np = audio[0, 0].float().cpu().numpy()
        sample_rate = codec.sample_rate

        # Free codec model
        free_gpu(codec)
        codec = None

        # --- Stage 3: Save audio ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_np, sample_rate)

        duration = len(audio_np) / sample_rate
        total_time = time.time() - t_start
        logger.info(
            f"Saved {duration:.2f}s audio to {output_path} "
            f"(total pipeline: {total_time:.1f}s, "
            f"realtime factor: {total_time / duration:.1f}x)"
        )
        return output_path.resolve()

    except Exception:
        # Ensure GPU is freed even on error
        free_gpu(codec)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Fish Speech TTS — text to speech on Jetson"
    )
    parser.add_argument("text", help="Text to convert to speech")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output WAV path (default: output/<uuid>.wav)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=FISH_SPEECH_DIR / "checkpoints" / "s2-pro",
        help="Path to model checkpoint directory",
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k sampling")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Max semantic tokens to generate",
    )

    args = parser.parse_args()

    if args.output is None:
        output_dir = FISH_SPEECH_DIR / "output"
        output_dir.mkdir(exist_ok=True)
        args.output = output_dir / f"{uuid.uuid4().hex[:8]}.wav"

    result_path = synthesize(
        text=args.text,
        output_path=args.output,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        precision=torch.float16,  # fp16 required for Jetson memory constraints
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    # Print the output path as the final line (for piping/scripting)
    print(result_path)


if __name__ == "__main__":
    main()
