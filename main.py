#!/usr/bin/env python3
"""
Text→MIDI generator CLI/WebUI for amaai-lab/text2midi.

- Keeps all UX improvements: CLI args, interactive mode, Gradio web UI with audio preview,
  safe model loading, single-shot HF cache via snapshot_download, logging.
- BUT the generation itself is EXACTLY the same as in your baseline snippet:
  * Tokenizer: google/flan-t5-base
  * Tokenization flow with pad_sequence on input_ids and attention_mask
  * Fixed max_len=2000 and temperature=1.0
  * No AMP/autocast
- Comments, docstrings, and logs in English (per preference).
"""

from __future__ import annotations

import argparse
import logging
import os
# Force math SDPA globally to match CPU numeric behavior on GPU/ROCm/CUDA
os.environ["PYTORCH_SDP_BACKEND"] = "math"

import re
from datetime import datetime
from pathlib import Path
import pickle

# Prefer MPS fallback on macOS (harmless elsewhere)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from transformers import T5Tokenizer
from huggingface_hub import snapshot_download
import gradio as gr

from model.transformer_model import Transformer  # your local module

# ------------------------ logging ------------------------

def setup_logging(verbosity: int) -> None:
    """Configure simple console logging."""
    level = logging.DEBUG if verbosity > 0 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ------------------------ device pick ------------------------

def _configure_numeric_precisions_backend() -> None:
    """Настроить матричную точность ТОЛЬКО на CUDA. На ROCm пропускаем."""
    # На ROCm (HIP) ничего не трогаем — иначе ловим W906
    if getattr(torch.version, "hip", None):
        return
    # Новые API PyTorch (без deprecated-предупреждений):
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"   # 'ieee' вместо TF32
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "conv"):
            torch.backends.cudnn.conv.fp32_precision = "fp32"   # для свёрток
    except Exception:
        pass

def pick_device() -> torch.device:
    """Choose best available device: ROCm/CUDA → MPS → CPU, with clear logging."""
    if torch.cuda.is_available():
        _configure_numeric_precisions_backend()
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        print(f"[device] Using {backend}: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("[device] Using Apple MPS")
        return torch.device("mps")
    print("[device] Using CPU (no ROCm/CUDA/MPS backends detected)")
    return torch.device("cpu")

# ------------------------ helpers ------------------------

def slugify(text: str, maxlen: int = 40) -> str:
    """Filesystem-friendly name from a prompt (ASCII-ish, trimmed)."""
    s = re.sub(r"\s+", " ", text).strip().lower()
    s = re.sub(r"[^a-z0-9\-_. ]", "", s)
    s = s[:maxlen]
    return re.sub(r"\s+", "_", s) or "prompt"

def download_assets(repo_id: str, cache_dir: str | None) -> tuple[str, str]:
    """Download model assets once; return absolute paths to weights and vocab pickle."""
    local_dir = snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        allow_patterns=["pytorch_model.bin", "vocab_remi.pkl"],
    )
    model_path = str(Path(local_dir) / "pytorch_model.bin")
    vocab_path = str(Path(local_dir) / "vocab_remi.pkl")
    return model_path, vocab_path

def load_remi_tokenizer(vocab_pkl_path: str):
    """Load the REMI tokenizer object from a pickle file."""
    with open(vocab_pkl_path, "rb") as f:
        r_tokenizer = pickle.load(f)
    return r_tokenizer

def build_model(vocab_size: int, weights_path: str, device: torch.device):
    """Instantiate the Transformer and load weights onto the selected device."""
    # Keep these hyperparameters aligned with the original snippet
    d_model = 768
    n_heads = 8
    ffn_dim = 2048
    n_layers = 18
    max_len = 1024
    tie_weights = False
    n_positions = 8

    model = Transformer(
        vocab_size,
        d_model,
        n_heads,
        ffn_dim,
        n_layers,
        max_len,
        tie_weights,
        n_positions,
        device=str(device),  # constructor expects string device in your codebase
    )

    # Safe-ish loading: try weights_only=True (PyTorch ≥ 2.4), else fallback
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# ------------------------ baseline generation (unchanged) ------------------------

_BASELINE_MAX_LEN = 2000
_BASELINE_TEMPERATURE = 1.0
_BASELINE_TOKENIZER_REPO = "google/flan-t5-base"

def get_text_tokenizer(cache_dir: str | None):
    """Load the EXACT tokenizer used in the baseline (google/flan-t5-base)."""
    return T5Tokenizer.from_pretrained(_BASELINE_TOKENIZER_REPO, cache_dir=cache_dir)

def baseline_tokenize(tok: T5Tokenizer, prompt: str, device: torch.device):
    """Baseline tokenization pipeline (exactly as in the snippet, including pad_sequence)."""
    inputs = tok(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first=True, padding_value=0)
    attention_mask = nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first=True, padding_value=0)
    return input_ids.to(device), attention_mask.to(device)

def baseline_generate_to_midi(
    model: torch.nn.Module,
    r_tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    out_path: str,
) -> str:
    """Run generation EXACTLY like the baseline and dump a MIDI file to out_path."""
    with torch.no_grad():  # no AMP/autocast; pure baseline
        output = model.generate(
            input_ids,
            attention_mask,
            max_len=_BASELINE_MAX_LEN,
            temperature=_BASELINE_TEMPERATURE,
        )
    output_list = output[0].tolist()
    midi = r_tokenizer.decode(output_list)
    midi.dump_midi(out_path)
    return out_path

# ------------------------ audio preview (FluidSynth) ------------------------

_DEFAULT_SF2 = Path.home() / ".fluidsynth" / "default_sound_font.sf2"

def _resolve_default_sf2() -> Path | None:
    """Return default SoundFont path if it exists, else None (handles ~ in symlink target)."""
    p = _DEFAULT_SF2.expanduser()
    if p.exists():
        return p
    if p.is_symlink():
        try:
            target = os.readlink(p)
            t_expanded = Path(os.path.expanduser(target)).resolve()
            if t_expanded.exists():
                return t_expanded
        except OSError:
            pass
    return None

def try_render_audio(midi_path: str, out_audio_path: str) -> tuple[str | None, str | None]:
    """Try to render MIDI -> WAV using FluidSynth via midi2audio.
    Returns (audio_path or None, error_message or None). Uses ~/.fluidsynth/default_sound_font.sf2.
    """
    try:
        from midi2audio import FluidSynth
    except Exception as e:
        logging.warning("Audio preview disabled: midi2audio not installed (%s)", e)
        return None, (
            "Audio preview requires midi2audio + system FluidSynth. "
            "Install: pip install midi2audio && (apt/brew/pacman) fluidsynth."
        )

    sf2_path = _resolve_default_sf2()
    if not sf2_path:
        return None, ("Default SoundFont not found at ~/.fluidsynth/default_sound_font.sf2 — "
                      "create it (symlink or copy a .sf2).")

    try:
        fs = FluidSynth(sound_font=str(sf2_path))
        fs.midi_to_audio(str(midi_path), str(out_audio_path))
        return str(out_audio_path), None
    except Exception as e:
        logging.exception("FluidSynth rendering failed")
        return None, f"Audio rendering failed: {e}"

# ------------------------ CLI flow (uses baseline generation) ------------------------

def process_prompt_cli(
    prompt: str,
    *,
    tok: T5Tokenizer,
    model: torch.nn.Module,
    r_tokenizer,
    device: torch.device,
    out_path: str | None,
) -> str:
    """Tokenize with baseline pipeline, run baseline generation, and save MIDI."""
    input_ids, attention_mask = baseline_tokenize(tok, prompt, device)

    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_name = f"output_{slugify(prompt)}_{stamp}.mid"
        out_path = str(Path(out_name).resolve())

    logging.info("Generating (baseline: max_len=%s, temperature=%.2f) → %s",
                 _BASELINE_MAX_LEN, _BASELINE_TEMPERATURE, out_path)
    result = baseline_generate_to_midi(model, r_tokenizer, input_ids, attention_mask, out_path)
    logging.info("Saved %s", result)
    print(result)  # print path for quick copy/paste
    return result

# ------------------------ Web UI (uses baseline generation) ------------------------

def launch_webui(args, model, r_tokenizer, tok, device):
    """Gradio UI to generate and immediately preview MIDI as audio (baseline generation)."""
    outputs_dir = Path("outputs")
    audio_dir = outputs_dir / "audio"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    def ui_generate(prompt: str, seed_text: str | None):
        # Validate prompt
        if not prompt or not prompt.strip():
            return None, None, "Please enter a prompt."

        # Optional seed for reproducibility (baseline didn't set it; this only seeds RNG)
        if seed_text:
            try:
                torch.manual_seed(int(seed_text))
            except ValueError:
                return None, None, "Seed must be an integer."

        # Baseline tokenization
        input_ids, attention_mask = baseline_tokenize(tok, prompt, device)

        # Save names
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = f"output_{slugify(prompt)}_{stamp}"
        midi_path = str((outputs_dir / f"{base_name}.mid").resolve())
        wav_path = str((audio_dir / f"{base_name}.wav").resolve())

        # Baseline generation
        try:
            midi_path = baseline_generate_to_midi(model, r_tokenizer, input_ids, attention_mask, midi_path)
        except Exception as e:
            logging.exception("Generation failed")
            return None, None, f"Generation failed: {e}"

        # Optional audio preview
        audio_path, err = try_render_audio(midi_path, wav_path)
        status = f"Saved **{Path(midi_path).name}** to `{Path(midi_path).parent}`"
        if err:
            status += f"<br/>{err}"
        else:
            status += f"<br/>Audio preview: **{Path(wav_path).name}**"

        return (audio_path if audio_path else None), midi_path, status

    with gr.Blocks(title="Text→MIDI — amaai-lab/text2midi") as app:
        gr.Markdown("# Text→MIDI — Baseline generation (unchanged) with handy UI & audio preview")
        with gr.Row():
            prompt = gr.Textbox(
                lines=6,
                label="Prompt",
                placeholder="Describe the music (style, instruments, key, tempo, mood)...",
            )
        # Fixed baseline params for clarity
        with gr.Row():
            gr.Number(value=_BASELINE_MAX_LEN, precision=0, label="max_len (fixed)", interactive=False)
            gr.Number(value=_BASELINE_TEMPERATURE, label="temperature (fixed)", interactive=False)
            seed = gr.Textbox(label="Seed (optional integer)", placeholder="e.g., 42")
        generate = gr.Button("Generate", variant="primary")
        audio_out = gr.Audio(label="Audio Preview (WAV)", type="filepath")
        midi_file = gr.File(label="MIDI file (download)")
        status = gr.Markdown()

        generate.click(ui_generate, [prompt, seed], [audio_out, midi_file, status])

    app.queue(default_concurrency_limit=1).launch(
        server_name=args.server_host,
        server_port=args.server_port,
        share=args.share,
        inbrowser=True,
    )

# ------------------------ main ------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MIDI (baseline) from a natural-language prompt using amaai-lab/text2midi."
    )
    parser.add_argument("--repo", default="amaai-lab/text2midi", help="HF repo with model + vocab.")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory (optional).")

    # NOTE: Tokenizer is fixed to flan-t5-base to keep baseline intact, so no tokenizer arg here.

    parser.add_argument("--prompt", "-p", default=None, help="Prompt text. If omitted, enter interactive mode.")
    parser.add_argument("--prompt-file", "-P", help="Read the prompt from a UTF-8 text file.")
    parser.add_argument("--out", "-o", default=None, help="Output MIDI path (optional).")
    # Kept for compatibility, but ignored by baseline generation:
    parser.add_argument("--max-len", type=int, default=_BASELINE_MAX_LEN, help="(Ignored; baseline uses 2000).")
    parser.add_argument("--temperature", type=float, default=_BASELINE_TEMPERATURE, help="(Ignored; baseline uses 1.0).")

    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    # Web UI
    parser.add_argument("--webui", action="store_true", help="Launch a simple Gradio web UI.")
    parser.add_argument("--server-host", default="127.0.0.1", help="Host/IP for the web UI.")
    parser.add_argument("--server-port", type=int, default=7860, help="Port for the web UI.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase log verbosity.")

    args = parser.parse_args()
    setup_logging(args.verbose)

    device = pick_device()
    logging.info("Using device: %s", device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Download assets & load REMI tokenizer
    model_path, vocab_path = download_assets(args.repo, args.cache_dir)
    r_tokenizer = load_remi_tokenizer(vocab_path)
    vocab_size = len(r_tokenizer)
    logging.info("Vocab size: %d", vocab_size)

    # Build model and load weights
    model = build_model(vocab_size, model_path, device)

    # Load EXACT baseline tokenizer (flan-t5-base)
    tok = get_text_tokenizer(args.cache_dir)

    # Web UI?
    if args.webui:
        launch_webui(args, model, r_tokenizer, tok, device)
        return

    # CLI flows
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
        process_prompt_cli(
            prompt_text,
            tok=tok,
            model=model,
            r_tokenizer=r_tokenizer,
            device=device,
            out_path=args.out,
        )
        return

    if args.prompt is not None:
        process_prompt_cli(
            args.prompt,
            tok=tok,
            model=model,
            r_tokenizer=r_tokenizer,
            device=device,
            out_path=args.out,
        )
        return

    # Interactive mode
    print("Enter a prompt to generate MIDI (empty line to quit).")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not prompt:
            print("Bye.")
            break
        process_prompt_cli(
            prompt,
            tok=tok,
            model=model,
            r_tokenizer=r_tokenizer,
            device=device,
            out_path=None,
        )

if __name__ == "__main__":
    main()
