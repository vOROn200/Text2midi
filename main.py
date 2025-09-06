#!/usr/bin/env python3
"""
Text→MIDI generator CLI for the amaai-lab/text2midi Transformer.

Improvements over the original snippet:
- Proper CLI (arguments + interactive mode) for convenient prompt entry.
- Safe model loading (`weights_only=True` when available) and explicit device handling.
- Uses `snapshot_download` to fetch model assets once and reuse cache.
- Avoids unnecessary download of 1GB+ model weights by using the tokenizer from
  a smaller T5 repo (default: `google/flan-t5-small`).
- Clean tokenization path (no redundant pad_sequence) and `torch.no_grad()`.
- Optional AMP autocast on CUDA/MPS for speed; robust logging and file naming.
- English comments, docstrings, and logging per user's preference.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import datetime
from pathlib import Path
import pickle

# Set MPS fallback early, before importing torch (best-effort; harmless elsewhere)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformers import T5Tokenizer
from huggingface_hub import snapshot_download
import gradio as gr

# Your local module
from model.transformer_model import Transformer


def setup_logging(verbosity: int) -> None:
    """Configure simple console logging."""
    level = logging.DEBUG if verbosity > 0 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def pick_device() -> torch.device:
    """Choose best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        device=str(device),  # keep compatibility with your constructor
    )

    # Safe loading: try to avoid unpickling arbitrary objects if PyTorch supports it
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)  # PyTorch ≥ 2.4
    except TypeError:  # older PyTorch versions do not support weights_only
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_text_tokenizer(name: str, cache_dir: str | None, legacy: bool | None):
    """Load a T5 tokenizer only (no model). Use a small repo to avoid big downloads."""
    kwargs = {"cache_dir": cache_dir}
    if legacy is not None:
        kwargs["legacy"] = legacy
    # NOTE: T5 family shares the same SentencePiece vocab; small variant is enough for tokenization.
    return T5Tokenizer.from_pretrained(name, **kwargs)


def tokenize_prompt(tok: T5Tokenizer, prompt: str, device: torch.device):
    """Tokenize a single prompt into input_ids and attention_mask tensors on the device."""
    enc = tok(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    return input_ids, attention_mask


def generate_midifile(
    model: torch.nn.Module,
    r_tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    out_path: str,
    max_len: int,
    temperature: float,
    device: torch.device,
) -> str:
    """Run generation and dump a MIDI file to out_path."""
    with torch.no_grad():
        use_amp = device.type in ("cuda", "mps")
        if use_amp:
            # Autocast can speed up generation and reduce memory on GPU/MPS
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                output = model.generate(
                    input_ids, attention_mask, max_len=max_len, temperature=temperature
                )
        else:
            output = model.generate(
                input_ids, attention_mask, max_len=max_len, temperature=temperature
            )

    output_list = output[0].tolist()
    midi = r_tokenizer.decode(output_list)
    midi.dump_midi(out_path)
    return out_path

# --- Audio preview helpers ----------------------------------------------------
from pathlib import Path as _Path

_DEFAULT_SF2 = _Path.home() / ".fluidsynth" / "default_sound_font.sf2"

def _resolve_default_sf2() -> _Path | None:
    """Return default SoundFont path if it exists, else None."""
    p = _DEFAULT_SF2.expanduser()
    return p if p.exists() else None


def try_render_audio(midi_path: str, out_audio_path: str) -> tuple[str | None, str | None]:
    """Try to render MIDI -> WAV using FluidSynth via midi2audio.
    Returns (audio_path or None, error_message or None).
    Uses ~/.fluidsynth/default_sound_font.sf2 automatically.
    """
    try:
        from midi2audio import FluidSynth
    except Exception as e:
        logging.warning("Audio preview disabled: midi2audio not installed (%s)", e)
        return None, (
            "Audio preview requires midi2audio + FluidSynth. Install 'midi2audio' and Homebrew 'fluidsynth'."
        )

    sf2_path = _resolve_default_sf2()
    if not sf2_path:
        return None, (
            "Default SoundFont not found at ~/.fluidsynth/default_sound_font.sf2 — create it (symlink or copy a .sf2)."
        )

    try:
        fs = FluidSynth(sound_font=str(sf2_path))
        fs.midi_to_audio(str(midi_path), str(out_audio_path))
        return str(out_audio_path), None
    except Exception as e:
        logging.exception("FluidSynth rendering failed")
        return None, f"Audio rendering failed: {e}"


def process_prompt(
    prompt: str,
    *,
    tok: T5Tokenizer,
    model: torch.nn.Module,
    r_tokenizer,
    device: torch.device,
    max_len: int,
    temperature: float,
    out_path: str | None,
) -> str:
    """Tokenize the prompt, run generation, name and save the MIDI file."""
    input_ids, attention_mask = tokenize_prompt(tok, prompt, device)

    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_name = f"output_{slugify(prompt)}_{stamp}.mid"
        out_path = str(Path(out_name).resolve())

    logging.info(
        "Generating (max_len=%s, temperature=%.3f) → %s",
        max_len,
        temperature,
        out_path,
    )
    result = generate_midifile(
        model,
        r_tokenizer,
        input_ids,
        attention_mask,
        out_path,
        max_len,
        temperature,
        device,
    )
    logging.info("Saved %s", result)
    print(result)  # Print the path for quick copy/paste
    return result



def launch_webui(args, model, r_tokenizer, tok, device):
    """Launch a Gradio UI to generate and immediately preview MIDI as audio.
    Uses ~/.fluidsynth/default_sound_font.sf2 automatically if present.
    """
    outputs_dir = Path("outputs")
    audio_dir = outputs_dir / "audio"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    def ui_generate(prompt: str, temperature: float, max_len: int, seed_text: str | None):
        # Validate prompt
        if not prompt or not prompt.strip():
            return None, None, "Please enter a prompt."
        # Optional seed
        if seed_text:
            try:
                torch.manual_seed(int(seed_text))
            except ValueError:
                return None, None, "Seed must be an integer."

        # Tokenize and generate MIDI
        input_ids, attention_mask = tokenize_prompt(tok, prompt, device)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = f"output_{slugify(prompt)}_{stamp}"
        midi_path = str((outputs_dir / f"{base_name}.mid").resolve())
        try:
            midi_path = generate_midifile(
                model, r_tokenizer, input_ids, attention_mask,
                midi_path, max_len, temperature, device
            )
        except Exception as e:
            logging.exception("Generation failed")
            return None, None, f"Generation failed: {e}"

        # Auto render to WAV using default SF2 path if available
        wav_path = str((audio_dir / f"{base_name}.wav").resolve())
        audio_path, err = try_render_audio(midi_path, wav_path)
        status = f"Saved **{Path(midi_path).name}** to `{Path(midi_path).parent}`"
        if err:
            # Keep MIDI available even if audio failed
            status += f"<br/>{err}"
        else:
            status += f"<br/>Audio preview: **{Path(wav_path).name}**"

        return (audio_path if audio_path else None), midi_path, status

    with gr.Blocks(title="Text→MIDI — amaai-lab/text2midi") as app:
        gr.Markdown("# Text→MIDI — Enter a description and preview the generated track.")
        with gr.Row():
            prompt = gr.Textbox(lines=6, label="Prompt",
                                placeholder="Describe the music (style, instruments, key, tempo, mood)...")
        with gr.Row():
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
            max_len = gr.Slider(128, 4096, value=500, step=64, label="Max length (tokens)")
        with gr.Row():
            seed = gr.Textbox(label="Seed (optional integer)", placeholder="e.g., 42")

        generate = gr.Button("Generate", variant="primary")
        audio_out = gr.Audio(label="Audio Preview (WAV)", type="filepath")
        midi_file = gr.File(label="MIDI file (download)")
        status = gr.Markdown()

        generate.click(
            ui_generate,
            [prompt, temperature, max_len, seed],
            [audio_out, midi_file, status],
            concurrency_limit=1,
        )

    app.queue(default_concurrency_limit=1).launch(
        server_name=args.server_host,
        server_port=args.server_port,
        share=args.share,
        inbrowser=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MIDI from a natural-language prompt using amaai-lab/text2midi.",
    )
    parser.add_argument("--repo", default="amaai-lab/text2midi", help="HF repo with model + vocab.")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory (optional).")
    parser.add_argument(
        "--tokenizer",
        default="google/flan-t5-base",
        help="Tokenizer repo.",
    )
    parser.add_argument("--prompt", "-p", default=None, help="Prompt text. If omitted, enter interactive mode.")
    parser.add_argument("--prompt-file", "-P", help="Read the prompt from a UTF-8 text file.")
    parser.add_argument("--out", "-o", default=None, help="Output MIDI path (optional).")
    parser.add_argument("--max-len", type=int, default=500, help="Max generated token length.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--legacy-tokenizer",
        choices=["auto", "true", "false"],
        default="auto",
        help="Set the T5Tokenizer legacy flag. 'auto' keeps library default.",
    )
    # Web UI options
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

    # Load a T5 tokenizer (only tokenizer files are needed)
    legacy_flag: bool | None
    if args.legacy_tokenizer == "auto":
        legacy_flag = None
    else:
        legacy_flag = args.legacy_tokenizer == "true"

    tok = get_text_tokenizer(args.tokenizer, args.cache_dir, legacy_flag)

    # If web UI is requested, launch it now
    if args.webui:
        launch_webui(args, model, r_tokenizer, tok, device)
        return


    # Resolve prompt source
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
        process_prompt(
            prompt_text,
            tok=tok,
            model=model,
            r_tokenizer=r_tokenizer,
            device=device,
            max_len=args.max_len,
            temperature=args.temperature,
            out_path=args.out,
        )
        return

    if args.prompt is not None:
        process_prompt(
            args.prompt,
            tok=tok,
            model=model,
            r_tokenizer=r_tokenizer,
            device=device,
            max_len=args.max_len,
            temperature=args.temperature,
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
        process_prompt(
            prompt,
            tok=tok,
            model=model,
            r_tokenizer=r_tokenizer,
            device=device,
            max_len=args.max_len,
            temperature=args.temperature,
            out_path=None,
        )


if __name__ == "__main__":
    main()
