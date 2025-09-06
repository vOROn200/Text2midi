#!/usr/bin/env python3
# gui.py
# Minimal server to browse MIDI files on the server and preview them as audio (no generation).
# Requires: gradio, midi2audio, fluidsynth, and a default SoundFont at ~/.fluidsynth/default_sound_font.sf2
#
# Notes:
# - All comments, docs, and logs are in English as requested.
# - We render MIDI -> WAV with FluidSynth via midi2audio on demand and cache results.
# - You can point the UI to any server folder; it will list *.mid files and let you play them.
# - No file upload; only server-side browsing.

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

# ------------------------------- Config / Logging -------------------------------

def setup_logging(verbosity: int) -> None:
    level = logging.DEBUG if verbosity > 0 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ------------------------------- MIDI → WAV utils -------------------------------

_DEFAULT_SF2 = Path.home() / ".fluidsynth" / "default_sound_font.sf2"
_CACHE_DIR = Path(".cache_midi_audio")  # local cache for rendered audio

def have_midi2audio() -> bool:
    try:
        import midi2audio  # noqa: F401
        return True
    except Exception:
        return False

def resolve_sf2() -> Optional[Path]:
    """Return default SoundFont path if it exists; handle symlink with '~' target."""
    p = _DEFAULT_SF2.expanduser()
    if p.exists():
        return p
    if p.is_symlink():
        try:
            target = os.readlink(p)  # raw string; may contain '~'
            t_expanded = Path(os.path.expanduser(target)).resolve()
            if t_expanded.exists():
                return t_expanded
        except OSError:
            pass
    return None

def cache_key_for(midi_path: Path, sf2_path: Path) -> str:
    """Stable cache key based on file path + size + mtime + SF2 mtime."""
    st = midi_path.stat()
    sf = sf2_path.stat()
    msg = f"{midi_path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{sf2_path.resolve()}|{int(sf.st_mtime)}"
    return hashlib.sha1(msg.encode("utf-8")).hexdigest()

def render_midi_to_wav(midi_path: Path, cache_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Render MIDI -> WAV using FluidSynth; return (wav_path or None, error or None)."""
    if not have_midi2audio():
        return None, (
            "Audio preview requires 'midi2audio' and system 'fluidsynth'. "
            "Install: pip install midi2audio && sudo apt install fluidsynth (or your distro)."
        )
    sf2 = resolve_sf2()
    if not sf2:
        return None, (
            "Default SoundFont not found at ~/.fluidsynth/default_sound_font.sf2. "
            "Create it (copy or symlink to a valid .sf2)."
        )
    from midi2audio import FluidSynth

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key_for(midi_path, sf2)
    wav_path = cache_dir / f"{key}.wav"
    if wav_path.exists():
        return wav_path, None  # cache hit

    try:
        logging.info("Rendering with FluidSynth | sf2=%s | midi=%s", sf2, midi_path)
        fs = FluidSynth(sound_font=str(sf2))
        fs.midi_to_audio(str(midi_path), str(wav_path))
        return wav_path, None
    except Exception as e:
        logging.exception("FluidSynth rendering failed")
        return None, f"Audio rendering failed: {e}"

# ------------------------------- Directory scanning ----------------------------

def list_midis(base_dir: Path, recursive: bool, limit: int = 5000) -> List[Path]:
    """Return sorted list of .mid/.midi paths under base_dir."""
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    globs = ["*.mid", "*.midi"]
    results: List[Path] = []
    if recursive:
        for pat in globs:
            results.extend(base_dir.rglob(pat))
    else:
        for pat in globs:
            results.extend(base_dir.glob(pat))
    # Deduplicate, sort by name, limit results to keep UI snappy
    uniq = sorted(set(p.resolve() for p in results), key=lambda p: str(p).lower())
    return uniq[:limit]

# ------------------------------- Gradio app ------------------------------------

def launch_app(args) -> None:
    base_dir = Path(args.base_dir).expanduser().resolve()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Shared state inside the closure
    state = {"base_dir": base_dir, "recursive": args.recursive}

    def scan_action(base: str, recursive: bool):
        """Scan for MIDI files and return (choices, status_text)."""
        bd = Path(base).expanduser()
        state["base_dir"] = bd
        state["recursive"] = recursive
        files = list_midis(bd, recursive=recursive)
        if not files:
            return gr.update(choices=[], value=None), f"Found 0 MIDI files under: {bd}"
        # Show relative paths if under base dir, otherwise absolute
        rel_choices = []
        for p in files:
            try:
                rel = p.relative_to(bd)
                rel_choices.append(str(rel))
            except ValueError:
                rel_choices.append(str(p))
        status = f"Found {len(files)} MIDI files under: {bd} (recursive={recursive})"
        return gr.update(choices=rel_choices, value=rel_choices[0]), status

    def _resolve_selection(selection: str) -> Optional[Path]:
        """Convert UI selection to absolute path."""
        if not selection:
            return None
        p = Path(selection)
        if not p.is_absolute():
            p = state["base_dir"] / p
        return p.resolve()

    def play_action(selection: str, direct_path: str):
        """Render selected MIDI to WAV and return (audio_path, status_text)."""
        # 1) Resolve which file to use: direct path has priority
        midi_path: Optional[Path] = None
        if direct_path and direct_path.strip():
            midi_path = Path(direct_path).expanduser().resolve()
        else:
            midi_path = _resolve_selection(selection)

        if not midi_path or not midi_path.exists() or not midi_path.is_file():
            return None, "Please select a MIDI file or provide a valid direct path."

        # 2) Render (or use cache)
        wav_path, err = render_midi_to_wav(midi_path, _CACHE_DIR)
        if err:
            return None, f"Selected: {midi_path}\n{err}"
        return str(wav_path), f"Selected: {midi_path}\nRendered: {wav_path}"

    with gr.Blocks(title="MIDI Browser & Player (server)") as app:
        gr.Markdown("# 🎵 MIDI Browser & Player\nBrowse server folders, pick a `.mid`, and listen in the browser.")
        with gr.Row():
            base_in = gr.Textbox(value=str(base_dir), label="Base directory on server")
        with gr.Row():
            recursive = gr.Checkbox(value=args.recursive, label="Scan recursively")
            refresh = gr.Button("Scan", variant="primary")

        with gr.Row():
            midi_list = gr.Dropdown(choices=[], value=None, label="MIDI files under base directory")
        with gr.Row():
            direct_path = gr.Textbox(value="", label="Or: direct path to a server-side .mid (optional)")
        with gr.Row():
            play_btn = gr.Button("Render & Play", variant="primary")

        audio_out = gr.Audio(label="Audio Preview (WAV)", type="filepath")
        status = gr.Markdown()

        # Wire up events
        refresh.click(fn=scan_action, inputs=[base_in, recursive], outputs=[midi_list, status])
        play_btn.click(fn=play_action, inputs=[midi_list, direct_path], outputs=[audio_out, status])

        # Initial scan
        app.load(fn=scan_action, inputs=[base_in, recursive], outputs=[midi_list, status])

    # Serialize rendering on server (optional: safer when multiple users click play)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.open_browser,
    )

# ------------------------------- CLI -------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serve a simple UI to browse server-side MIDI files and preview them as audio."
    )
    p.add_argument("--base-dir", default="outputs", help="Base directory to scan for MIDI files.")
    p.add_argument("--recursive", action="store_true", help="Scan subdirectories recursively.")
    p.add_argument("--host", default="127.0.0.1", help="Server host/IP.")
    p.add_argument("--port", type=int, default=7861, help="Server port.")
    p.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    p.add_argument("--open-browser", action="store_true", help="Open browser automatically.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase log verbosity.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.info("Base directory: %s", Path(args.base_dir).resolve())
    logging.info("Expecting SF2 at: %s", _DEFAULT_SF2)
    if not have_midi2audio():
        logging.warning("Package 'midi2audio' not installed. Audio preview will not work.")
    if not resolve_sf2():
        logging.warning("Default SoundFont not found at %s", _DEFAULT_SF2)
    launch_app(args)

if __name__ == "__main__":
    main()
