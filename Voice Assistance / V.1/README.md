# Multilingual Voice Assistant with Male & Female Characters

A voice assistant you can talk to in many languages, with two selectable characters (**Layla**, female, and **Omar**, male). The repo has two parts:

1. **A free, ready-to-use assistant** (Jupyter / Google Colab): speech recognition + a local LLM through Ollama + text-to-speech. No API key needed.
2. **A text-to-speech model trained from scratch** in PyTorch with two speakers (male and female), so you can build and understand your own voice.

> **Status:** the code was syntax-checked but has not been fully tested end to end. Expect to fix small environment issues on first run, and please open an issue if you hit one.

## How it works

```
microphone -> faster-whisper (speech to text, language detection)
           -> LLM (Ollama, or the Claude API)      <- the "smart" part
           -> TTS in the same language, male or female voice
           -> audio out
```

The from-scratch model only replaces the **voice** (TTS). It does not make the assistant smarter; that comes from the LLM.

## Repository contents

| File | What it is |
|---|---|
| `voice_assistant_free.ipynb` | **Start here.** Whisper + Ollama (local LLM) + Edge voices. Free, no API key. Colab and local Jupyter. |
| `voice_assistant.ipynb` | Same assistant using the Claude API for the LLM (needs an Anthropic API key, billed per token). |
| `tts_from_scratch.ipynb` | Train a multi-speaker (female + male) TTS from scratch on CMU ARCTIC or your own data. |
| `tts_scratch.py` | Command-line version of the from-scratch TTS (preprocess, train, synthesize). The assistant notebook can import it to use your own voice. |
| `voice_assistant.py` | Command-line version of the assistant (Claude API, optional offline XTTS voice cloning). |

## Quick start (free version, Google Colab)

1. Open `voice_assistant_free.ipynb` in Google Colab.
2. Choose a GPU runtime: *Runtime -> Change runtime type -> T4 GPU*.
3. Run the cells from top to bottom. The Ollama cell installs Ollama and downloads the model the first time (a few GB).
4. Test without a microphone first: `chat("What is the capital of Egypt?", "en")`.
5. Then talk: run `talk(seconds=6)` and allow microphone access in the browser.

**Local Jupyter:** install [Ollama](https://ollama.com) first, uncomment the `sounddevice` line in the first cell, and run the notebook. Recording stops automatically when you stop talking.

## Configuration (top of the notebook)

| Setting | Meaning |
|---|---|
| `LLM_MODEL` | Ollama model. Default `qwen3:8b` (strong multilingual, about 5-6 GB of VRAM). Use `gemma3:4b` on weaker hardware. |
| `WHISPER_SIZE` | `tiny` / `base` / `small` / `medium` / `large-v3`. Bigger is more accurate (especially Arabic) but slower. |
| `ALLOWED_LANGS` | Languages you speak, e.g. `["ar", "en"]`. The assistant chooses only among these, which fixes most wrong-language detection. |
| `FORCE_LANG` | Set e.g. `"ar"` to always assume one language. `None` for mixed languages. |
| `state["char"]` | `"female"` (Layla) or `"male"` (Omar). You can also say "male voice" / "female voice". |

## Languages and voices

- **Speech recognition:** Whisper handles about 99 languages.
- **Speech output:** [Edge neural voices](https://github.com/rany2/edge-tts) have both a male and a female voice for most languages. This needs an internet connection. If a language has no matching voice, a multilingual fallback voice is used.
- **LLM:** quality depends on the model. Local models are noticeably weaker on Arabic dialects than on Modern Standard Arabic.

## Train your own voice (from scratch)

Open `tts_from_scratch.ipynb` (GPU recommended). It builds a Tacotron-2-style model:

`text -> encoder + speaker embedding -> attention -> decoder -> mel spectrogram -> Griffin-Lim -> audio`

Each folder under `data/` becomes one speaker, so two folders (`female`, `male`) give two characters:

```
data/female/wavs/<id>.wav    data/female/metadata.csv   # id|transcript
data/male/wavs/<id>.wav      data/male/metadata.csv
```

Command-line equivalent:

```bash
pip install torch librosa soundfile "numpy<2.3"
python tts_scratch.py preprocess --data data --out features
python tts_scratch.py train --features features --ckpt tts.pt --steps 100000
python tts_scratch.py synth --ckpt tts.pt --speaker female --text "Hello world." --out f.wav
python tts_scratch.py synth --ckpt tts.pt --speaker male   --text "Hello world." --out m.wav
```

**Be realistic about quality:**
- About 1 hour per voice (CMU ARCTIC) only proves the pipeline works and gives rough speech. For intelligible voices you need roughly 10-20+ hours per voice and 100k+ training steps.
- Griffin-Lim output sounds robotic. A neural vocoder such as HiFi-GAN is the natural next step.
- The character set is English only. Other languages need their own data and a phonemizer front-end (for example espeak-ng).
- To use your trained voice in the assistant, put `tts_scratch.py` and `tts.pt` next to the notebook and run the optional "own voice" cell (English replies only).
- If you want production quality, fine-tuning an existing TTS model on your two voices usually beats training from scratch.

## Troubleshooting

| Problem | Fix |
|---|---|
| pip warning: `numba requires numpy<2.3` | The install cells pin `"numpy<2.3"`. Restart the runtime and run from the first cell. |
| `RuntimeError: Library libcublas.so.12 is not found` | The GPU runtime lacks CUDA 12 libraries for Whisper. The notebook now tests the GPU at load time and falls back to CPU automatically. Use `WHISPER_SIZE = "small"` on CPU. |
| Arabic detected as Hindi / Urdu | Set `ALLOWED_LANGS = ["ar", "en"]` or `FORCE_LANG = "ar"`, speak for at least 3-4 seconds, and try `WHISPER_SIZE = "medium"`. |
| Replies are slow | Use a smaller `LLM_MODEL` and Whisper size, or run Ollama on a GPU. |
| Colab cannot record | Allow microphone access in the browser, or use `chat("text", "ar")` to test without a mic. |

## Requirements

- Python 3.10+
- `faster-whisper`, `edge-tts`, `ollama`, `soundfile`, `numpy<2.3`, `nest_asyncio` (voice assistant)
- `torch`, `librosa`, `soundfile`, `numpy<2.3`, `matplotlib` (from-scratch TTS)
- [Ollama](https://ollama.com) (installed automatically on Colab)
- A GPU is recommended for the LLM and strongly recommended for training

## Licenses and responsible use

- Check the license of every model and dataset you use (Whisper, your Ollama model, CMU ARCTIC, any data you record). The optional Coqui XTTS-v2 backend in `voice_assistant.py` has a **non-commercial** license.
- Edge voices use Microsoft's online service through an unofficial library; do not rely on it for commercial products.
- Only train or clone a voice with the speaker's consent.
- Add your own project license file (for example MIT) before publishing.
