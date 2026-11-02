# Book AI Illustrator

Paste a page of a book and get an illustration of it, rendered on your own PC. Nothing is sent to the cloud, and there is no per-image cost.

The app is a small local web page in front of [ComfyUI](https://github.com/Comfy-Org/ComfyUI), which runs the [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) model on an 8 GB NVIDIA GPU.

The hard parts are speed on consumer hardware and keeping characters looking the same from one page to the next. This project is a working attempt at both.

## Features

- Paste a page, click Generate, get an image. On an RTX 4060 Ti (8 GB) that takes roughly 10–25 seconds, depending on the step count and whether the models are already loaded.
- Each book is a data file with its own visual style and character descriptions.
- A character's description is added to the prompt only when that character's name appears in the page.
- A live progress bar driven by ComfyUI: loading, sampling step by step, decoding, saving.
- Python standard library only. Nothing to `pip install`.

## Requirements

- Windows 10 or 11 (`run.bat` is Windows-only; `server.py` is plain Python)
- An NVIDIA GPU with 8 GB of VRAM or more
- Python 3.10 or newer (developed on 3.12)
- About 13 GB of disk space for ComfyUI and the models

## Setup

### 1. Install ComfyUI

Download `ComfyUI_windows_portable_nvidia.7z` from the [ComfyUI releases](https://github.com/Comfy-Org/ComfyUI/releases) and extract it. Run `run_nvidia_gpu.bat` once to check that it starts, then close it.

### 2. Download the models

Put these three files into the ComfyUI folder:

| File | Put it in `ComfyUI\models\` | Download |
|---|---|---|
| `flux-2-klein-4b-fp8.safetensors` (about 4 GB) | `diffusion_models` | [link](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors) |
| `qwen_3_4b.safetensors` (about 8 GB) | `text_encoders` | [link](https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) |
| `flux2-vae.safetensors` (about 0.3 GB) | `vae` | [link](https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/main/split_files/vae/flux2-vae.safetensors) |

This repository does not include model files.

### 3. Get the app

```
git clone https://github.com/jperod/book-ai-illustrator.git
cd book-ai-illustrator
```

### 4. Add your books

Copy `books/example-book.json` to `books/<your-book>.json` and edit it. Book files are git-ignored on purpose: they hold your own notes, and character descriptions for copyrighted books should stay out of a public repository.

## Running

1. Start ComfyUI (`run_nvidia_gpu.bat` in its folder) and wait until it has finished loading.
2. Run `run.bat` in this folder. The app opens at http://127.0.0.1:8000.
3. Pick a book, paste a page, and click Generate. Images are saved in `outputs/`.

## Book files

Each book is one JSON file in `books/`:

```json
{
  "id": "example-book",
  "title": "Example Book",
  "author": "Author Name",
  "series": "",
  "book_number": 1,
  "visual_style": "Watercolor and ink, muted earth tones, soft directional light.",
  "characters": [
    { "name": "Mara", "description": "young woman, short dark hair, green travelling cloak" }
  ]
}
```

- `visual_style` is added to every prompt for that book.
- A character is added to a prompt only if their name appears in the pasted text. Matching is case-sensitive, so a lowercase "patience" does not pull in a character named Patience.
- Edits to the Visual style and Characters boxes are saved back to the book file.
- The book `id` must use only lowercase letters, digits and hyphens. It is also the file name.

## Configuration

Settings live in `config.json`:

| Key | Default | Meaning |
|---|---|---|
| `comfy_url` | `http://127.0.0.1:8188` | Where ComfyUI is running |
| `port` | `8000` | Port for this app |
| `width`, `height` | `1024` | Image size in pixels |
| `steps` | `8` | Sampling steps. More steps take longer and can add detail |
| `timeout_seconds` | `600` | How long to wait for ComfyUI before giving up |
| `open_browser` | `true` | Open the page when the app starts |

## Project layout

```
server.py           local web server, ComfyUI client, prompt builder, progress relay
index.html          the page (one file, no build step)
workflow_api.json   FLUX.2 Klein 4B text-to-image graph in ComfyUI's API format
config.json         settings
books/              one JSON file per book (your own books are git-ignored)
tests/              tests that use a fake ComfyUI, so no GPU is needed
run.bat             starts the app and opens the browser (Windows)
```

## Tests

```
python -m unittest discover -s tests -v
```

The tests use a fake ComfyUI and temporary folders. They don't need a GPU, and they don't touch your books or outputs.

## Roadmap

Ideas, not built yet:

- Reference images per character, so the same face and outfit carry across pages
- A quality mode using the undistilled FLUX.2 Klein base model
- Upscaling
- Illustrating pages while reading EPUB or PDF files, for example through a reader such as OpenReader

## Credits and licenses

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) runs the models. See its repository for its license.
- [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) by Black Forest Labs, Apache 2.0.
- The text encoder and VAE are repackaged by [Comfy-Org](https://huggingface.co/Comfy-Org/flux2-klein-4B), Apache 2.0.

The code in this repository is MIT licensed. See [LICENSE](LICENSE).
