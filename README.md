# Book AI Illustrator

Paste a page of a book and get an illustration of it, generated locally on your own GPU. No cloud service and no per-image cost.

The app is a small web page in front of [ComfyUI](https://github.com/comfyanonymous/ComfyUI), which runs the [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) image model.

The hard parts are speed on consumer hardware and keeping characters looking the same from one page to the next. This project is an attempt at both.

## Features

- Paste a page, click Generate. On an 8 GB NVIDIA card (RTX 4060 Ti) an image takes roughly 10–25 seconds, depending on the step count and whether the models are already loaded.
- Each book is a JSON file with its own visual style and character descriptions.
- A character's description is added to the prompt only when that character's name appears in the page.
- Live progress from ComfyUI: loading, sampling step by step, decoding, saving.
- The app itself uses only the Python standard library. Nothing to install with pip.

## Requirements

- Linux with an NVIDIA GPU (8 GB of VRAM or more) and a working driver
- Python 3.10 or newer, `git`, and `curl`
- About 13 GB of disk space for ComfyUI and the models

## Setup

### 1. Install ComfyUI

Clone ComfyUI and follow its README to install it, choosing the PyTorch build that matches your CUDA version:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
```

Check that it starts with `python3 main.py`, then stop it again.

### 2. Download the models

From the ComfyUI folder:

```bash
mkdir -p models/diffusion_models models/text_encoders models/vae

curl -L -o models/diffusion_models/flux-2-klein-4b-fp8.safetensors \
  https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors

curl -L -o models/text_encoders/qwen_3_4b.safetensors \
  https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors

curl -L -o models/vae/flux2-vae.safetensors \
  https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/main/split_files/vae/flux2-vae.safetensors
```

The downloads are about 4 GB, 8 GB and 0.3 GB. This repository does not include model files.

### 3. Get the app

```bash
git clone https://github.com/jperod/book-ai-illustrator.git
cd book-ai-illustrator
```

### 4. Add your books

Copy `books/example-book.json` to `books/<your-book>.json` and edit it. New book files are git-ignored by default, so personal notes don't get committed by accident. To share a book in the repository, add a `!books/<your-book>.json` line to `.gitignore`.

The repository includes `books/assassins-apprentice.json` as a worked example: a full visual style and a character list for Robin Hobb's novel.

## Running

1. Start ComfyUI with `python3 main.py` in its folder, and wait until it reports that it is running.
2. In this folder, run `python3 server.py`. The app opens at http://127.0.0.1:8000, or open that address yourself.
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
- The book `id` may use only lowercase letters, digits and hyphens. It is also the file name.

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
books/              one JSON file per book (new books are git-ignored unless listed in .gitignore)
tests/              tests that use a fake ComfyUI, so no GPU is needed
outputs/            generated images (git-ignored, created on first run)
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests use a fake ComfyUI and temporary folders. They don't need a GPU, and they don't touch your books or outputs. They pass on Python 3.12.

## Roadmap

Ideas, not built yet:

- Reference images per character, so the same face and outfit carry across pages
- A quality mode using the undistilled FLUX.2 Klein base model
- Upscaling
- Illustrating pages while reading EPUB or PDF files, for example through a reader such as OpenReader

## Credits and licenses

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) runs the models. See its repository for its license.
- [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) by Black Forest Labs, Apache 2.0.
- The text encoder and VAE are repackaged by [Comfy-Org](https://huggingface.co/Comfy-Org/flux2-klein-4B), Apache 2.0.

The code in this repository is MIT licensed. See [LICENSE](LICENSE).

## Disclaimer

Not affiliated with any author or publisher. For personal use; you're responsible for the rights to any book you use.
