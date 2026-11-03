# Book AI Illustrator

Paste a page of a book and get an illustration of it, generated locally on your own GPU. No cloud service and no per-image cost.

The app is a small web page in front of [ComfyUI](https://github.com/comfyanonymous/ComfyUI), which runs the [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) image model. An optional local language model, served by [Ollama](https://ollama.com/), chooses the moment to draw and who is in it.

The hard parts are speed on consumer hardware and keeping characters looking the same from one page to the next. This project is an attempt at both.

## Features

- Paste a page, click Generate. On an 8 GB NVIDIA card (RTX 4060 Ti) an image takes roughly 10–25 seconds, depending on the step count and whether the models are already loaded.
- **Scene planner (optional).** A local model decides who is visible in the chosen moment and writes a short scene for the image. If it is off or fails, the app falls back to name matching. See [Scene planner](#scene-planner-optional).
- **Scene box.** Type the moment yourself to skip the planner.
- **Books as JSON.** Each book has its own visual style, characters, narrator and glossary.
- A character is added to the prompt only when the name appears on the page, or when the planner says they are visible.
- **Glossary.** Invented terms such as "Forged" get a meaning for the planner and visual notes for the image. They are used only when the term appears on the page.
- **Narrator picker** for first-person books.
- **Prompt used** panel shows the exact prompt sent to the image model. The status line shows the planner's time and who it put in the picture.
- Live progress from ComfyUI: loading, sampling step by step, decoding, saving.
- Edits to a book's style and characters are saved back to its file, written atomically.
- The app itself uses only the Python standard library. Nothing to install with pip.

## Tech stack

| Area | What is used | Notes |
|---|---|---|
| App server | Python 3.12, standard library only: `http.server`, `urllib`, `threading`, `difflib` | No pip dependencies. Managed with uv |
| Progress | ComfyUI's `/ws` socket, read by a small WebSocket client in `server.py` | Feeds the live progress bar |
| Page | One `index.html` with plain JavaScript and CSS | No framework, no build step |
| Image model | FLUX.2 Klein 4B (fp8), Qwen3 4B text encoder and FLUX.2 VAE, run in ComfyUI | Graph in `workflow_api.json`, in ComfyUI's API format |
| Scene planner | Ollama `/api/chat` with a JSON schema. Default model `qwen3:8b`, thinking off | Runs on the GPU and is unloaded before FLUX runs |
| Evaluation | `eval/harness.py` (offline prompt checks) and `eval/bench_planner.py` (planner benchmark) | Standard library only |
| Tests | `unittest`, with a fake ComfyUI and a fake Ollama | No GPU needed |
| Target hardware | NVIDIA RTX 4060 Ti, 8 GB VRAM. Linux and Windows | |

## Requirements

- An NVIDIA GPU with 8 GB of VRAM or more, and a working driver
- [uv](https://docs.astral.sh/uv/), which installs the Python version this project needs
- `git` and `curl`
- About 13 GB of disk space for ComfyUI and the image models
- For the scene planner (optional): [Ollama](https://ollama.com/download) and about 5.3 GB of disk space for `qwen3:8b`

The app runs on Linux and Windows. The `uv` commands below are the same in either system.
The download commands in step 2 use bash syntax, so on Windows run them in Git Bash or WSL.

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
uv sync
```

`uv sync` creates the project environment in `.venv/` and uses the Python version from `.python-version`. The app itself has no dependencies.

If you use both Windows and WSL on the same folder, keep their environments apart by giving WSL its own location:

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/book-illustrator uv sync
```

### 4. Add your books

Copy `books/example-book.json` to `books/<your-book>.json` and edit it. New book files are git-ignored by default, so personal notes don't get committed by accident. To share a book in the repository, add a `!books/<your-book>.json` line to `.gitignore`.

The repository includes `books/assassins-apprentice.json` as a worked example: a visual style, a character list, a narrator and a glossary for Robin Hobb's novel.

## Running

1. Start ComfyUI with `python3 main.py` in its folder, and wait until it reports that it is running.
2. If the scene planner is on, make sure Ollama is running too. The Ollama desktop app starts it in the background.
3. In this folder, run `uv run python server.py`. The app opens at http://127.0.0.1:8000, or open that address yourself.
4. Pick a book, paste a page, and click Generate. Images are saved in `outputs/`.

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

- `visual_style` is added to every prompt for that book. If `visual_style_short` is also set, it is used instead.
- `characters` are matched by name. Matching is case-sensitive, so a lowercase "patience" does not pull in a character named Patience.
- `narrator` names the viewpoint character. Pick it in the Narrator box. The narrator is included when the page is first person ("I") or when they are named.
- `glossary` is a list of `{ "term", "meaning", "visual" }` entries. `meaning` goes to the planner and `visual` goes into the image prompt. A term is used only when it appears on the page. See `books/assassins-apprentice.json` for an example.
- Edits to the Visual style and Characters boxes are saved back to the book file.
- The book `id` may use only lowercase letters, digits and hyphens. It is also the file name.

## Scene planner (optional)

A small local model does two jobs for each page. Its candidates are the names found on the page, plus the narrator when the narration is first person.

1. **Who is in the picture.** It marks each candidate visible or not in the chosen moment. Someone who is only remembered, mentioned, absent or dead is left out.
2. **The scene.** It writes up to 60 words of visible detail for the image: poses, setting, props.

If you type into the Scene box, the planner is skipped and your text is used. If the planner is off, fails, or Ollama does not answer in time, the app falls back to name matching and the page text.

Setup:

1. Install [Ollama](https://ollama.com/download). On Linux: `curl -fsSL https://ollama.com/install.sh | sh`.
2. Pull the model: `ollama pull qwen3:8b`.
3. Make sure `"llm_enabled": true` is set in `config.json`. The code default is off.

The planner runs on the GPU by default. Before it runs, the app asks ComfyUI to release its models, and it unloads the planner before the image is generated. The next image therefore takes about 8 seconds longer while FLUX reloads.

The page text is sent only to the Ollama address in `llm_url`, which is on your own machine by default.

Benchmarks on an RTX 4060 Ti (8 GB): six test cases, two passes each, using the app's settings (`checks2` mode, thinking off, temperature 0, 2048-token context). Seconds are the warm median per page.

| Model (Ollama tag) | Where | Seconds per page | Recall | Precision | Forbidden names |
|---|---|---|---|---|---|
| `qwen3:8b` | GPU | 2.0 | 0.97 | 1.00 | 0 |
| `qwen3:4b-instruct-2507-q4_K_M` | GPU | 1.2 | 0.92 | 1.00 | 0 |
| `qwen3.5:9b` | GPU | 2.8 | 0.92 | 1.00 | 0 |
| `qwen3:4b-instruct-2507-q4_K_M` | CPU | 7.8 | 0.92 | 1.00 | 0 |

The sample is small, so treat these as indicative. See [Evaluation](#evaluation) to rerun them.

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
| `llm_enabled` | `false` (`true` in this repo's `config.json`) | Turn the scene planner on |
| `llm_url` | `http://127.0.0.1:11434` | Where Ollama is running |
| `llm_model` | `qwen3:8b` | Ollama model used by the planner |
| `llm_timeout` | `60` | Seconds to wait for the planner before falling back |
| `llm_num_gpu` | `99` | Layers on the GPU. `0` runs the planner on the CPU |
| `llm_keep_alive` | `"2m"` | How long Ollama keeps the planner loaded between requests |

UI choices such as the default book are kept in `settings.json`, which is git-ignored.

## Project layout

```
server.py           local web server, ComfyUI client, scene planner, prompt builder, progress relay
index.html          the page (one file, no build step)
workflow_api.json   FLUX.2 Klein 4B text-to-image graph in ComfyUI's API format
config.json         settings (see Configuration)
settings.json       UI choices such as the default book (git-ignored)
pyproject.toml      project metadata; uv reads it
uv.lock             locked environment (no dependencies)
.python-version     Python version used by uv
books/              one JSON file per book (new books are git-ignored unless listed in .gitignore)
eval/               planner benchmark and offline prompt checks (see Evaluation)
tests/              unit tests with a fake ComfyUI and a fake Ollama, so no GPU is needed
outputs/            generated images (git-ignored, created on first run)
TODO.md             ideas and open work
```

## Evaluation

- `uv run python eval/harness.py` builds the prompts the app would send for each case in `eval/local/cases/` and checks who is listed, that the narrator comes first, and that the scene is not too long. It needs no GPU, ComfyUI or Ollama.
- `uv run python eval/bench_planner.py --model qwen3:8b --mode checks2 --num-gpu 99 --think off --candidates` runs the planner through Ollama and writes a report to `eval/local/out/`. Use `--num-gpu 0` for CPU only.

Planner cases are in `eval/cases/planner/` (synthetic, committed) and `eval/local/cases/planner/` (real book pages, git-ignored).

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

The tests use a fake ComfyUI, a fake Ollama and temporary folders. They don't need a GPU or Ollama, and they don't touch your books or outputs. They pass on Python 3.12.

## Linting and pre-commit hooks

[Ruff](https://docs.astral.sh/ruff/) checks and formats the code, and [pre-commit](https://pre-commit.com/) runs it, plus basic file checks, before each commit. Both are dev dependencies, so the same versions run on Linux and Windows.

```bash
uv sync                              # installs ruff and pre-commit
uv run pre-commit install            # once per clone: run the hooks on every commit
uv run pre-commit run --all-files    # run every hook by hand
uv run ruff check --fix              # lint only
uv run ruff format                   # format only
```

Ruff's settings are in `pyproject.toml`; the hooks are in `.pre-commit-config.yaml`.

## Roadmap

The full list is in [TODO.md](TODO.md). The next items include:

- Detect when the planner marks someone absent but describes them in the scene, and retry
- Glossary cases in the benchmark, so terms are measured as well as names
- Character aliases, so first names and nicknames match
- Reference images per character, so the same face and outfit carry across pages
- Several candidate images per click, with seeds you can keep
- A quality mode using the undistilled FLUX.2 Klein base model
- Illustrating pages while reading EPUB or PDF files, for example through OpenReader

## Credits and licenses

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) runs the image models. See its repository for its license.
- [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) by Black Forest Labs, Apache 2.0.
- The text encoder and VAE are repackaged by [Comfy-Org](https://huggingface.co/Comfy-Org/flux2-klein-4B), Apache 2.0.
- [Ollama](https://github.com/ollama/ollama) runs the planner model. See its repository for its license.
- The planner models are [Qwen3 and Qwen3.5](https://huggingface.co/Qwen) by Alibaba Cloud. Check the license of each model on its Ollama page.

The code in this repository is MIT licensed. See [LICENSE](LICENSE).

## Disclaimer

Not affiliated with any author or publisher. For personal use; you're responsible for the rights to any book you use.
