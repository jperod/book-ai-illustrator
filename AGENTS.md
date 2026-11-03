# Notes for coding agents

## The app server: port 8000 only

- The Book Illustrator server (`server.py`) runs on **port 8000 and nowhere else**. Test, evaluate and debug against
  `http://127.0.0.1:8000`.
- **Do not start extra instances on other ports** (8001, 8002, ...). Do not set `BOOK_ILLUSTRATOR_PORT` to dodge a
  busy port. Extra servers share the same GPU, Ollama and ComfyUI, slow every request down, and are easy to forget.
- **If port 8000 is busy, stop the server that holds it and restart it** with the current code:
  - Stop only the process running this project's `server.py`. Never stop ComfyUI (`main.py`, port 8188), Ollama (port 11434), or editor tools (such as VS Code's `lsp_server.py`).
  - Start it again: `uv run python server.py`.
- A running server keeps the code it started with. After changing `server.py`, `prompting.py` or `bible.py`, restart
  it before testing.
- When you are done, leave one server on 8000 (or none, if the user asked for that). Never leave stray servers behind.

## Other rules

- `books/<id>.json` is the only per-book data. All code stays generic: content decisions (who is in a scene, the
  setting, pronouns, style problems) are made by the LLM, never by keyword lists or regex. No book character names
  in code, prompts or examples; use invented names.
- Before committing: `uv run python -m unittest discover -s tests` and `uv run pre-commit run --all-files`.
- Commits: small logical chunks, and no AI attribution lines in commit messages.
- Real book text stays in the git-ignored `eval/local/` folder; only synthetic material is committed.
