# Prompt checks

`harness.py` checks the prompts the app would send, without a GPU, ComfyUI or a network connection.

    uv run python eval/harness.py

Cases are JSON files in `eval/local/cases/`. That folder is git-ignored because cases hold real
book text, so they stay on your machine. The format is described at the top of `harness.py`.
