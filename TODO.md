# TODO

Ideas and open work for Book AI Illustrator, grouped by how soon they matter.

## Next: image accuracy
- [ ] **LLM scene extraction.** Turn page prose into one short, visible scene before generating (Claude API, Haiku-class, under a cent per page; fall back to the plain page if it fails). Needs an API-key setting read from the environment, a privacy note in the README, and a `provider: off` default.
- [ ] **Stronger age wording.** Fitz's description says "young man around fifteen", and the model still draws an adult. Try "teenage boy, about fifteen, slim, beardless" and check with the harness.
- [ ] **Clean up character descriptions.** Remove "weathered" and "slight limp" from Burrich, and "about Burrich's age" from Galen, so no description depends on another character or invites an old look. Needs your wording approval.
- [ ] **Several candidates per click.** Generate 2–4 seeds, show thumbnails, pick one, re-roll a seed, and keep a seed you like. The cheapest fix for glitches like the extra leg.
- [ ] **Return and store the seed per image.** Already returned by the API. Store it next to the image so results can be reproduced.
- [ ] **Online evaluation mode for the harness.** Run each case with variants (scene vs page, 4 vs 8 steps, resolution) and write a contact-sheet HTML page to `eval/local/out/`.
- [ ] **Tune resolution.** Compare 1024 and 1280 squares, staying at or below 1536.

## Later: consistency across pages
- [ ] **Reference images per character.** The installed FLUX.2 Klein fp8 model already supports reference editing (`ReferenceLatent`). Store one reference per character, upload it to ComfyUI's `/upload/image`, and use at most two references per image. Measure VRAM first; each reference adds a full image's worth of tokens.
- [ ] **Bootstrap references.** Let the user save a chosen candidate as a character's reference.
- [ ] **Base-model quality mode.** Use FLUX.2 Klein base 4B (20 steps, CFG 5, real negative prompt). Slower, and the model isn't installed yet.
- [ ] **Upscaling** as a final step, after the image is accepted.
- [ ] **Character aliases.** Match first names and nicknames (currently case-sensitive whole-word names only).
- [ ] **Narrator override per request** for books with several viewpoints.

## Bigger ideas
- [ ] **Integrate with OpenReader.** Illustrate pages while reading an EPUB or PDF: pick the page, send the text, show the image beside it. Needs a look at how OpenReader exposes page text and whether it can call a local service.
- [ ] **Save prompts and scenes per page.** So a page can be regenerated later without retyping.
- [ ] **Strip the prompt from saved PNGs.** ComfyUI writes the prompt into each PNG's metadata, so shared images carry the book text.

## Housekeeping and decisions for you
- [ ] **Public repo content.** `books/assassins-apprentice.json` (a novel's character list) and your email in the commit history are public. Removing either from history needs a rewrite and force-push.
- [ ] **Browser check.** The sidebar, Narrator picker, Scene box and "Prompt used" panel have not been checked in a browser.
- [ ] **Linux GPU run.** The test suite passes on Linux, but a full generation has only run on Windows.
- [ ] **Commit the current work** once you've reviewed it.

## Deliberately not planned
- **LoRA training.** Needs about 24 GB for a 4B model, so it's out of reach on an 8 GB card.
- **Windows system-file ignores** (`Thumbs.db`, `desktop.ini`). Left out because the repo is Linux-only.
