"""Prompt text for one illustration: who is in the picture, the setting, the scene, and the style.

Pure functions only. This module does not decide who is in a scene: that is the planner LLM's job
(server.plan_scene). It only lays out what it is given. PROMPT_VERSION is recorded with every image; change it
whenever the layout or wording below changes, so results stay traceable to the prompt that made them.
"""

from bible import character_line, count_line, normalise_character

# m3: fixed "Scene:" wording (the word "book" invited title lettering), cast lines from the book's own fields.
PROMPT_VERSION = "m3-2026-11-03"

SINGLE_IMAGE_NOTE = "A single painting of one continuous scene filling the whole canvas, edge to edge."


def characters_in_scene(characters, narrator="", present=None):
    """Exactly the characters the planner said are visible (`present`), narrator first. With no planner
    answer (present is empty or None) nobody is listed: the app does not guess from the text."""
    if not present:
        return []
    wanted = {p.casefold() for p in present}
    chosen = [normalise_character(c) for c in characters if str(c.get("name", "")).casefold() in wanted]
    return sorted((c for c in chosen if c), key=lambda c: c["name"] != narrator)


def build_prompt(style, text, characters=(), narrator="", scene="", present=None, glossary_notes=None, setting=""):
    """The full prompt. `characters` is the book's character list; `present` is the planner's list of visible
    names. `scene` replaces the page text when given. `setting` adds a 'Setting:' line."""
    parts = [SINGLE_IMAGE_NOTE]
    chars = characters_in_scene(characters, narrator, present)
    if chars:
        parts += ["", "Characters in this scene (keep their appearance consistent):"]
        parts += [character_line(c, narrator) for c in chars]
    if glossary_notes:
        parts += ["", "Visual notes:"] + [f"- {term}: {visual}" for term, visual in glossary_notes]
    if chars:
        parts += ["", count_line(chars)]
    if setting.strip():
        parts += ["", f"Setting: {setting.strip()}"]
    parts += ["", "Scene:", scene.strip() or text.strip()]
    if style.strip():
        parts += ["", f"Visual style: {style.strip()}"]
    return "\n".join(parts)
