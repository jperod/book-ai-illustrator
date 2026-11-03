"""Book bible: the structured facts about a book's characters, read from books/<id>.json.

Pure functions only, with no I/O and no globals. This module checks structure (required fields, allowed
values, duplicate names) and renders the book's own fields into prompt lines. It makes no judgement about
content: there are no word lists here. Content problems (a look that contradicts the age, scene objects in the
style) are found by the LLM book review in server.py.

A character is a dict:
  name     the name the page uses. Unique in the book, aliases included.
  aliases  other names the text uses for the same character.
  kind     "person" (default), "animal" or "creature".
  sex      "male", "female" or "unspecified". People only.
  age      "child", "teen", "young adult", "adult", "middle-aged" or "elderly". People only.
  species  for animals and creatures, e.g. "wolf".
  look     appearance details: build, hair, clothes, role.

Old files (schema 1) hold only name and description. They still load: description becomes look.
"""

SCHEMA_VERSION = 2
KINDS = ("person", "animal", "creature")
SEXES = ("male", "female", "unspecified")
AGES = ("child", "teen", "young adult", "adult", "middle-aged", "elderly")


def _text(value):
    return "" if value is None else str(value).strip()


def normalise_character(raw):
    """The character in the current shape, or None if it has no usable name."""
    if not isinstance(raw, dict):
        return None
    name = _text(raw.get("name"))
    if not name:
        return None
    aliases = raw.get("aliases") if isinstance(raw.get("aliases"), list) else []
    return {
        "name": name,
        "aliases": [_text(a) for a in aliases if _text(a)],
        "kind": _text(raw.get("kind")).lower() or "person",
        "sex": _text(raw.get("sex")).lower(),
        "age": _text(raw.get("age")).lower(),
        "species": _text(raw.get("species")).lower(),
        "look": _text(raw.get("look", raw.get("description"))),
    }


def as_characters(value):
    """A list of character dicts in the current shape; entries without a name are dropped."""
    return [c for c in (normalise_character(item) for item in (value or [])) if c]


def normalise_glossary(items):
    out = []
    for g in items if isinstance(items, list) else []:
        if not isinstance(g, dict) or not _text(g.get("term")):
            continue
        out.append({"term": _text(g.get("term")), "meaning": _text(g.get("meaning")), "visual": _text(g.get("visual"))})
    return out


def normalise_book(book):
    """The book in the current shape. Other keys are kept as they are."""
    out = dict(book)
    out["schema_version"] = SCHEMA_VERSION
    out["characters"] = as_characters(book.get("characters", []))
    out["glossary"] = normalise_glossary(book.get("glossary", []))
    return out


def facts(character, narrator=""):
    """The book's own fields for this character, as written: 'narrator; male, teen' or 'wolf'."""
    parts = []
    if narrator and character["name"] == narrator:
        parts.append("narrator")
    if character.get("kind", "person") == "person":
        details = ", ".join(v for v in (character.get("sex", ""), character.get("age", "")) if v)
    else:
        details = character.get("species") or character.get("kind", "")
    if details:
        parts.append(details)
    return "; ".join(parts)


def character_line(character, narrator=""):
    """'- Ren (narrator; male, teen): dark hair, plain wool shirt'. Fields are copied, not translated."""
    f = facts(character, narrator)
    head = f"- {character['name']} ({f})" if f else f"- {character['name']}"
    look = character.get("look", "")
    return f"{head}: {look}" if look else head


def count_line(chars):
    """'Figures in the picture, exactly 2: Ren, Ama.' Empty when there are no characters."""
    if not chars:
        return ""
    return f"Figures in the picture, exactly {len(chars)}: " + ", ".join(c["name"] for c in chars) + "."


def lint_book(book):
    """Structural problems, as [{"level": "error" | "warning", "message": str}]: missing or invalid fields,
    names used twice, a narrator who is not a character. Content is reviewed by the LLM, not here."""
    problems = []

    def add(level, message):
        problems.append({"level": level, "message": message})

    raw = book.get("characters", [])
    if not isinstance(raw, list):
        add("error", "characters must be a list")
        raw = []
    chars = []
    for index, item in enumerate(raw, start=1):
        c = normalise_character(item)
        if c is None:
            add("error", f"character {index} has no name")
        else:
            chars.append(c)

    owner = {}  # lower-cased name or alias -> the character it belongs to
    for c in chars:
        name = c["name"]
        if c["kind"] not in KINDS:
            add("error", f"{name}: kind must be one of {', '.join(KINDS)}")
        if c["sex"] and c["sex"] not in SEXES:
            add("error", f"{name}: sex must be one of {', '.join(SEXES)}")
        if c["age"] and c["age"] not in AGES:
            add("error", f"{name}: age must be one of {', '.join(AGES)}")
        for label in [name] + c["aliases"]:
            key = label.casefold()
            if key in owner and owner[key] != name:
                add("error", f"{label!r} is used by both {owner[key]} and {name}")
            owner.setdefault(key, name)
        if c["kind"] == "person":
            if not c["sex"]:
                add("error", f"{name}: set sex (male, female or unspecified)")
            if not c["age"]:
                add("error", f"{name}: set age (one of: {', '.join(AGES)})")
        elif not c["species"]:
            add("warning", f"{name}: no species set")
        if not c["look"]:
            add("warning", f"{name}: no look, so the picture has little to go on")

    narrator = _text(book.get("narrator"))
    if narrator and narrator not in {c["name"] for c in chars}:
        add("error", f"narrator {narrator!r} is not in the characters list")

    for index, g in enumerate(book.get("glossary", []) or [], start=1):
        term = _text(g.get("term")) if isinstance(g, dict) else ""
        if not term:
            add("error", f"glossary entry {index} has no term")
            continue
        if not _text(g.get("meaning")):
            add("warning", f"{term}: no meaning, so the planner does not know it")
        if not _text(g.get("visual")):
            add("warning", f"{term}: no visual note, so the picture does not show it")
    return problems
