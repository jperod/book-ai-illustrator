"""File storage for books, settings, and output sidecar metadata."""

import json
import os
import re
import threading

import config
from bible import lint_book, normalise_book

BOOK_ID_RE = re.compile(r"^[a-z0-9-]+$")

BOOK_WRITE_LOCK = threading.Lock()
SETTINGS_LOCK = threading.Lock()


class BookFileError(Exception):
    """A book file exists but cannot be read as a book."""


EMPTY_PLAN = {"scene": "", "present": [], "setting": "", "terms": []}


def plan_from_request(plan):
    """A plan in the shape plan_scene returns ({scene, present, setting, terms}), checked. None if not a dict.
    Used for eval variants and for a re-roll's saved plan. An empty scene means the page text is drawn."""
    if not isinstance(plan, dict):
        return None
    present = plan.get("present")
    terms = plan.get("terms")
    return {
        "scene": str(plan.get("scene") or "").strip(),
        "present": [str(p) for p in present] if isinstance(present, list) else [],
        "setting": str(plan.get("setting") or ""),
        "terms": [str(t) for t in terms] if isinstance(terms, list) else [],
    }


def write_json_atomic(path, obj, lock=None):
    """Writes JSON to a temp file, then replaces the target, so a crash never leaves half a file."""
    tmp = path + ".tmp"
    with lock or threading.Lock():
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)


def book_file(book_id):
    if not BOOK_ID_RE.fullmatch(book_id):
        raise ValueError(f"bad book id: {book_id!r}")
    return os.path.join(config.BOOKS_DIR, book_id + ".json")


def load_book(book_id):
    """The book, normalised to the current shape (see bible.normalise_book). Everything downstream relies on it."""
    path = book_file(book_id)
    if not os.path.exists(path):
        raise FileNotFoundError(book_id)
    try:
        with open(path, encoding="utf-8") as f:
            book = json.load(f)
    except (OSError, ValueError) as e:
        raise BookFileError(f"{book_id}.json could not be read: {e}") from e
    if not isinstance(book, dict):
        raise BookFileError(f"{book_id}.json does not contain a JSON object")
    # The file name is the book's id, so a copied file cannot take over another book.
    book["id"] = book_id
    return normalise_book(book)


def save_book(book):
    """Writes books/<id>.json atomically."""
    os.makedirs(config.BOOKS_DIR, exist_ok=True)
    write_json_atomic(book_file(book["id"]), book, BOOK_WRITE_LOCK)


def list_books():
    books = []
    if os.path.isdir(config.BOOKS_DIR):
        for fn in os.listdir(config.BOOKS_DIR):
            stem = fn[: -len(".json")] if fn.endswith(".json") else ""
            if not BOOK_ID_RE.fullmatch(stem):
                continue
            try:
                with open(os.path.join(config.BOOKS_DIR, fn), encoding="utf-8") as f:
                    b = json.load(f)
                books.append(
                    {
                        "id": stem,
                        "title": str(b.get("title", stem)),
                        "author": str(b.get("author", "")),
                        "series": str(b.get("series", "")),
                        "book_number": int(b.get("book_number", 0) or 0),
                    }
                )
            except (OSError, ValueError, TypeError, AttributeError) as e:
                print(f"Skipping unreadable book file {fn}: {e}")
    return sorted(books, key=lambda b: (b["series"], b["book_number"], b["title"]))


def book_view(book):
    """What the page needs: the book's fields, characters, glossary and its structural problems."""
    return {
        "id": book["id"],
        "title": book.get("title", book["id"]),
        "author": book.get("author", ""),
        "series": book.get("series", ""),
        "book_number": book.get("book_number", 0),
        "visual_style": book.get("visual_style", ""),
        "narrator": book.get("narrator", ""),
        "characters": book["characters"],
        "glossary": book["glossary"],
        "problems": lint_book(book),
    }


def update_book(book_id, fields):
    """Applies whichever of visual_style, narrator, characters (a list) and glossary (a list) are present,
    saves, and returns the book."""
    book = load_book(book_id)
    if "visual_style" in fields:
        book["visual_style"] = str(fields["visual_style"])
    if isinstance(fields.get("characters"), list):
        book["characters"] = fields["characters"]
    if isinstance(fields.get("glossary"), list):
        book["glossary"] = fields["glossary"]
    if "narrator" in fields:
        book["narrator"] = str(fields["narrator"]).strip()
    book = normalise_book(book)
    save_book(book)
    return book


def read_settings():
    """Settings saved on this machine. A missing or unreadable file means the defaults."""
    try:
        with open(config.SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        print(f"Ignoring unreadable settings file: {e}")
        return {}
    return data if isinstance(data, dict) else {}


def write_settings(settings):
    write_json_atomic(config.SETTINGS_PATH, settings, SETTINGS_LOCK)


def sidecar_path(image_name):
    """outputs/<image>.json for an image this app saved, or None for any other name."""
    name = os.path.basename(str(image_name or ""))
    if name != image_name or not name.endswith(".png"):
        return None
    return os.path.join(config.OUTPUT_DIR, name[: -len(".png")] + ".json")


def read_saved_plan(image_name):
    """The plan saved with an earlier image (its sidecar), so a re-roll draws the same moment."""
    path = sidecar_path(image_name)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return plan_from_request(json.load(f).get("plan"))
    except (OSError, ValueError, AttributeError):
        return None


def write_sidecar(image_name, record):
    """Writes outputs/<image>.json next to the picture: what made it, so any image can be traced or re-made."""
    write_json_atomic(sidecar_path(image_name), record)
