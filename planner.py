"""Scene planner and reviewer using a local LLM through Ollama.

The planner reads the page with the book's whole cast and glossary, and decides who is visible in one
moment, which glossary terms it shows, where it happens, and writes a short visual scene. A second short
call (the reviewer) checks that scene and fixes it. All content judgements are made by the LLM: there are
no word lists or text patterns in this code. Every example uses invented names. It runs only when
llm_enabled is true; any failure gives None, and the app uses the pasted page with no character list.
"""

import contextlib
import json
import time
import traceback
import urllib.request

import config
from bible import facts
from comfy import free_comfy
from config import add_stage, log
from planner_prompts import (
    BOOK_REVIEW_SCHEMA,
    BOOK_REVIEW_SYSTEM,
    PLANNER_FEW_SHOT,
    PLANNER_SCHEMA,
    PLANNER_SYSTEM,
    REVIEW_FEW_SHOT,
    REVIEW_SCHEMA,
    REVIEW_SYSTEM,
    SETTING_SCHEMA,
)
from storage import EMPTY_PLAN, plan_from_request, read_saved_plan

__all__ = [
    "BOOK_REVIEW_SCHEMA",
    "BOOK_REVIEW_SYSTEM",
    "EMPTY_PLAN",
    "NeedsSceneError",
    "PLANNER_FEW_SHOT",
    "PLANNER_SCHEMA",
    "PLANNER_SYSTEM",
    "PlannerError",
    "REVIEW_FEW_SHOT",
    "REVIEW_SCHEMA",
    "REVIEW_SYSTEM",
    "SETTING_SCHEMA",
    "cast_text",
    "decide_scene",
    "free_comfy_gpu",
    "llm_json",
    "llm_release",
    "ollama_post",
    "plan_from_request",
    "plan_scene",
    "planner_schema",
    "review_book",
    "review_scene",
    "setting_line",
]


class PlannerError(Exception):
    """The planner LLM gave no usable answer."""


class NeedsSceneError(Exception):
    """The planner is off and no scene was typed, so there is nothing sensible to draw."""


def ollama_post(path, payload, timeout):
    req = urllib.request.Request(
        config.CONFIG["llm_url"].rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def llm_release():
    """Unload the planner model so its VRAM is free for ComfyUI."""
    if not config.CONFIG.get("llm_enabled"):
        return
    with contextlib.suppress(Exception):
        ollama_post("/api/generate", {"model": config.CONFIG["llm_model"], "keep_alive": 0}, timeout=30)


def free_comfy_gpu():
    """Ask ComfyUI to unload its models, so the planner has room on the GPU."""
    free_comfy()


def setting_line(setting):
    """'open road, outdoors, daytime' from the planner's setting object. Unknown parts are left out."""
    if not isinstance(setting, dict):
        return ""
    parts = [str(setting.get("place", "")).strip()]
    if setting.get("indoors") == "yes":
        parts.append("indoors")
    elif setting.get("indoors") == "no":
        parts.append("outdoors")
    parts.append(str(setting.get("time", "")).strip())
    return ", ".join(p for p in parts if p and p.lower() != "unknown")


def context_size(prompt_chars, num_predict):
    """A context window that holds the request (about 3 characters per token) and the answer."""
    need = prompt_chars // 3 + num_predict + 256
    return next((size for size in (4096, 8192, 16384) if need <= size), 32768)


def llm_json(system, examples, user, schema, num_predict):
    """One chat call that must answer with JSON matching `schema`. Examples are (user, answer) pairs.
    The context window is sized from the request. An answer cut off by the token budget (unparseable JSON)
    is retried once with double the budget. Returns (raw Ollama reply, parsed object, seconds)."""
    messages = [{"role": "system", "content": system}]
    for example_user, example_answer in examples:
        messages.append({"role": "user", "content": example_user})
        messages.append({"role": "assistant", "content": json.dumps(example_answer)})
    messages.append({"role": "user", "content": user})
    prompt_chars = sum(len(m["content"]) for m in messages) + len(json.dumps(schema))
    t0 = time.perf_counter()
    for attempt in (1, 2):
        data = ollama_post(
            "/api/chat",
            {
                "model": config.CONFIG["llm_model"],
                "stream": False,
                "think": False,
                "keep_alive": config.CONFIG["llm_keep_alive"],
                "messages": messages,
                "format": schema,
                "options": {
                    "temperature": 0,
                    "num_ctx": context_size(prompt_chars, num_predict),
                    "num_predict": num_predict,
                    "num_gpu": config.CONFIG["llm_num_gpu"],
                },
            },
            timeout=config.CONFIG["llm_timeout"],
        )
        try:
            obj = json.loads(data["message"]["content"])
            break
        except ValueError as e:
            if attempt == 2:
                raise PlannerError(
                    f"the model's answer was not complete JSON even with {num_predict} tokens "
                    f"(done_reason: {data.get('done_reason')}): {e}"
                ) from e
            log(
                "WARN",
                "LLM answer cut off; retrying with a larger budget",
                num_predict=num_predict,
                done_reason=data.get("done_reason"),
                error=str(e),
            )
            num_predict *= 2
    seconds = round(time.perf_counter() - t0, 2)
    return data, obj, seconds


def planner_schema(names, terms):
    """PLANNER_SCHEMA with the book's own names and glossary terms as the only allowed values, so the
    model cannot invent or misspell a name."""
    schema = json.loads(json.dumps(PLANNER_SCHEMA))
    if names:
        schema["properties"]["checks"]["items"]["properties"]["name"]["enum"] = names
    if terms:
        schema["properties"]["terms"]["items"]["enum"] = terms
    return schema


def cast_text(characters, narrator, glossary):
    """The cast and glossary terms as the planner reads them, from the book's own fields (bible.facts)."""
    lines = ["Cast:"]
    for c in characters:
        tags = ['narrator, "I"'] if narrator and c["name"] == narrator else []
        tags += [t for t in (facts(c), "also called " + ", ".join(c["aliases"]) if c["aliases"] else "") if t]
        lines.append(f"- {c['name']} ({'; '.join(tags)})" if tags else f"- {c['name']}")
    if len(lines) == 1:
        lines.append("- none")
    terms = [g["term"] for g in glossary]
    lines.append("Glossary terms: " + (", ".join(terms) if terms else "none"))
    return "\n".join(lines)


def review_scene(scene, present, setting):
    """The reviewer call. Returns (ok, problems, fixed scene or "", seconds)."""
    user = (
        f"Visible: {', '.join(present) if present else 'nobody'}\n"
        f"Setting: {setting_line(setting) or 'unknown'}\nScene: {scene}"
    )
    _, obj, seconds = llm_json(REVIEW_SYSTEM, REVIEW_FEW_SHOT, user, REVIEW_SCHEMA, num_predict=600)
    problems = [str(p) for p in obj.get("problems", []) if str(p).strip()]
    return bool(obj.get("ok")), problems, str(obj.get("scene", "")).strip(), seconds


def plan_scene(text, characters, narrator, glossary):
    """Returns {"present", "scene", "setting", "terms", "seconds", "timings", "review"}, or None. Never raises.
    `characters` and `glossary` come from load_book (already normalised). With planner_validate on, the
    reviewer checks the scene and its fixed scene is used; with no usable fix the planner's scene is kept.
    Returns None when the planner is off. Raises PlannerError when the LLM gives no usable answer: the raw
    page is never drawn in its place."""
    if not config.CONFIG.get("llm_enabled"):
        return None
    try:
        alias_of = {}
        for c in characters:
            for label in [c["name"]] + c["aliases"]:
                alias_of[label] = c["name"]
        term_names = [g["term"] for g in glossary]
        system = PLANNER_SYSTEM
        meanings = [g for g in glossary if g.get("meaning")]
        if meanings:
            system += "\n\nGlossary (use these meanings, do not invent others):\n" + "\n".join(
                f"- {g['term']}: {g['meaning']}" for g in meanings
            )
        t_free = time.perf_counter()
        if config.CONFIG["llm_num_gpu"] > 0:
            free_comfy_gpu()
        free_s = time.perf_counter() - t_free
        data, obj, seconds = llm_json(
            system,
            PLANNER_FEW_SHOT,
            cast_text(characters, narrator, glossary) + "\nPage:\n" + text,
            planner_schema(list(alias_of), term_names),
            num_predict=400 + 30 * len(characters),  # a check per cast member, plus setting and scene
        )
        checks, scene = obj.get("checks"), obj.get("scene")
        if (
            not isinstance(checks, list)
            or not isinstance(scene, str)
            or not scene.strip()
            or not all(isinstance(c, dict) and isinstance(c.get("visible_in_moment"), bool) for c in checks)
        ):
            raise PlannerError("the planner's answer had the wrong shape")
        present = []
        for c in checks:
            name = alias_of.get(c.get("name"))
            if c["visible_in_moment"] and name and name not in present:
                present.append(name)
        terms = [t for t in term_names if t in (obj.get("terms") or [])]
        setting = obj.get("setting") if isinstance(obj.get("setting"), dict) else {}
        scene = scene.strip()
        review = {"used": False}
        review_s = 0.0
        if config.CONFIG.get("planner_validate"):
            t_review = time.perf_counter()
            try:
                ok, problems, fixed, review_s = review_scene(scene, present, setting)
            except Exception as e:  # a failed review keeps the planner's scene rather than losing the plan
                review_s = round(time.perf_counter() - t_review, 2)
                log("WARN", "scene review failed; keeping the planner's scene", error=str(e))
                ok, problems, fixed = True, [], ""
            review = {"used": True, "ok": ok, "problems": problems}
            if not ok and fixed:
                scene = fixed  # with no usable fix, the planner's own scene is kept
            log("INFO", "scene review", ok=ok, problems=problems, fixed=bool(fixed), seconds=review_s)
        timings = {
            "free_comfy": round(free_s, 2),
            "load": round(data.get("load_duration", 0) / 1e9, 2),
            "prompt": round(data.get("prompt_eval_duration", 0) / 1e9, 2),
            "generate": round(data.get("eval_duration", 0) / 1e9, 2),
            "wall": seconds,
            "review": review_s,
        }
        log("INFO", "planner done", present=present, terms=terms, setting=setting_line(setting), **timings)
        return {
            "present": present,
            "scene": " ".join(scene.split()[:60]),
            "setting": setting_line(setting),
            "terms": terms,
            "seconds": round(seconds + review_s, 2),
            "timings": timings,
            "review": review,
        }
    except PlannerError as e:
        log("ERROR", "planner failed", error=str(e))
        raise
    except Exception as e:
        log("ERROR", "planner failed", error=str(e), traceback=traceback.format_exc())
        raise PlannerError(str(e)) from e


def review_book(book):
    """Asks the LLM to read the book's setup and list content problems. Returns [{level, message}]."""
    payload = {
        "visual_style": book.get("visual_style", ""),
        "narrator": book.get("narrator", ""),
        "characters": book["characters"],
        "glossary": book["glossary"],
    }
    _, obj, _ = llm_json(
        BOOK_REVIEW_SYSTEM, [], json.dumps(payload, ensure_ascii=False), BOOK_REVIEW_SCHEMA, num_predict=900
    )
    out = []
    for p in obj.get("problems", []):
        if isinstance(p, dict) and str(p.get("message", "")).strip():
            level = p.get("level") if p.get("level") in ("error", "warning") else "warning"
            field = str(p.get("field", "")).strip()
            out.append(
                {
                    "level": level,
                    "message": (f"{field}: " if field else "") + str(p["message"]).strip(),
                    "source": "review",
                }
            )
    return out


def decide_scene(body, text, book, characters, timeline):
    """What to draw: (plan, source, planner info). Sources, in order: an eval variant, the saved plan of an
    earlier image (a re-roll), or the planner; a scene typed by the user replaces only the planner's scene.
    The raw page is never drawn: with no typed scene, a planner failure raises PlannerError and a planner that
    is off raises NeedsSceneError."""
    variant = body.get("variant")
    if isinstance(variant, dict) and ("scene" in variant or "present" in variant):
        return plan_from_request(variant), "variant", {"used": False}
    if body.get("from_image"):
        saved = read_saved_plan(body["from_image"])
        if saved:
            return saved, "saved", {"used": False}
    typed = str(body.get("scene", "")).strip()
    llm = f"{config.CONFIG['llm_model']} ({'GPU' if config.CONFIG['llm_num_gpu'] > 0 else 'CPU'})"
    t = time.perf_counter()
    try:
        planned = plan_scene(text, characters, book.get("narrator", ""), book["glossary"])
    except PlannerError:
        add_stage(timeline, "planner failed", time.perf_counter() - t, kind="llm", model=llm)
        if not typed:
            raise
        planned = None  # the typed scene is drawn as typed, with no character list
    wall = time.perf_counter() - t
    if planned:
        tm = planned["timings"]
        add_stage(timeline, "unload ComfyUI models for the LLM", tm["free_comfy"], kind="image", model="ComfyUI")
        add_stage(timeline, "load model", tm["load"], kind="llm", model=llm)
        add_stage(timeline, "read page and write scene", tm["prompt"] + tm["generate"], kind="llm", model=llm)
        if tm["review"]:
            add_stage(timeline, "review scene", tm["review"], kind="llm", model=llm)
        add_stage(timeline, "HTTP and parsing", wall - tm["wall"] - tm["review"], kind="llm", model=llm)
    elif not typed:
        raise NeedsSceneError("The scene planner is off (llm_enabled in config.json), so type a scene to draw.")
    info = {
        "used": bool(planned),
        "called": config.CONFIG.get("llm_enabled", False),
        "seconds": planned["seconds"] if planned else None,
        "review": planned["review"] if planned else {"used": False},
    }
    plan = plan_from_request(planned) if planned else dict(EMPTY_PLAN)
    if typed:
        plan["scene"] = typed
        return plan, "typed", info
    return plan, "planner", info
