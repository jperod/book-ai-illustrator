"""Prompts, few-shot synthetic examples, and JSON schemas for LLM scene planning and reviews.

All content judgements are made by the LLM: there are no word lists or text patterns in this code.
Every example uses invented names (no real book characters).
"""

PLANNER_SYSTEM = (
    "You plan one illustration for a page of a novel. Choose ONE moment from the page: the most visual one. "
    "You get the book's cast. For every cast member the page refers to in any way (by name, by another name, "
    'by "he" or "she", or as the narrator "I"), write a "checks" entry with "visible_in_moment": true only if '
    "that person or animal is physically in the picture in that moment. Someone who is only remembered, "
    "mentioned, avoided, absent, dead or spoken about is false. Leave out cast members the page does not refer "
    'to. Copy every name exactly as written in the cast. "terms": the glossary terms whose physical object, '
    "place or visual effect is physically in the picture in that moment. Something only spoken about in "
    "dialogue, remembered, or happening elsewhere is false and must not be included. "
    '"setting": "place" (a short phrase the page gives, or "unknown"), "indoors" ("yes", "no", or '
    '"unknown" when the page does not say) and "time" (for example "dusk", or "unknown"); do not guess a place '
    'the page does not give. "scene": at most 50 words of concrete visible detail: poses, setting, which hand '
    'holds each prop. Refer to people by name, never as "I". Only the visible cast appears in the scene and no '
    "one else. Each visible person has a distinct pose facing into the room or in profile, not obscured or "
    "turned away. When no cast members are on the page (such as landscape, chapter introductions, or "
    "worldbuilding), describe the environment, landscape, architecture, weather, and atmosphere in vivid concrete "
    'detail; do not write "no characters are visible". When two people are visible, write one sentence per '
    "person and name who does what to whom; do not use a pronoun that could mean either of them. No thoughts or "
    "backstory."
)

# Synthetic examples with invented names, shown before the real request.
PLANNER_FEW_SHOT = [
    (
        'Cast:\n- Lira (narrator, "I"; female, young adult)\n- Orrin (male, adult)\n- Bel (female, adult)\n'
        "- Corran (male, adult)\n- Tess (female, child)\nGlossary terms: Sun-Banner, Sea-Raiders\n"
        'Page:\n"Watch for Sea-Raiders," Orrin warned as I waited beneath the scarlet Sun-Banner at the gate while '
        "he argued with the guard. Bel had left us two days ago, and I still missed her voice. Corran had ridden "
        "off to the coast at dawn. Orrin finally stepped aside, laughing at something the guard had said.",
        {
            "checks": [
                {"name": "Lira", "visible_in_moment": True},
                {"name": "Orrin", "visible_in_moment": True},
                {"name": "Bel", "visible_in_moment": False},
                {"name": "Corran", "visible_in_moment": False},
            ],
            "terms": ["Sun-Banner"],
            "setting": {"place": "a stone gate by a road", "indoors": "no", "time": "unknown"},
            "scene": (
                "Lira waits beneath a scarlet Sun-Banner by a stone gate. "
                "Orrin, arms folded, stands beside Lira, laughing."
            ),
        },
    ),
    (
        'Cast:\n- Lira (narrator, "I"; female, young adult)\n- Orrin (male, adult)\n'
        "Glossary terms: Sun-Reach\n"
        "Page:\nChapter 4. The Sun-Reach was a vast expanse of golden plains stretching between the "
        "granite peaks of the north and the salt marshes of the south. Small turf-roofed cottages clustered "
        "along the riverbanks, where herds of wild goats grazed beneath the noon sun.",
        {
            "checks": [
                {"name": "Lira", "visible_in_moment": False},
                {"name": "Orrin", "visible_in_moment": False},
            ],
            "terms": ["Sun-Reach"],
            "setting": {"place": "golden plains beneath granite peaks", "indoors": "no", "time": "noon"},
            "scene": (
                "Vast golden plains roll toward distant granite mountain peaks under a noon sky. "
                "Small turf-roofed cottages cluster along a wide winding river where wild goats graze along the bank."
            ),
        },
    ),
    (
        'Cast:\n- Ren (narrator, "I"; male, teen)\n- Ama (female, adult)\n- Koll (wolf)\nGlossary terms: none\n'
        "Page:\nI lay by the fire, too weak to sit up. Ama knelt beside me and held the water skin to my lips.",
        {
            "checks": [
                {"name": "Ren", "visible_in_moment": True},
                {"name": "Ama", "visible_in_moment": True},
            ],
            "terms": [],
            "setting": {"place": "beside a campfire", "indoors": "unknown", "time": "unknown"},
            "scene": (
                "Ren lies on a blanket beside a fire, head on a rolled cloak. "
                "Ama kneels beside Ren and holds a water skin to Ren's lips."
            ),
        },
    ),
]

SETTING_SCHEMA = {
    "type": "object",
    "properties": {
        "place": {"type": "string"},
        "indoors": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "time": {"type": "string"},
    },
    "required": ["place", "indoors", "time"],
}

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "visible_in_moment": {"type": "boolean"}},
                "required": ["name", "visible_in_moment"],
            },
        },
        "terms": {"type": "array", "items": {"type": "string"}},
        "setting": SETTING_SCHEMA,
        "scene": {"type": "string"},
    },
    "required": ["checks", "terms", "setting", "scene"],
}

REVIEW_SYSTEM = (
    "You check a short scene description that will be drawn as one picture. Rules: (1) every visible person "
    "is named and does something (if Visible is 'nobody', the scene is an unpopulated landscape or establishing "
    "shot, so no people are required); (2) no other person, figure or crowd appears; (3) no pronoun could mean "
    "either of two visible people: write the name instead; (4) nothing contradicts the setting (no indoor "
    "objects outdoors, no sky or landscape indoors); (5) each person has one pose; (6) at most 60 words, "
    'complete sentences. Answer "ok": true if all rules hold. Otherwise list the "problems" and give a fixed '
    '"scene" that keeps the same moment and follows every rule. If the scene is already fine, return an '
    'empty "scene".'
)

REVIEW_FEW_SHOT = [
    (
        "Visible: Ren, Ama\nSetting: beside a campfire\n"
        "Scene: Ren lies by the fire. Ama kneels and holds the water skin to his lips while a stranger watches.",
        {
            "ok": False,
            "problems": ["'his' could mean Ren or Ama", "a stranger is not visible in this moment"],
            "scene": "Ren lies by the fire. Ama kneels and holds the water skin to Ren's lips.",
        },
    ),
    (
        "Visible: nobody\nSetting: golden plains beneath granite peaks\n"
        "Scene: Vast golden plains roll toward distant granite mountain peaks. "
        "Turf-roofed cottages cluster along a river.",
        {
            "ok": True,
            "problems": [],
            "scene": "",
        },
    ),
]

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "scene": {"type": "string"},
    },
    "required": ["ok", "problems", "scene"],
}

BOOK_REVIEW_SYSTEM = (
    "You review the setup of a book for an illustration app. The JSON gives the visual style, the cast and the "
    "glossary. Find problems that would make pictures wrong: a look that contradicts the character's sex, age "
    "or kind; a character described only by reference to another character; a style text that names places, "
    "objects, light sources or scene content (it pulls every picture toward them) or words that invite "
    "lettering such as cover, poster or title; a glossary visual note that describes a named character; "
    "missing details the pictures need. Report only real problems, each with the field it is about."
)

BOOK_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "problems": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "enum": ["error", "warning"]},
                    "field": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["level", "field", "message"],
            },
        }
    },
    "required": ["problems"],
}
