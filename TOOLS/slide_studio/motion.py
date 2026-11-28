"""Shared effect catalog and assembled browser runtime for every presentation surface."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UI = ROOT / "ui"
CATALOG = json.loads((UI / "motion.json").read_text("utf8"))
NATIVE_EFFECT = {
    (entry["native"], entry["group"] == "out"): key
    for key, entry in CATALOG["effects"].items()
    if "native" in entry
}
NATIVE_TRANSITION = {v["native"]: k for k, v in CATALOG["transitions"].items()}


def playback_catalog(deck):
    """Describe only effects used by this document. Explicit step paths travel with the slide."""
    effects, transitions = {}, {}
    for slide in deck["slides"].values():
        transition = slide["transition"]
        if transition in CATALOG["transitions"]:
            transitions[transition] = {"name": CATALOG["transitions"][transition]["name"]}
        for step in slide["steps"]:
            key = step["effect"]
            spec = CATALOG["effects"].get(key)
            if spec is None:  # The player handles basic aliases and component actions directly.
                continue
            if key not in effects:
                effects[key] = {"group": spec["group"], "name": spec["name"]}
            if "path" in spec and not step.get("path"):
                effects[key]["path"] = spec["path"]
    return {"effects": effects, "transitions": transitions}


def player_script(deck=None):
    catalog = json.dumps(CATALOG if deck is None else playback_catalog(deck), ensure_ascii=True, separators=(",", ":"))
    scripts = [(UI / name).read_text("utf8") for name in ("motion.js", "morph.js", "component.js", "text-motion.js", "player.js")]
    return "\n".join([f"const StudioCatalog = {catalog};", *scripts])


def templates(store=None):
    specs = (
        ("折射实验", "refraction", {"angle": 35, "medium": "air-glass", "normal": True}, ["play", "pause", "reset"]),
        ("三维地球", "earth", {"latitude": 23.5, "longitude": 30, "axis": True, "daylight": True}, ["play", "pause", "reset"]),
    )
    return [
        dict(name=name, type="component", html=(ROOT / "templates" / (file + ".html")).read_text("utf8"),
             parameters=params, actions=actions,
             **({"poster": store.asset("image/png", (ROOT / "templates" / (file + ".png")).read_bytes())} if store else {}))
        for name, file, params, actions in specs
    ]
