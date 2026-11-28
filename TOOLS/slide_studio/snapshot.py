import base64
import os
from pathlib import Path
import subprocess
import tempfile
import time

from deck import asset_ids, check
from portable import ROOT
from motion import player_script
from store import dump


def snapshot(store, id, slide, step=None, reference=False, revision=None):
    if revision is None:
        deck = store.state(id)["deck"]
    else:
        check(type(revision) is int and revision >= 0, "无效的快照版本")
        with store.db() as db:
            deck = store.version(db, id, revision)
    page = deck["slides"][slide]
    if reference:
        check(page.get("reference"), "此页没有 WPS 参考图")
        with store.db() as db:
            return db.execute(
                "SELECT content FROM assets WHERE id=?", (page["reference"],)
            ).fetchone()[0]
    deck.update(order=[slide], slides={slide: page})
    assets = {}
    with store.db() as db:
        for key in asset_ids(deck):
            row = db.execute(
                "SELECT mime,content FROM assets WHERE id=?", (key,)
            ).fetchone()
            assets[key] = f"data:{row[0]};base64," + base64.b64encode(row[1]).decode()
    candidates = [
        Path(os.getenv(env, ".")) / suffix
        for env in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")
        for suffix in (
            "Google/Chrome/Application/chrome.exe",
            "Microsoft/Edge/Application/msedge.exe",
        )
    ]
    exe = next((p for p in candidates if p.is_file()), None)
    check(exe, "快照需要本机 Chrome 或 Edge")
    with tempfile.TemporaryDirectory(
        prefix="classpage-render-", ignore_cleanup_errors=True
    ) as tmp:
        root = Path(tmp)
        html = root / "page.html"
        target = root / "shot.png"
        pack = dump([deck, assets, step]).replace("<", "\\u003c")
        html.write_text(
            "<!doctype html><meta charset=utf-8><style>"
            + (ROOT / "ui/player.css").read_text("utf8")
            + "</style><main class=show-stage id=h><div id=c></div></main><script>"
            + player_script()
            + "</script><script>const [d,a,s]="
            + pack
            + ";const p=d.slides[d.order[0]];Classpage.mount(c,d,p,k=>a[k],true);Classpage.fit(c,h,d);new ResizeObserver(()=>Classpage.fit(c,h,d)).observe(h);if(s!==null)Classpage.animate(c,p,s,true,true);</script>",
            "utf8",
        )
        w = 1200
        h = round(w * deck["height"] / deck["width"])
        subprocess.run(
            [
                str(exe),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--force-device-scale-factor=1",
                "--user-data-dir=" + str(root / "profile"),
                f"--window-size={w},{h}",
                "--screenshot=" + str(target),
                "--virtual-time-budget=1500",
                html.as_uri(),
            ],
            capture_output=True,
            timeout=60,
            check=True,
        )
        for _ in range(100):
            if target.is_file():
                data = target.read_bytes()
                if data.endswith(b"IEND\xaeB`\x82"):
                    return data
            time.sleep(0.1)
        raise ValueError("浏览器未写出完整 PNG")
