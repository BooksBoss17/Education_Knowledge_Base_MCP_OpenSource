"""Save the classroom delivery format using the source PNGs and playable HTML."""

from io import BytesIO
import zipfile

from portable import filename, offline
from snapshot import snapshot


def package(store, key):
    state = store.state(key)
    deck = state["deck"]
    out = BytesIO()
    name = filename(deck["title"])
    top = name[:-5] + "/"
    with zipfile.ZipFile(
        out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=2
    ) as archive:
        archive.writestr(top + name, offline(store, key))
        for i, sid in enumerate(deck["order"], 1):
            page = deck["slides"][sid]
            if page.get("reference"):
                with store.db() as db:
                    data = db.execute(
                        "SELECT content FROM assets WHERE id=?", (page["reference"],)
                    ).fetchone()[0]
            else:
                data = snapshot(store, key, sid)
            archive.writestr(top + f"images/{i:03}.png", data)
    if store.state(key)["revision"] != state["revision"]:
        raise ValueError("打包时课件已更改，请重试")
    return out.getvalue()
