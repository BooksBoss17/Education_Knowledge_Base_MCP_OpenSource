"""Visible path bounds, respecting PDF transparency groups and clipping."""


def visible_vector_boxes(page):
    """Yield conservative bounds of painted paths; never count clip/group records."""
    try:
        drawings = page.get_drawings(extended=True)
    except TypeError:
        # Older compatible providers do not expose the graphics-state hierarchy.
        drawings = page.get_drawings()
    stack = []
    for drawing in drawings:
        level = drawing.get("level", 0)
        while stack and stack[-1][0] >= level:
            stack.pop()
        kind = drawing.get("type")
        if kind == "group":
            stack.append((level, drawing.get("opacity", 1.0) == 0, None))
            continue
        if kind == "clip":
            stack.append((level, False, drawing.get("scissor")))
            continue
        if any(hidden for _, hidden, _ in stack):
            continue
        if kind in {"f", "s", "fs"}:
            fill = "f" in kind and drawing.get("fill_opacity", 1.0) != 0
            stroke = "s" in kind and drawing.get("stroke_opacity", 1.0) != 0
            if not (fill or stroke):
                continue
        rect = drawing.get("rect")
        if rect is None:
            continue
        box = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
        for clip in [page.rect, *(clip for _, _, clip in stack if clip is not None)]:
            box = [max(box[0], clip.x0), max(box[1], clip.y0),
                   min(box[2], clip.x1), min(box[3], clip.y1)]
        if box[0] <= box[2] and box[1] <= box[3]:
            yield box
