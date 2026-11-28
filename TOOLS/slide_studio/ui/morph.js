/* Match unambiguous layers, then use layout rectangles to interpolate their geometry. */
function StudioMorph(oldCanvas, canvas, oldSlide, slide, mode) {
  if (!oldCanvas || !oldSlide) return;
  const duration = 1000 * (slide.transitionDuration ?? 0.3);
  const scale =
    canvas.getBoundingClientRect().width / parseFloat(canvas.style.width);
  if (!duration || !scale) return;
  const rects = (root, data) =>
    [...root.querySelectorAll("[data-object]")].map((node) => {
      const id = node.dataset.object,
        obj = data.elements[id];
      return { id, obj, node, rect: node.getBoundingClientRect() };
    });
  const old = rects(oldCanvas, oldSlide),
    current = rects(canvas, slide);
  const animatedText = new Set();
  function tokens(layers, split) {
    const result = [];
    for (const layer of layers) {
      const face = layer.node.firstChild,
        text = face.firstChild;
      if (
        text?.nodeType !== Node.TEXT_NODE ||
        !text.length ||
        face.childNodes.length !== 1
      )
        continue;
      for (const token of split.segment(text.data)) {
        if (!token.segment.trim()) continue;
        const range = document.createRange();
        range.setStart(text, token.index);
        range.setEnd(text, token.index + token.segment.length);
        result.push({
          token: token.segment,
          rect: range.getBoundingClientRect(),
          face,
        });
      }
    }
    return result;
  }
  if (mode !== "page-MorphByObject" && typeof Intl.Segmenter === "function") {
    const split = new Intl.Segmenter("zh", {
      granularity: mode.endsWith("Word") ? "word" : "grapheme",
    });
    const origin = new Map();
    for (const t of tokens(old, split)) {
      if (!origin.has(t.token)) origin.set(t.token, []);
      origin.get(t.token).push(t);
    }
    const pieces = tokens(current, split),
      viewport = canvas.getBoundingClientRect();
    for (const t of pieces) {
      const from = origin.get(t.token)?.shift(),
        style = getComputedStyle(t.face);
      const span = document.createElement("span");
      span.textContent = t.token;
      Object.assign(span.style, {
        position: "absolute",
        pointerEvents: "none",
        whiteSpace: "pre",
        left: (t.rect.left - viewport.left) / scale + "px",
        top: (t.rect.top - viewport.top) / scale + "px",
        font: style.font,
        color: style.color,
      });
      canvas.append(span);
      const delta = from
        ? "translate(" +
          (from.rect.x - t.rect.x) / scale +
          "px," +
          (from.rect.y - t.rect.y) / scale +
          "px)"
        : "none";
      const anim = span.animate(
        [
          { transform: delta, opacity: from ? 1 : 0 },
          { transform: "none", opacity: 1 },
        ],
        { duration },
      );
      anim.onfinish = () => span.remove();
      animatedText.add(t.face);
    }
    for (const face of animatedText)
      face.animate([{ opacity: 0 }, { opacity: 0 }], { duration });
  }
  const unused = new Set(old),
    used = new Set();
  const signatures = [
    (x) => x.obj.morphKey,
    (x) => x.id,
    (x) => x.obj.type + "|" + x.obj.name,
    (x) => x.obj.asset,
    (x) => x.obj.text && x.obj.type + "|" + x.obj.text,
  ];
  for (const key of signatures) {
    const source = new Map(),
      dest = new Map();
    for (const [seq, map] of [
      [unused, source],
      [current.filter((x) => !used.has(x)), dest],
    ]) {
      for (const x of seq) {
        const k = key(x);
        if (k) map.set(k, map.has(k) ? null : x);
      }
    }
    for (const [k, to] of dest) {
      const from = source.get(k);
      if (
        !to ||
        !from ||
        to.obj.type !== from.obj.type ||
        animatedText.has(to.node.firstChild)
      )
        continue;
      unused.delete(from);
      used.add(to);
      const a = from.rect,
        b = to.rect;
      if (!a.width || !b.width || !a.height || !b.height) continue;
      from.node.style.visibility = "hidden";
      const delta = [(a.left - b.left) / scale, (a.top - b.top) / scale];
      to.node.animate(
        [
          {
            translate: delta.map((d) => d + "px").join(" "),
            scale: a.width / b.width + " " + a.height / b.height,
            transformOrigin: "top left",
          },
          { translate: "0px 0px", scale: "1 1", transformOrigin: "top left" },
        ],
        { duration, easing: "ease-in-out" },
      );
    }
  }
}
