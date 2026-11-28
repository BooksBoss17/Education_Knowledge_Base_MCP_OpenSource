/* Temporarily typeset individual text units; restore the original face at completion. */
function studioTextMotion(face, object, step, spec, frames, finished) {
  const text = object.text;
  if (
    !text ||
    step.unit === "object" ||
    !step.unit ||
    !["in", "out"].includes(spec.group)
  )
    return null;
  const pieces =
    step.unit === "paragraph"
      ? text.split(/(\n)/)
      : Array.from(
          new Intl.Segmenter("zh", {
            granularity: step.unit === "word" ? "word" : "grapheme",
          }).segment(text),
          (p) => p.segment,
        );
  const content = document.createElement("div"),
    original = [...face.childNodes];
  Object.assign(content.style, {
    height: "100%",
    fontFamily: object.font,
    fontSize: object.size + "px",
    fontWeight: object.bold ? "bold" : "normal",
    color: object.color,
    textAlign: object.align,
  });
  const fragments = [];
  // Batch unusually long pieces of text to keep the number of animated DOM nodes bounded.
  const stride = Math.max(1, Math.ceil(pieces.length / 500));
  for (let i = 0; i < pieces.length; i += stride) {
    const value = pieces.slice(i, i + stride).join("");
    if (!value.trim()) {
      content.append(document.createTextNode(value));
      continue;
    }
    const span = document.createElement("span");
    span.style.display = "inline-block";
    span.textContent = value;
    content.append(span);
    fragments.push(span);
  }
  if (!fragments.length) return null;
  face.replaceChildren(content);
  const total = step.duration * 1000,
    duration = fragments.length === 1 ? total : total * 0.45;
  const timeline = fragments.map((span, i) => {
    const delay = ((total - duration) * i) / Math.max(1, fragments.length - 1);
    const keys =
      spec.name === "Appear"
        ? spec.group === "in"
          ? [{ opacity: 0 }, { opacity: 1 }]
          : [{ opacity: 1 }, { opacity: 0 }]
        : frames;
    return span.animate(keys, {
      duration,
      delay,
      fill: "both",
      easing: "ease-out",
    });
  });
  const restore = () => {
    if (content.parentNode === face) face.replaceChildren(...original);
  };
  timeline.at(-1).onfinish = () => {
    restore();
    finished();
  };
  return () => {
    for (const a of timeline) {
      a.onfinish = null;
      a.cancel();
    }
    restore();
  };
}
