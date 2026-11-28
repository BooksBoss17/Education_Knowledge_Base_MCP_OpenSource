/* Shared by the editor, presenter and exported offline document. */
window.Classpage = (() => {
  const el = (tag, cls) =>
    Object.assign(document.createElement(tag), { className: cls || "" });
  const initialStyles = new WeakMap();
  function formula(text) {
    if (!text.trimStart().startsWith("<math")) return null;
    const tree = new DOMParser().parseFromString(text, "application/xml");
    const allowed = new Set(
      "math mrow mi mn mo mtext mfrac msqrt mroot msub msup msubsup mover munder munderover mtable mtr mtd mspace".split(
        " ",
      ),
    );
    const copy = (node) => {
      if (node.nodeType === 3) return document.createTextNode(node.textContent);
      if (!allowed.has(node.localName)) throw Error("Unsupported math");
      const n = document.createElementNS(
        "http://www.w3.org/1998/Math/MathML",
        node.localName,
      );
      for (const child of node.childNodes) n.append(copy(child));
      return n;
    };
    try {
      return copy(tree.documentElement);
    } catch {
      return null;
    }
  }
  function renderObject(obj, assetURL, live = false, wheelNavigation = false) {
    const box = el("div", "layer"),
      face = el("div", "visual");
    Object.assign(box.style, {
      left: obj.x + "px",
      top: obj.y + "px",
      width: obj.w + "px",
      height: obj.h + "px",
      transform: "rotate(" + obj.rotation + "deg)",
    });
    box.append(face);
    const nativeSkin =
      obj.asset &&
      obj.original &&
      Object.keys(obj.original).every((k) => obj.original[k] === obj[k]);
    if (obj.type === "component") {
      if (!live && obj.poster) {
        const img = el("img");
        img.src = assetURL(obj.poster);
        img.alt = obj.name;
        img.draggable = false;
        face.append(img);
      } else face.append(StudioComponent.create(obj, live, wheelNavigation));
    } else if (nativeSkin || (obj.type === "image" && obj.asset)) {
      const img = el("img");
      img.src = assetURL(obj.asset);
      img.alt = obj.name;
      img.draggable = false;
      face.append(img);
    } else if (["video", "audio"].includes(obj.type) && obj.asset) {
      const media = el(obj.type);
      media.controls = true;
      media.preload = "none";
      if (obj.type === "video" && obj.poster)
        media.poster = assetURL(obj.poster);
      media.src = assetURL(obj.asset);
      face.append(media);
    } else {
      Object.assign(face.style, {
        background: obj.fill,
        color: obj.color,
        fontSize: obj.size + "px",
        fontFamily: obj.font,
        fontWeight: obj.bold ? "700" : "400",
        textAlign: obj.align,
      });
      if (obj.type === "ellipse") face.style.borderRadius = "50%";
      if (obj.type === "arrow")
        face.style.clipPath =
          "polygon(0 28%,85% 28%,85% 0,100% 50%,85% 100%,85% 72%,0 72%)";
      if (obj.type === "table") {
        const table = el("table");
        for (const line of obj.text.split("\n")) {
          const tr = table.insertRow();
          for (const v of line.split("\t")) tr.insertCell().textContent = v;
        }
        face.append(table);
      } else if (obj.type === "chart") {
        const rows = obj.text
          .split("\n")
          .map((l) => l.split("\t"))
          .map(([label, v]) => [label, Number(v)])
          .filter(([, v]) => Number.isFinite(v));
        const max = Math.max(1, ...rows.map(([, v]) => Math.abs(v)));
        for (const [label, v] of rows) {
          const row = el("div", "bar-row");
          const tag = el("span");
          tag.textContent = label;
          const bar = el("i");
          bar.style.width = (Math.abs(v) / max) * 55 + "%";
          bar.style.background = obj.color;
          row.append(tag, bar, document.createTextNode(String(v)));
          face.append(row);
        }
      } else {
        face.textContent = obj.text;
        if (obj.type === "formula") {
          face.style.fontFamily = '"Cambria Math",serif';
          const math = formula(obj.text);
          if (math) face.replaceChildren(math);
        }
      }
    }
    return box;
  }
  function styleCanvas(canvas, deck, slide, assetURL) {
    canvas.className = "slide-canvas";
    Object.assign(canvas.style, {
      width: deck.width + "px",
      height: deck.height + "px",
      backgroundColor: slide.background,
      backgroundImage: slide.backdrop
        ? 'url("' + assetURL(slide.backdrop) + '")'
        : "none",
    });
  }
  function mount(
    canvas,
    deck,
    slide,
    assetURL,
    live = false,
    wheelNavigation = false,
  ) {
    canvas.replaceChildren();
    styleCanvas(canvas, deck, slide, assetURL);
    for (const id of slide.order) {
      const box = renderObject(slide.elements[id], assetURL, live, wheelNavigation);
      box.dataset.object = id;
      canvas.append(box);
    }
  }
  function groups(steps) {
    const out = [[]];
    let group = out[0],
      last = 0,
      end = 0;
    for (const step of steps) {
      if (step.trigger === "click") {
        group = [];
        out.push(group);
        last = 0;
        end = 0;
      }
      const t =
        (step.trigger === "after" ? end : step.trigger === "with" ? last : 0) +
        step.delay;
      group.push({ ...step, time: t });
      last = t;
      end = t + step.duration;
    }
    return out;
  }
  function animate(
    canvas,
    slide,
    stepIndex,
    instant = false,
    replayActions = false,
  ) {
    const sequence = groups(slide.steps),
      timers = [],
      running = [],
      textRuns = [];
    const box = {
      w: parseFloat(canvas.style.width),
      h: parseFloat(canvas.style.height),
    };
    const nodes = new Map(
      [...canvas.querySelectorAll("[data-object]")].map((n) => [
        n.dataset.object,
        n.firstChild,
      ]),
    );
    for (const node of nodes.values()) {
      if (!initialStyles.has(node)) initialStyles.set(node, node.style.cssText);
      node.style.cssText = initialStyles.get(node);
      const counter = node.querySelector(".number-reveal");
      if (counter) counter.remove();
    }
    const seen = new Set();
    for (const s of slide.steps) {
      if (!seen.has(s.target) && StudioMotion.describe(s.effect).group === "in")
        nodes.get(s.target).style.visibility = "hidden";
      seen.add(s.target);
    }
    function apply(s, animated, performAction) {
      const node = nodes.get(s.target),
        spec = StudioMotion.describe(s.effect);
      if (spec.group === "component") {
        if (performAction)
          StudioComponent.send(node.querySelector("iframe"), s.action);
        return;
      }
      if (spec.group === "media") {
        const media = node.querySelector("video,audio");
        if (media && performAction) {
          if (
            spec.name === "MediaPlay" ||
            spec.name === "MediaPlayFromBookmark"
          ) {
            if (spec.name === "MediaPlayFromBookmark")
              media.currentTime = s.bookmark ?? 0;
            media.play().catch(() => {});
          } else {
            media.pause();
            if (spec.name === "MediaStop") media.currentTime = 0;
          }
        }
        return;
      }
      const frames = StudioMotion.frames(s, box);
      const finish = () => {
        if (spec.group === "out") node.style.visibility = "hidden";
        else if (spec.group === "in") node.style.visibility = "visible";
        else {
          for (const [key, value] of Object.entries(frames.at(-1)))
            if (key !== "offset") node.style[key] = value;
        }
      };
      if (spec.group === "in" || spec.group === "out")
        node.style.visibility = "visible";
      if (animated && s.duration && s.unit) {
        const stop = studioTextMotion(
          node,
          slide.elements[s.target],
          s,
          spec,
          frames,
          finish,
        );
        if (stop) {
          textRuns.push(stop);
          return;
        }
      }
      if (!animated || !s.duration || spec.name === "Appear") {
        finish();
        return;
      }
      if (spec.name === "NumberReveal") {
        const obj = slide.elements[s.target],
          match = obj.text.trim().match(/^([+-]?\d+(?:\.\d+)?)(%?)$/);
        if (match) {
          const label = el("span", "number-reveal"),
            start = performance.now();
          label.style.cssText =
            "position:absolute;inset:0;background:" +
            slide.background +
            ";color:" +
            obj.color +
            ";font-size:" +
            obj.size +
            "px";
          node.append(label);
          const decimals = (match[1].split(".")[1] || "").length;
          const frame = () => {
            const t = Math.min(
              1,
              (performance.now() - start) / (s.duration * 1000),
            );
            label.textContent =
              (Number(match[1]) * t).toFixed(decimals) + match[2];
            if (t < 1 && label.isConnected)
              rafs.push(requestAnimationFrame(frame));
            else label.remove();
          };
          frame();
        }
      }
      const animation = node.animate(frames, {
        duration: s.duration * 1000,
        easing: spec.group === "path" ? "linear" : "ease-out",
      });
      running.push(animation);
      animation.onfinish = finish;
    }
    const rafs = [];
    for (let i = 0; i <= stepIndex && i < sequence.length; i++) {
      for (const s of sequence[i]) {
        if (i === stepIndex && !instant)
          timers.push(setTimeout(() => apply(s, true, true), s.time * 1000));
        else apply(s, false, replayActions);
      }
    }
    return () => {
      timers.forEach(clearTimeout);
      textRuns.forEach((stop) => stop());
      rafs.forEach(cancelAnimationFrame);
      running.forEach((a) => {
        a.onfinish = null;
        a.cancel();
      });
    };
  }
  function fit(canvas, holder, deck) {
    const scale = Math.min(
      holder.clientWidth / deck.width,
      holder.clientHeight / deck.height,
    );
    canvas.style.transform = "scale(" + scale + ")";
    return scale;
  }
  function play(deck, assetURL, stage, controls, opts = {}) {
    let lastSlide = null;
    let page = Math.max(0, Math.min(deck.order.length - 1, opts.page || 0)),
      step = opts.step || 0,
      cancel = () => {};
    stage.classList.add("show-stage");
    controls.classList.add("show-controls");
    let canvas = el("div"),
      underlay = null,
      transitionAnimation = null;
    const status = el("span"),
      notes = el("aside", "speaker-notes");
    notes.hidden = true;
    stage.replaceChildren(canvas, notes);
    function button(label, action) {
      const b = el("button");
      b.textContent = label;
      b.onclick = action;
      return b;
    }
    function next() {
      if (step < groups(current().steps).length - 1) {
        step++;
        draw(false);
      } else if (page < deck.order.length - 1) {
        page++;
        step = 0;
        draw(true);
      }
    }
    function previous() {
      if (step > 0) {
        step--;
        draw(false, true);
      } else if (page > 0) {
        page--;
        step = groups(current().steps).length - 1;
        draw(true, true);
      }
    }
    const current = () => deck.slides[deck.order[page]];
    function draw(change, instant = false) {
      cancel();
      if (change) {
        transitionAnimation?.cancel();
        underlay?.remove();
        underlay = null;
        if (
          canvas.childElementCount &&
          current().transition !== "none" &&
          current().transitionDuration !== 0
        ) {
          const old = canvas;
          for (const media of old.querySelectorAll("iframe,video,audio")) {
            if (media.tagName === "VIDEO" && media.poster) {
              const poster = el("img");
              poster.src = media.poster;
              media.replaceWith(poster);
            } else media.remove();
          }
          underlay = el("div", "transition-underlay");
          underlay.inert = true;
          underlay.setAttribute("aria-hidden", "true");
          stage.insertBefore(underlay, old);
          const shadow = underlay.attachShadow({ mode: "open" }),
            style = el("style");
          const sheet = [...document.styleSheets].find(
            (s) =>
              s.href?.endsWith("/player.css") ||
              (!s.href && s.ownerNode.textContent.includes(".slide-canvas")),
          );
          if (sheet)
            style.textContent = [...sheet.cssRules]
              .map((r) => r.cssText)
              .join("\n");
          shadow.append(style, old);
          canvas = el("div");
          stage.insertBefore(canvas, notes);
        }
      }
      if (change || instant) mount(canvas, deck, current(), assetURL, true, true);
      fit(canvas, stage, deck);
      notes.textContent = current().notes || current().title;
      cancel = animate(canvas, current(), step, instant, instant);
      status.textContent =
        page + 1 + " / " + deck.order.length + "　·　步骤 " + step;
      if (change && current().transition !== "none") {
        const frames = StudioMotion.transition(current().transition, {
          w: deck.width,
          h: deck.height,
        });
        if (underlay && current().transition.startsWith("page-Morph")) {
          StudioMorph(
            underlay.shadowRoot.querySelector(".slide-canvas"),
            canvas,
            lastSlide,
            current(),
            current().transition,
          );
        }
        const base = canvas.style.transform;
        const fitted = frames.map(
          ({ transform, translate, scale, ...rest }) => {
            const move =
              translate && translate !== "none"
                ? " translate(" + translate.trim().split(/\s+/).join(",") + ")"
                : "";
            const grow =
              scale && scale !== "none" ? " scale(" + scale + ")" : "";
            const extra =
              transform && transform !== "none" ? " " + transform : "";
            return { ...rest, transform: base + move + grow + extra };
          },
        );
        const duration = 1000 * (current().transitionDuration ?? 0.3),
          back = underlay;
        if (back && /push/i.test(current().transition) && frames[0].translate) {
          const delta = frames[0].translate
            .split(" ")
            .map((v) => -parseFloat(v) * parseFloat(base.slice(6)) + "px")
            .join(" ");
          back.animate([{ translate: "0px 0px" }, { translate: delta }], {
            duration,
            fill: "forwards",
          });
        }
        transitionAnimation = canvas.animate(fitted, { duration });
        transitionAnimation.onfinish = () => {
          back?.remove();
          if (underlay === back) underlay = null;
        };
      }
      lastSlide = current();
    }
    const key = (e) => {
      if (e.key === " " && e.target === toggle) return;
      if (e.key === "Escape" && opts.close) {
        opts.close();
        return;
      }
      if (["ArrowRight", " ", "PageDown"].includes(e.key)) {
        e.preventDefault();
        next();
      }
      if (["ArrowLeft", "PageUp"].includes(e.key)) {
        e.preventDefault();
        previous();
      }
    };
    controls.replaceChildren(
      button("‹ 上一步", previous),
      status,
      button("下一步 ›", next),
      button("备注", () => {
        notes.hidden = !notes.hidden;
      }),
      button("全屏", opts.fullscreen || (() => stage.parentElement.requestFullscreen?.().catch(() => {}))),
    );
    if (opts.feedback)
      controls.append(
        button("反馈此刻", () => opts.feedback(deck.order[page], step)),
      );
    if (opts.close) controls.append(button("退出", opts.close));
    const toggle = button("", () =>
      setControlsCollapsed(!controls.classList.contains("controls-collapsed")),
    );
    toggle.type = "button";
    toggle.className = "controls-toggle";
    function setControlsCollapsed(collapsed) {
      controls.classList.toggle("controls-collapsed", collapsed);
      const label = collapsed ? "展开放映控制" : "收起放映控制";
      toggle.setAttribute("aria-expanded", String(!collapsed));
      toggle.setAttribute("aria-label", label);
      toggle.title = label;
    }
    setControlsCollapsed(true);
    controls.append(toggle);

    // Treat the momentum events from a continuous scroll as one navigation.
    let lastWheel = -Infinity,
      direction = 0,
      accumulated = 0,
      handled = false;
    function navigateByWheel(delta) {
      const now = performance.now(),
        sign = Math.sign(delta);
      if (now - lastWheel > 180 || direction !== sign) {
        accumulated = 0;
        handled = false;
      }
      lastWheel = now;
      direction = sign;
      accumulated += Math.abs(delta);
      if (handled || accumulated < 24) return;
      handled = true;
      if (sign > 0) next();
      else previous();
    }
    function wheel(event) {
      const delta = slideWheelDelta(event);
      if (!delta) return;
      event.preventDefault();
      navigateByWheel(delta);
    }
    function componentWheel(event) {
      const data = event.data;
      if (data?.type !== "studio-wheel" || !Number.isFinite(data.delta) || !data.delta)
        return;
      const frames = canvas.querySelectorAll("iframe.component-frame");
      if ([...frames].some(frame => frame.contentWindow === event.source))
        navigateByWheel(data.delta);
    }
    const click = (e) => {
      if (!e.target.closest("audio,video,button,.speaker-notes")) next();
    };
    stage.addEventListener("click", click);
    stage.addEventListener("wheel", wheel, { passive: false });
    controls.addEventListener("wheel", wheel, { passive: false });
    window.addEventListener("message", componentWheel);
    document.addEventListener("keydown", key);
    const ro = new ResizeObserver(() => fit(canvas, stage, deck));
    ro.observe(stage);
    draw(true);
    return () => {
      cancel();
      ro.disconnect();
      transitionAnimation?.cancel();
      underlay?.remove();
      document.removeEventListener("keydown", key);
      stage.removeEventListener("click", click);
      stage.removeEventListener("wheel", wheel);
      controls.removeEventListener("wheel", wheel);
      window.removeEventListener("message", componentWheel);
      controls.classList.remove("controls-collapsed");
      stage.replaceChildren();
      controls.replaceChildren();
    };
  }
  return { styleCanvas, mount, renderObject, groups, animate, fit, play };
})();
