/* Browser-native choreography. Coordinates in imported motion paths are fractions of the slide. */
const StudioMotion = (() => {
  const legacy = {
    appear: ["in", "Appear"],
    fade: ["in", "Fade"],
    rise: ["in", "RiseUp"],
    zoom: ["in", "Zoom"],
    pulse: ["accent", "GrowShrink"],
    exit: ["out", "Fade"],
    flyout: ["out", "Fly"],
    "custom-path": ["path", "Custom"],
    component: ["component", "Action"],
  };
  const describe = (key) =>
    StudioCatalog.effects[key] || {
      group: legacy[key]?.[0],
      name: legacy[key]?.[1],
    };
  const from = (css) => [
    css,
    Object.fromEntries(Object.keys(css).map((k) => [k, neutral[k] ?? "none"])),
  ];
  const neutral = {
    opacity: 1,
    transform: "none",
    filter: "none",
    clipPath: "inset(0%)",
    maskImage: "none",
    translate: "0px 0px",
  };
  const transform = (...values) => values.map((transform) => ({ transform }));
  const pulse = (css) => [{}, css, {}];
  const directions = {
    1: [0, -1],
    2: [1, 0],
    3: [0, 1],
    4: [-1, 0],
    6: [-1, -1],
    7: [1, -1],
    8: [1, 1],
    9: [-1, 1],
    10: [0, -1],
    11: [0, 1],
    12: [-1, -1],
    13: [1, -1],
    14: [1, 1],
    15: [-1, 1],
  };
  const sample = (n, fn) =>
    Array.from({ length: n + 1 }, (_, i) => ({ offset: i / n, ...fn(i / n) }));

  function maskFrames(name, dir, spokes = 1) {
    return sample(32, (t) => {
      const stop = (100 * t).toFixed(2) + "%";
      const stripe = dir === 17 ? "90deg" : "0deg";
      if (name === "Wheel" || name === "Wedge") {
        const sector = 360 / spokes;
        return {
          maskImage:
            "repeating-conic-gradient(#000 0deg " +
            sector * t +
            "deg,transparent " +
            sector * t +
            "deg " +
            sector +
            "deg)",
        };
      }
      if (name === "Blinds")
        return {
          maskImage:
            "repeating-linear-gradient(" +
            stripe +
            ",#000 0 " +
            t * 10 +
            "%,transparent " +
            t * 10 +
            "% 10%)",
        };
      if (
        name === "Checkerboard" ||
        name === "Dissolve" ||
        name === "RandomBars"
      ) {
        const cols = name === "RandomBars" ? 1 : 12,
          rows = name === "Checkerboard" ? 8 : 14;
        let rects = "";
        for (let y = 0; y < rows; y++)
          for (let x = 0; x < cols; x++) {
            const rank = ((x * 37 + y * 59 + x * y * 7) % 101) / 101;
            const fraction =
              name === "Checkerboard"
                ? Math.min(1, Math.max(0, 2 * t - ((x + y) % 2)))
                : Number(t >= rank && t > 0);
            if (fraction)
              rects +=
                '<rect x="' +
                x +
                '" y="' +
                y +
                '" width="' +
                fraction +
                '" height="1.01"/>';
          }
        const svg =
          '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' +
          cols +
          " " +
          rows +
          '" preserveAspectRatio="none">' +
          rects +
          "</svg>";
        return {
          maskImage:
            'url("data:image/svg+xml,' + encodeURIComponent(svg) + '")',
        };
      }
      if (name === "Plus") {
        const a = 50 * (1 - t),
          b = 100 - a;
        return {
          clipPath:
            "polygon(" +
            a +
            "% 0," +
            b +
            "% 0," +
            b +
            "% " +
            a +
            "%,100% " +
            a +
            "%,100% " +
            b +
            "%," +
            b +
            "% " +
            b +
            "%," +
            b +
            "% 100%," +
            a +
            "% 100%," +
            a +
            "% " +
            b +
            "%,0 " +
            b +
            "%,0 " +
            a +
            "%," +
            a +
            "% " +
            a +
            "%)",
        };
      }
      return {
        maskImage:
          "linear-gradient(135deg,#000 " + stop + ",transparent " + stop + ")",
      };
    });
  }
  function entrance(name, dir, box) {
    const [x, y] = directions[dir] || [0, 1];
    const move = "translate(" + x * box.w + "px," + y * box.h + "px)";
    const clip = (shape) => [{ clipPath: shape[0] }, { clipPath: shape[1] }];
    switch (name) {
      case "Appear":
        return [{ opacity: 1 }, { opacity: 1 }];
      case "Shimmer":
        return [
          { opacity: 0, filter: "brightness(2)" },
          { opacity: 0.8, filter: "brightness(1.5)" },
          { opacity: 1, filter: "none" },
        ];
      case "3DArrive":
        return from({
          transform: "perspective(800px) translateZ(-200px) rotateY(75deg)",
          opacity: 0,
        });
      case "Fade":
        return from({ opacity: 0 });
      case "Fly":
      case "Crawl":
        return from({ transform: move });
      case "Blinds":
      case "Checkerboard":
      case "Dissolve":
      case "RandomBars":
      case "Wedge":
      case "Wheel":
      case "Plus":
      case "Strips":
        return maskFrames(name, dir);
      case "Box":
        return clip(["inset(50%)", "inset(0%)"]);
      case "Circle":
        return clip(["circle(0%)", "circle(75%)"]);
      case "Diamond":
        return clip([
          "polygon(50% 50%,50% 50%,50% 50%,50% 50%)",
          "polygon(50% -50%,150% 50%,50% 150%,-50% 50%)",
        ]);
      case "FlashOnce":
        return [
          { opacity: 0 },
          { opacity: 1 },
          { opacity: 0.2 },
          { opacity: 1 },
        ];
      case "Peek":
      case "Wipe": {
        const inset =
          x < 0
            ? "0 100% 0 0"
            : x > 0
              ? "0 0 0 100%"
              : y < 0
                ? "100% 0 0 0"
                : "0 0 100% 0";
        return [
          {
            clipPath: "inset(" + inset + ")",
            transform:
              name === "Peek"
                ? "translate(" + -x * 35 + "%," + -y * 35 + "%)"
                : "none",
          },
          { clipPath: "inset(0%)", transform: "none" },
        ];
      }
      case "Spiral":
        return from({ transform: "scale(.05) rotate(-420deg)", opacity: 0 });
      case "Split":
        return clip([
          dir === 16 || dir === 23 ? "inset(0 50%)" : "inset(50% 0)",
          "inset(0%)",
        ]);
      case "Stretch":
        return from({ transform: dir === 17 ? "scaleY(.01)" : "scaleX(.01)" });
      case "Swivel":
        return from({
          transform: "perspective(800px) rotateY(180deg)",
          opacity: 0,
        });
      case "Zoom":
        return from({ transform: "scale(.08)" });
      case "RandomEffects":
        return entrance(
          ["Wipe", "Circle", "Zoom"][Math.floor(Math.random() * 3)],
          dir,
          box,
        );
      case "Boomerang":
        return transform(
          "translate(-80%,50%) rotate(-90deg)",
          "translate(15%,-20%) rotate(22deg)",
          "none",
        );
      case "Bounce":
        return transform(
          "translateY(-120%)",
          "none",
          "translateY(-36%)",
          "none",
          "translateY(-10%)",
          "none",
        );
      case "ColorReveal":
        return from({ filter: "hue-rotate(150deg) saturate(3)", opacity: 0.1 });
      case "Credits":
        return from({ transform: "translateY(" + box.h + "px)" });
      case "EaseIn":
        return from({ transform: "translateX(-30%)", opacity: 0 });
      case "Float":
        return from({
          transform: "translate(-18%,35%) rotate(-12deg)",
          opacity: 0,
        });
      case "GrowAndTurn":
        return from({ transform: "scale(.1) rotate(-95deg)", opacity: 0.1 });
      case "LightSpeed":
        return transform(
          "translateX(-150%) skewX(35deg)",
          "translateX(5%) skewX(-12deg)",
          "none",
        );
      case "Pinwheel":
        return from({ transform: "scale(.15) rotate(-260deg)" });
      case "RiseUp":
        return from({ transform: "translateY(30px)", opacity: 0 });
      case "Swish":
        return from({
          transform: "translate(-45%,40%) skewX(-25deg)",
          opacity: 0,
        });
      case "ThinLine":
        return transform("scale(.04,.02)", "scale(1,.02)", "none");
      case "Unfold":
        return from({
          transform: "perspective(650px) rotateX(-85deg) scaleY(.1)",
        });
      case "Whip":
        return transform(
          "translateX(-70%) skewY(14deg)",
          "skewY(-8deg)",
          "none",
        );
      case "Ascend":
        return from({ transform: "translateY(85%)", opacity: 0 });
      case "CenterRevolve":
        return from({ transform: "rotateY(275deg) scale(.65)", opacity: 0.1 });
      case "FadedSwivel":
        return from({ transform: "rotateY(100deg)", opacity: 0 });
      case "Descend":
        return from({ transform: "translateY(-85%)", opacity: 0 });
      case "Sling":
        return transform(
          "translate(-80%,-35%) rotate(-65deg)",
          "rotate(12deg)",
          "none",
        );
      case "Spinner":
        return from({ transform: "rotate(540deg) scale(.4)", opacity: 0 });
      case "Stretchy":
        return transform("scale(.15,1.3)", "scale(1.16,.9)", "none");
      case "Zip":
        return from({ transform: "translateX(-100%) scaleX(.15)" });
      case "ArcUp":
        return transform(
          "translate(-45%,60%) rotate(-30deg)",
          "translate(-25%,-15%) rotate(-12deg)",
          "none",
        );
      case "FadedZoom":
        return from({ transform: "scale(.55)", opacity: 0 });
      case "Glide":
        return from({ transform: "translate(55%,-15%)", opacity: 0.05 });
      case "Expand":
        return from({ transform: "scaleX(.05)", opacity: 0 });
      case "Flip":
        return from({
          transform: "perspective(650px) rotateX(120deg)",
          opacity: 0,
        });
      case "Fold":
        return from({
          transform: "perspective(500px) rotateY(-85deg) skewY(20deg)",
          opacity: 0.1,
        });
      case "NumberReveal":
        return clip(["inset(0 100% 0 0)", "inset(0%)"]);
      default:
        throw Error("未知进入效果 " + name);
    }
  }
  function emphasis(name) {
    switch (name) {
      case "ChangeFillColor":
        return [{}, { backgroundColor: "#ffe5a0" }];
      case "ChangeFont":
        return [{}, { fontFamily: "serif" }];
      case "ChangeFontColor":
        return [{}, { color: "#c45728" }];
      case "ChangeFontSize":
        return transform("scale(1)", "scale(1.24)");
      case "ChangeFontStyle":
        return [{}, { fontStyle: "italic" }];
      case "GrowShrink":
        return pulse({ transform: "scale(1.15)" });
      case "ChangeLineColor":
        return [{}, { outline: "2px solid #168974" }];
      case "Spin":
        return transform("rotate(0deg)", "rotate(360deg)");
      case "3DSwing":
        return transform(
          "none",
          "perspective(700px) rotateY(-35deg)",
          "perspective(700px) rotateY(35deg)",
          "none",
        );
      case "3DTurntable":
        return transform(
          "perspective(700px) rotateY(0deg)",
          "perspective(700px) rotateY(360deg)",
        );
      case "3DJumpAndTurn":
        return transform(
          "none",
          "perspective(700px) translateY(-35%) rotateY(180deg)",
          "perspective(700px) rotateY(360deg)",
        );
      case "Transparency":
        return [{ opacity: 1 }, { opacity: 0.45 }];
      case "BoldFlash":
        return pulse({ fontWeight: "900", filter: "contrast(1.5)" });
      case "Blast":
        return pulse({ transform: "scale(1.22)", filter: "brightness(1.6)" });
      case "BoldReveal":
        return [{}, { fontWeight: "900" }];
      case "BrushOnColor":
        return [{ color: "inherit" }, { color: "#168974" }];
      case "BrushOnUnderline":
        return [{}, { textDecoration: "underline" }];
      case "ColorBlend":
        return [{}, { filter: "sepia(1)" }, { filter: "hue-rotate(120deg)" }];
      case "ColorWave":
        return pulse({ filter: "hue-rotate(190deg) saturate(2)" });
      case "ComplementaryColor":
        return [{}, { filter: "hue-rotate(180deg)" }];
      case "ComplementaryColor2":
        return [{}, { filter: "invert(1) hue-rotate(180deg)" }];
      case "ContrastingColor":
        return [{}, { filter: "contrast(2) saturate(2)" }];
      case "Darken":
        return [{}, { filter: "brightness(.6)" }];
      case "Desaturate":
        return [{}, { filter: "grayscale(1)" }];
      case "FlashBulb":
        return pulse({ filter: "brightness(2)", opacity: 0.65 });
      case "Flicker":
        return [1, 0.15, 1, 0.3, 1].map((opacity) => ({ opacity }));
      case "GrowWithColor":
        return pulse({ transform: "scale(1.18)", filter: "saturate(2)" });
      case "Lighten":
        return [{}, { filter: "brightness(1.5)" }];
      case "StyleEmphasis":
        return pulse({ filter: "drop-shadow(0 0 5px #e99534)" });
      case "Teeter":
        return transform(
          "rotate(0deg)",
          "rotate(-8deg)",
          "rotate(8deg)",
          "none",
        );
      case "VerticalGrow":
        return transform("scaleY(1)", "scaleY(1.3)");
      case "Wave":
        return transform(
          "none",
          "translateY(-9%) rotate(-4deg)",
          "translateY(6%) rotate(3deg)",
          "none",
        );
      default:
        throw Error("未知强调效果 " + name);
    }
  }
  function pathFrames(path, box) {
    const curve = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "path",
    );
    curve.setAttribute("d", path.replace(/\s+[Ee]\s*$/, ""));
    const length = curve.getTotalLength();
    return sample(80, (t) => {
      const pt = curve.getPointAtLength(length * t);
      return { translate: pt.x * box.w + "px " + pt.y * box.h + "px" };
    });
  }
  function frames(step, box) {
    const spec = describe(step.effect);
    if (spec.group === "path") return pathFrames(step.path || spec.path, box);
    if (spec.group === "accent") return emphasis(spec.name);
    const keys = entrance(spec.name, step.direction, box);
    if (spec.group === "out")
      return keys.reverse().map((k) => {
        const frame = { ...k };
        if (frame.offset !== undefined) frame.offset = 1 - frame.offset;
        return frame;
      });
    return keys;
  }
  function transition(key, box) {
    const name = StudioCatalog.transitions[key]?.name || key;
    const dx = /Left/.test(name) ? -1 : /Right/.test(name) ? 1 : 0;
    const dy = /Up|Top/.test(name) ? -1 : /Down|Bottom/.test(name) ? 1 : 0;
    const dir = dx < 0 ? 4 : dx > 0 ? 2 : dy < 0 ? 1 : 3;
    if (/^CutThroughBlack|RevealBlack/.test(name))
      return from({ filter: "brightness(0)" });
    if (/^Appear|^Cut$/.test(name)) return [{ opacity: 1 }, { opacity: 1 }];
    if (/^Morph/.test(name)) return from({ opacity: 0 });
    if (/^(FallOver|PageCurl|PeelOff|Origami)/.test(name))
      return from({
        transform:
          "perspective(1000px) rotateY(" + (dx || 1) * 75 + "deg) skewY(-7deg)",
        opacity: 0.2,
      });
    if (/^(Drape|Curtains)/.test(name)) return entrance("Split", 16, box);
    if (/^(Crush|Fracture)/.test(name)) return maskFrames("Dissolve", dir);
    if (/^Airplane|^Wind/.test(name))
      return from({
        transform:
          "translateX(" + (dx || 1) * 100 + "%) rotate(-14deg) scale(.6)",
        opacity: 0,
      });
    if (name === "Prestige")
      return from({
        transform: "perspective(800px) rotateX(60deg) scale(.2)",
        opacity: 0,
      });
    if (/^Fade|^fade$/.test(name)) return from({ opacity: 0 });
    if (/^Random$/.test(name)) return entrance("RandomEffects", dir, box);
    if (/^Wheel|^Wedge/.test(name))
      return maskFrames("Wheel", dir, Number(name.match(/\d+/)?.[0] || 1));
    if (/^Blinds|^Checkerboard|^RandomBars|^Dissolve|^Strips/.test(name))
      return maskFrames(
        name.match(/^Blinds|^Checkerboard|^RandomBars|^Dissolve|^Strips/)[0],
        /Vertical/.test(name) ? 17 : 16,
      );
    if (/^Glitter|^Shred|^Honeycomb/.test(name))
      return maskFrames(
        /Diamond|Hexagon|Honeycomb/.test(name) ? "Checkerboard" : "Dissolve",
        dir,
      );
    if (/^Circle|^Ripple/.test(name)) return entrance("Circle", dir, box);
    if (/^Diamond|^Plus/.test(name))
      return entrance(name.startsWith("Plus") ? "Plus" : "Diamond", dir, box);
    if (/^Box(In|Out)$/.test(name)) return entrance("Box", dir, box);
    if (/^Split|^Window|^Doors|^Comb/.test(name))
      return entrance("Split", /Vertical/.test(name) ? 17 : 16, box);
    if (/^Wipe|^Reveal|^Peek|^Uncover/.test(name))
      return entrance("Wipe", dir, box);
    if (
      /^Cube|^Rotate|^Flip|^Gallery|^Conveyor|^Switch|^Box|^Orbit|^Ferris/.test(
        name,
      )
    ) {
      const axis = dy ? "X" : "Y",
        angle = (dx || dy || 1) * (/^Cube|^Box/.test(name) ? 90 : 65);
      return from({
        transform:
          "perspective(900px) rotate" + axis + "(" + angle + "deg) scale(.85)",
        opacity: 0.15,
      });
    }
    if (/^Zoom|^zoom|^FlyThrough|^Warp/.test(name))
      return from({ scale: /Out/.test(name) ? "1.5" : ".6", opacity: 0 });
    if (/^Stretch|^Swivel|^Spiral|^Vortex|^Newsflash/.test(name))
      return entrance(
        name.startsWith("Stretch")
          ? "Stretch"
          : name === "Swivel"
            ? "Swivel"
            : "Spiral",
        dir,
        box,
      );
    if (/^Flash/.test(name))
      return from({ filter: "brightness(3)", opacity: 0.15 });
    return from({
      translate: (dx || (dy ? 0 : -1)) * box.w + "px " + dy * box.h + "px",
    });
  }
  return { describe, frames, transition };
})();
