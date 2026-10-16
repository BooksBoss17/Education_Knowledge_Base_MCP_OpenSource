(() => {
  const token = new URLSearchParams(location.search).get("token") || "";
  const editor = document.getElementById("formula-editor");
  const message = document.getElementById("message");
  const crop = document.getElementById("source-crop");
  const progress = document.getElementById("progress");
  const advanced = document.getElementById("advanced-latex");
  const confirm = document.getElementById("confirm");
  let tasks = [], index = 0, original = "", zoom = 1, dirty = false;
  if (window.MathfieldElement) {
    window.MathfieldElement.fontsDirectory = "./mathlive/fonts";
    window.MathfieldElement.soundsDirectory = "./mathlive/sounds";
  }
  editor.mathVirtualKeyboardPolicy = "manual";
  editor.addEventListener("input", () => {
    advanced.textContent = editor.value;
    dirty = true;
    confirm.textContent = "确认修改";
  });
  const keys = {
    common: [["+","+"],["−","-"],["×","\\times"],["·","\\cdot"],["÷","\\div"],["=","="],["≠","\\ne"],["≈","\\approx"],["≤","\\le"],["≥","\\ge"],["±","\\pm"],["∝","\\propto"],["∞","\\infty"]],
    structure: [["分数","\\frac{#0}{#?}"],["上标","^{#?}"],["下标","_{#?}"],["√□","\\sqrt{#0}"],["( )","\\left(#0\\right)"],["| |","\\left|#0\\right|"],["向量","\\vec{#0}"],["∫","\\int"],["Σ","\\sum"]],
    greek: [["α","\\alpha"],["β","\\beta"],["γ","\\gamma"],["δ","\\delta"],["ε","\\epsilon"],["θ","\\theta"],["λ","\\lambda"],["μ","\\mu"],["ρ","\\rho"],["σ","\\sigma"],["φ","\\phi"],["ω","\\omega"],["Δ","\\Delta"],["Ω","\\Omega"]],
    physics: [["→","\\to"],["°","^{\\circ}"],["Δ","\\Delta"],["矢量模板","\\vec{#0}"],["点乘","\\cdot"]]
  };
  function renderKeys(name) {
    const grid = document.getElementById("key-grid");
    grid.replaceChildren(...keys[name].map(([label, value]) => {
      const button = document.createElement("button"); button.type = "button"; button.textContent = label;
      button.addEventListener("click", () => { editor.focus(); editor.insert(value); }); return button;
    }));
  }
  async function load() {
    const response = await fetch(`/api/tasks?token=${encodeURIComponent(token)}`);
    if (!response.ok) throw new Error("无法打开本地审查会话。");
    tasks = (await response.json()).tasks; show(0);
  }
  function show(nextIndex) {
    if (!tasks.length) { progress.textContent = "0 / 0"; return; }
    index = Math.max(0, Math.min(nextIndex, tasks.length - 1));
    const task = tasks[index]; original = task.current_latex; editor.value = original; advanced.textContent = original; dirty = false; confirm.textContent = "识别正确";
    crop.src = `/api/crop/${encodeURIComponent(task.formula_id)}?token=${encodeURIComponent(token)}`;
    progress.textContent = `${index + 1} / ${tasks.length}`; message.textContent = ""; zoom = 1; crop.style.transform = "scale(1)";
  }
  async function submit(action) {
    const task = tasks[index];
    const payload = {formula_id: task.formula_id, action, latex: editor.value, math_json: editor.getValue ? editor.getValue("math-json") : null, resolver_id: "local-human-session"};
    const response = await fetch(`/api/human-resolution?token=${encodeURIComponent(token)}`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    const body = await response.json(); if (!response.ok) { message.textContent = body.message; return; }
    tasks[index] = body; message.textContent = action === "UNRESOLVED" ? "已保留原图，稍后可以继续处理。" : "这个公式已完成确认。";
  }
  document.querySelectorAll("[data-tab]").forEach(button => button.addEventListener("click", () => { document.querySelectorAll("[data-tab]").forEach(item => item.classList.toggle("active", item === button)); renderKeys(button.dataset.tab); }));
  document.querySelectorAll("[data-command]").forEach(button => button.addEventListener("click", () => { const command = button.dataset.command; if (command === "restore") { editor.value = original; advanced.textContent = original; dirty = false; confirm.textContent = "识别正确"; } else editor.executeCommand(command); editor.focus(); }));
  document.querySelectorAll("[data-zoom]").forEach(button => button.addEventListener("click", () => { zoom = button.dataset.zoom === "reset" ? 1 : Math.max(.6, Math.min(2.5, zoom + (button.dataset.zoom === "in" ? .2 : -.2))); crop.style.transform = `scale(${zoom})`; }));
  document.querySelector(".guide-close").addEventListener("click", () => document.getElementById("guide").hidden = true);
  document.getElementById("previous").addEventListener("click", () => show(index - 1)); document.getElementById("next").addEventListener("click", () => show(index + 1));
  confirm.addEventListener("click", () => submit(dirty ? "EDITED" : "CONFIRMED_CURRENT")); document.getElementById("edit").addEventListener("click", () => editor.focus()); document.getElementById("unresolved").addEventListener("click", () => submit("UNRESOLVED"));
  renderKeys("common"); load().catch(error => { message.textContent = error.message; });
})();
