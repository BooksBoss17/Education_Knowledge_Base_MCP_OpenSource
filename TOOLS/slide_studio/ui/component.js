function slideWheelDelta(event) {
  if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.shiftKey)
    return 0;

  const { deltaMode, deltaX, deltaY, target } = event;
  if (Math.abs(deltaY) <= Math.abs(deltaX)) return 0;
  if (target instanceof Element) {
    if (
      target.closest(
        "input, select, textarea, audio, video, [contenteditable], .speaker-notes",
      )
    )
      return 0;
    // Leave the wheel to scrollable content in notes, components and tables.
    for (let node = target; node; node = node.parentElement) {
      if (
        node.scrollHeight > node.clientHeight &&
        /^(auto|scroll|overlay)$/.test(getComputedStyle(node).overflowY)
      )
        return 0;
    }
  }
  return deltaY * (deltaMode === 1 ? 16 : deltaMode === 2 ? innerHeight : 1);
}

/* Interactive content runs in a document with a separate, opaque origin. */
const StudioComponent = (() => {
  const channels = new WeakMap();
  function create(obj, live, wheelNavigation = false) {
    if (!live) {
      const card = document.createElement("div");
      card.className = "component-card";
      card.textContent = "▷   " + obj.name;
      return card;
    }
    const frame = document.createElement("iframe");
    frame.className = "component-frame";
    frame.title = obj.name;
    frame.setAttribute("sandbox", "allow-scripts");
    frame.referrerPolicy = "no-referrer";
    const params = JSON.stringify(obj.parameters || {}).replaceAll(
      "<",
      "\\u003c",
    );
    const policy =
      "default-src 'none'; script-src 'unsafe-inline' blob:; style-src 'unsafe-inline'; img-src data: blob:; media-src data: blob:; font-src data:; connect-src 'none'; frame-src 'none'; form-action 'none'; base-uri 'none'";
    const wheelBridge = wheelNavigation
      ? slideWheelDelta.toString() +
        ";addEventListener('load',()=>addEventListener('wheel',event=>{" +
        "const delta=slideWheelDelta(event);if(!delta)return;event.preventDefault();" +
        "parent.postMessage({type:'studio-wheel',delta},'*');" +
        "},{passive:false}),{once:true});"
      : "";
    const bridge =
      "window.studioParams=" +
      params +
      ";addEventListener('message',e=>{if(e.source===parent&&e.data?.type==='studio-action')dispatchEvent(new CustomEvent('studio-action',{detail:e.data.action}))});" +
      wheelBridge;
    frame.srcdoc =
      '<!doctype html><meta charset=utf-8><meta http-equiv="Content-Security-Policy" content="' +
      policy +
      '"><style>html,body{margin:0;width:100%;height:100%}</style><script>' +
      bridge +
      "<\/script>" +
      obj.html;
    const state = { ready: false, pending: [] };
    channels.set(frame, state);
    frame.addEventListener("load", () => {
      state.ready = true;
      for (const action of state.pending.splice(0)) send(frame, action);
    });
    return frame;
  }
  function send(frame, action) {
    const state = channels.get(frame);
    if (!state) return;
    if (!state.ready) state.pending.push(action);
    else
      frame.contentWindow?.postMessage({ type: "studio-action", action }, "*");
  }
  return { create, send };
})();
