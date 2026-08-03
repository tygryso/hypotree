/* hypotree dashboard — Vue 3 over server-computed coordinates.
 *
 * The server does the graph theory: layers, ordering, x/y. This file does what a
 * browser is actually good at — pan, zoom, hit-testing and transitions. That
 * split is why there is no graph library here beyond d3-zoom.
 */

const { createApp, ref, reactive, computed, onMounted, watch, nextTick } = Vue;

const metaToken = document.querySelector('meta[name="hypotree-token"]');
const TOKEN =
  new URLSearchParams(location.search).get("t") ||
  (metaToken && metaToken.content !== "__TOKEN__" ? metaToken.content : "");

const api = async (path, opts) => {
  const sep = path.includes("?") ? "&" : "?";
  const res = await fetch(`${path}${sep}t=${encodeURIComponent(TOKEN)}`, opts);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
  return res.json();
};

// Node statements and reasons are written by an agent and are untrusted — the
// guide says so in as many words — and `marked` does not sanitize. The strict
// script-src already stops an injected handler from running; this stops it being
// in the document at all, which is the half that survives a policy change.
const ALLOWED_TAGS = new Set(
  ("p br hr h1 h2 h3 h4 h5 h6 ul ol li strong em b i del code pre blockquote " +
   "table thead tbody tr th td a span div").split(" ")
);
const ALLOWED_ATTRS = new Set(["href", "title", "class", "colspan", "rowspan", "align"]);

function sanitize(html) {
  // DOMParser does not execute anything it parses, so this is safe to inspect.
  const doc = new DOMParser().parseFromString(html, "text/html");
  for (const el of [...doc.body.querySelectorAll("*")]) {
    if (!ALLOWED_TAGS.has(el.tagName.toLowerCase())) {
      el.replaceWith(...el.childNodes);
      continue;
    }
    for (const attr of [...el.attributes]) {
      const name = attr.name.toLowerCase();
      const value = attr.value.trim().toLowerCase();
      const isUnsafeUrl =
        name === "href" && !/^(https?:|mailto:|#|\/)/.test(value) && value !== "";
      if (!ALLOWED_ATTRS.has(name) || name.startsWith("on") || isUnsafeUrl) {
        el.removeAttribute(attr.name);
      }
    }
  }
  return doc.body.innerHTML;
}

createApp({
  setup() {
    const meta = ref(null);
    const graph = reactive({ nodes: [], edges: [], stats: {} });
    const frontier = ref([]);
    const narrative = ref("");
    const timeline = ref({ ticks: [], from: null, to: null });

    const goalId = ref("");
    const tab = ref("path");
    const selected = ref(null);
    const detail = ref(null);
    const live = ref(false);
    const busyUntil = ref(0);
    const now = ref(Date.now());
    const scrub = ref(0);
    const playing = ref(false);
    const tip = reactive({ show: false, x: 0, y: 0, node: null });
    const error = ref("");
    const missing = ref([]);

    // ---- left panel width ------------------------------------------------

    const PANEL_GROWTH = 400;    // how much wider than its CSS default it may go
    // Null until the panel has been measured, so the first paint uses the
    // stylesheet's width rather than a guess at what a rem is worth here.
    const panelBase = ref(0);
    const panelWidth = ref(null);
    const resizing = ref(false);
    // Never past half the window: a reading panel that eats the graph defeats
    // the point of putting them side by side.
    const panelMax = () =>
      Math.min(panelBase.value + PANEL_GROWTH, Math.round(window.innerWidth / 2));
    const clampPanel = (w) => Math.max(panelBase.value, Math.min(panelMax(), w));

    function startResize(ev) {
      ev.preventDefault();
      const aside = ev.currentTarget.closest("aside");
      if (!aside) return;
      const left = aside.getBoundingClientRect().left;
      resizing.value = true;
      const move = (e) => (panelWidth.value = clampPanel(e.clientX - left));
      const stop = () => {
        resizing.value = false;
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", stop);
        fit();
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", stop);
    }

    // "The agent is working" is a claim about the last minute, not about whether
    // a socket is open. A dashboard that always looks busy tells you nothing.
    const working = computed(() => now.value < busyUntil.value);
    // The scrubber runs left-to-right through history, so its right-hand end is
    // the present. Anywhere short of that end is a rewind.
    const lastTick = computed(() => Math.max(0, timeline.value.ticks.length - 1));
    const atLive = computed(() => scrub.value >= lastTick.value);
    const at = computed(() => {
      const t = timeline.value.ticks;
      return atLive.value || !t.length ? null : t[scrub.value].t;
    });
    const scrubLabel = computed(() => {
      const t = timeline.value.ticks;
      if (!t.length) return "no history yet";
      if (atLive.value) return `live · ${t.length} changes`;
      const tick = t[scrub.value];
      return `${scrub.value + 1}/${t.length} · ${tick.node_id} → ${tick.status}`;
    });

    const q = () => (goalId.value ? `goal_id=${encodeURIComponent(goalId.value)}` : "");

    async function loadMeta() {
      meta.value = await api("/api/meta");
      missing.value = meta.value.missing_assets || [];
    }
    async function loadGraph() {
      const parts = [q(), at.value ? `at=${encodeURIComponent(at.value)}` : ""].filter(Boolean);
      const g = await api(`/api/graph${parts.length ? "?" + parts.join("&") : ""}`);
      graph.nodes = g.nodes;
      graph.edges = g.edges;
      graph.stats = g.stats;
    }
    async function loadNarrative() {
      // The narrative is pinned to the same instant as the graph, so a rewound
      // picture is never captioned with conclusions it has not reached.
      const parts = [q(), at.value ? `at=${encodeURIComponent(at.value)}` : ""].filter(Boolean);
      const r = await api(`/api/learning-path${parts.length ? "?" + parts.join("&") : ""}`);
      narrative.value = r.markdown || "";
    }
    async function loadPanels() {
      const query = q();
      frontier.value = (await api(`/api/frontier?k=5${query ? "&" + query : ""}`)).candidates;
      await loadNarrative();
      const wasLive = atLive.value;
      timeline.value = await api(`/api/timeline${query ? "?" + query : ""}`);
      // New history arriving must not silently rewind a viewer who was watching
      // the present — the handle rides the end.
      if (wasLive) scrub.value = lastTick.value;
    }
    async function refreshAll() {
      try {
        error.value = "";
        await Promise.all([loadGraph(), loadPanels()]);
      } catch (e) {
        error.value = String(e.message || e);
      }
    }

    // ---- graph rendering -------------------------------------------------

    const byId = computed(() => Object.fromEntries(graph.nodes.map((n) => [n.id, n])));
    const SPREAD_X = 116, SPREAD_Y = 96;
    const px = (n) => n.x * SPREAD_X;
    const py = (n) => n.y * SPREAD_Y;

    const edgeLines = computed(() =>
      graph.edges
        .map((e) => {
          const a = byId.value[e.src], b = byId.value[e.dst];
          if (!a || !b) return null;
          const dead = ["PRUNED", "INVALIDATED"].includes(b.status) ||
                       ["PRUNED", "INVALIDATED"].includes(a.status);
          return { key: `${e.src}->${e.dst}:${e.type}`, x1: px(a), y1: py(a), x2: px(b), y2: py(b), dead };
        })
        .filter(Boolean)
    );

    const radius = (n) => (n.is_goal ? 13 : 8 + 7 * (n.p_select || 0));
    // Untested nodes glow at their real chance of being dispatched next.
    const nodeOpacity = (n) =>
      n.status === "UNTESTED" ? 0.32 + 0.68 * Math.min(1, (n.p_select || 0) * 3) : 1;

    function fit() {
      const svg = document.querySelector("svg.graph");
      if (!svg || !graph.nodes.length || !window.d3 || !d3.zoomIdentity) return;
      const xs = graph.nodes.map(px), ys = graph.nodes.map(py);
      const w = svg.clientWidth, h = svg.clientHeight;
      const bw = Math.max(...xs) - Math.min(...xs) + 180;
      const bh = Math.max(...ys) - Math.min(...ys) + 140;
      const k = Math.min(1.5, Math.min(w / bw, h / bh));
      const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
      const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
      d3.select(svg).transition().duration(450).call(
        zoomBehaviour.transform,
        d3.zoomIdentity.translate(w / 2, h / 2).scale(k).translate(-cx, -cy)
      );
    }

    let zoomBehaviour = null;
    function installZoom() {
      // d3-zoom writes one transform on one <g>. The browser composites it on
      // the GPU, so panning a few hundred nodes costs nothing and Vue never
      // re-renders during a drag.
      if (!window.d3 || !d3.zoom) return;
      const svg = d3.select("svg.graph");
      zoomBehaviour = d3.zoom().scaleExtent([0.15, 4]).on("zoom", (ev) => {
        document.getElementById("viewport").setAttribute("transform", ev.transform);
      });
      svg.call(zoomBehaviour);
    }

    function zoomBy(k) {
      if (!zoomBehaviour || !window.d3) return;
      d3.select("svg.graph").transition().duration(200).call(zoomBehaviour.scaleBy, k);
    }

    function pick(node, ev) {
      selected.value = node.id;
      detail.value = null;
      api(`/api/node/${encodeURIComponent(node.id)}`).then((d) => (detail.value = d));
      tab.value = "path";
      if (ev) hover(node, ev);
    }
    function hover(node, ev) {
      tip.show = true; tip.node = node;
      const box = document.querySelector(".graph-wrap").getBoundingClientRect();
      tip.x = ev.clientX - box.left + 14;
      tip.y = ev.clientY - box.top + 12;
    }

    // ---- live ------------------------------------------------------------

    function connect() {
      const es = new EventSource(`/api/events?t=${encodeURIComponent(TOKEN)}`);
      es.onopen = () => (live.value = true);
      es.onerror = () => { live.value = false; setTimeout(connect, 3000); es.close(); };
      es.onmessage = (msg) => {
        live.value = true;
        const data = JSON.parse(msg.data);
        if (meta.value && data.revision !== meta.value.revision) {
          // Something changed: mark the agent as working for the next minute and
          // refetch. The stream carries a number; the client fetches the rest.
          busyUntil.value = Date.now() + 60_000;
          meta.value.revision = data.revision;
          if (atLive.value) refreshAll();
        }
      };
    }

    async function directive(mode) {
      if (!selected.value) return;
      try {
        await api("/api/directive", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ node_id: selected.value, mode }),
        });
        await refreshAll();
      } catch (e) {
        error.value = String(e.message || e);
      }
    }

    let timer = null;
    function goLive() {
      playing.value = false;
      scrub.value = lastTick.value;
    }
    watch(playing, (on) => {
      clearInterval(timer);
      if (!on) return;
      // Pressing play at the present rewinds to the start — otherwise there is
      // nothing left to play.
      if (atLive.value) scrub.value = 0;
      timer = setInterval(() => {
        if (scrub.value >= lastTick.value) { playing.value = false; return; }
        scrub.value += 1;
      }, 420);
    });
    watch(scrub, () => { loadGraph(); loadNarrative(); });
    watch(goalId, async () => { await refreshAll(); await nextTick(); fit(); });

    onMounted(async () => {
      await loadMeta();
      await refreshAll();
      await nextTick();
      installZoom();
      fit();
      connect();
      setInterval(() => (now.value = Date.now()), 5000);
      const aside = document.querySelector("aside");
      if (aside) panelBase.value = Math.round(aside.getBoundingClientRect().width);
      window.addEventListener("resize", () => {
        if (panelWidth.value !== null) panelWidth.value = clampPanel(panelWidth.value);
      });
    });

    const renderMd = (src) =>
      window.marked
        ? sanitize(marked.parse(src || ""))
        : `<pre>${(src || "").replace(/[<>&]/g, "")}</pre>`;

    return {
      meta, graph, frontier, narrative, timeline, goalId, tab, selected, detail,
      live, working, scrub, scrubLabel, playing, tip, error, missing,
      lastTick, atLive, goLive, panelWidth, resizing, startResize, zoomBy,
      edgeLines, radius, nodeOpacity, px, py, pick, hover, directive, fit,
      refreshAll, renderMd, at,
    };
  },
}).mount("#app");
