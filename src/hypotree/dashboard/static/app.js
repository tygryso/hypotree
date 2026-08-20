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
    const graph = reactive({ nodes: [], edges: [], stats: {}, unwired_goal: null, goal_wiring: null });
    const frontier = ref([]);
    const doubt = ref([]);
    const conflicts = ref([]);
    const claims = ref([]);
    const narrative = ref("");
    const timeline = ref({ ticks: [], from: null, to: null });

    const goalId = ref("");
    const tab = ref("path");
    const selected = ref(null);
    const detail = ref(null);
    const evidenceKind = ref("");
    const evidenceQuery = ref("");
    const evidenceOffset = ref(0);
    const evidenceLimit = 10;
    const live = ref(false);
    const busyUntil = ref(0);
    const now = ref(Date.now());
    const scrub = ref(0);
    const playing = ref(false);
    const tip = reactive({ show: false, x: 0, y: 0, node: null });
    const error = ref("");
    const missing = ref([]);
    const bannerDismissed = ref(false);

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
    const activeGoal = computed(() => {
      if (!meta.value || !meta.value.goals.length) return null;
      return meta.value.goals.find((goal) => goal.id === goalId.value) || null;
    });
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

    // ---- belief diff -----------------------------------------------------

    // The scrubber already picks an instant; picking a second one turns the
    // narrative into "what changed between them", which is the question a
    // standup or a PR description actually asks. Same control, used twice.
    const sinceTick = ref(null);
    const since = computed(() => {
      const t = timeline.value.ticks;
      if (sinceTick.value === null || !t.length) return null;
      return t[Math.min(sinceTick.value, t.length - 1)].t;
    });
    const sincePct = computed(() => {
      const max = lastTick.value;
      return max > 0 && sinceTick.value !== null ? (sinceTick.value / max) * 100 : 0;
    });
    const markSince = () => { sinceTick.value = scrub.value; loadNarrative(); };
    const clearSince = () => { sinceTick.value = null; loadNarrative(); };
    const inWindow = (bin) => {
      if (sinceTick.value === null || !activity.value.length) return false;
      const n = timeline.value.ticks.length || 1;
      const start = Math.floor((sinceTick.value / n) * activity.value.length);
      return bin >= start && bin <= binOfScrub.value;
    };

    // ---- activity histogram ---------------------------------------------

    // The run's shape is information — where the bursts were, where it stalled —
    // and a featureless slider track threw all of it away. Ticks are binned by
    // position rather than by wall-clock so an overnight pause does not flatten
    // the whole run into one column.
    const BINS = 64;
    const activity = computed(() => {
      const n = timeline.value.ticks.length;
      if (!n) return [];
      const bins = new Array(Math.min(BINS, Math.max(8, n))).fill(0);
      for (let i = 0; i < n; i++) bins[Math.floor((i / n) * bins.length)] += 1;
      const peak = Math.max(...bins, 1);
      return bins.map((c) => ({ h: Math.round((c / peak) * 100) }));
    });
    const binOfScrub = computed(() => {
      const n = timeline.value.ticks.length;
      if (!n || !activity.value.length) return -1;
      return Math.min(activity.value.length - 1, Math.floor((scrub.value / n) * activity.value.length));
    });
    const cursorPct = computed(() => {
      const max = lastTick.value;
      return max > 0 ? (scrub.value / max) * 100 : 0;
    });

    // Local time, seconds precision. The belief state stores UTC and the reader
    // is looking at their own clock.
    const stamp = (iso) => {
      if (!iso) return "—";
      const d = new Date(iso);
      return isNaN(d) ? String(iso).slice(0, 19).replace("T", " ") : d.toLocaleString();
    };

    // What an experiment cost, in the unit a reader thinks in. Seconds for a
    // unit test, days for a fine-tune — the whole point is that they differ.
    const cost = (s) => {
      if (s === null || s === undefined) return "";
      if (s < 90) return `${s < 1 ? s.toFixed(2) : Math.round(s)}s`;
      if (s < 5400) return `${(s / 60).toFixed(1)}m`;
      if (s < 172800) return `${(s / 3600).toFixed(1)}h`;
      return `${(s / 86400).toFixed(1)}d`;
    };

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
      graph.unwired_goal = g.unwired_goal || null;
      graph.goal_wiring = g.goal_wiring || null;
    }
    async function loadNarrative() {
      // The narrative is pinned to the same instant as the graph, so a rewound
      // picture is never captioned with conclusions it has not reached. With a
      // start marker set it becomes a diff over that window instead.
      const parts = [
        q(),
        at.value ? `at=${encodeURIComponent(at.value)}` : "",
        since.value ? `since=${encodeURIComponent(since.value)}` : "",
      ].filter(Boolean);
      const r = await api(`/api/learning-path${parts.length ? "?" + parts.join("&") : ""}`);
      narrative.value = r.markdown || "";
    }
    async function loadFrontier() {
      const query = q();
      frontier.value = (await api(`/api/frontier?k=5${query ? "&" + query : ""}`)).candidates;
    }
    async function loadDoubt() {
      const query = q();
      doubt.value = (await api(`/api/counterfactual?k=5${query ? "&" + query : ""}`)).beliefs;
    }
    async function loadGovernance() {
      const [conflictData, claimData] = await Promise.all([
        api("/api/conflicts?open_only=true"),
        api("/api/claims"),
      ]);
      conflicts.value = conflictData.conflicts || [];
      claims.value = claimData.claims || [];
    }
    async function loadPanels() {
      const query = q();
      await loadFrontier();
      await loadDoubt();
      await loadGovernance();
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
        await Promise.all([loadMeta(), loadGraph(), loadPanels()]);
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
          return {
            key: `${e.src}->${e.dst}:${e.type}`,
            type: e.type,
            x1: px(a), y1: py(a), x2: px(b), y2: py(b), dead,
          };
        })
        .filter(Boolean)
    );

    const radius = (n) => (n.is_goal ? 13 : 8 + 7 * (n.p_select || 0));
    const nodeLabel = (n) => {
      const label = n.title || n.id;
      return label.length > 20 ? `${label.slice(0, 19)}…` : label;
    };
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
      // `selected` drives the panel, not `detail`. Clearing `detail` to fetch
      // the next node used to mount the whole narrative for one frame before
      // replacing it again, which read as the panel flashing on every click.
      // The card renders from `selected` immediately and fills in on arrival.
      const id = node.id;
      selected.value = id;
      detail.value = null;
      evidenceOffset.value = 0;
      tab.value = "path";
      loadDetail(id)
        .then((d) => {
          // A slow response for a node the reader has already moved off must
          // not overwrite the one they are looking at now.
          if (selected.value === id) detail.value = d;
        })
        .catch((e) => (error.value = String(e.message || e)));
      if (ev) hover(node, ev);
    }
    function detailPath(id) {
      const params = new URLSearchParams({
        limit: String(evidenceLimit),
        offset: String(evidenceOffset.value),
      });
      if (evidenceKind.value) params.set("kind", evidenceKind.value);
      if (evidenceQuery.value.trim()) params.set("q", evidenceQuery.value.trim());
      return `/api/node/${encodeURIComponent(id)}?${params.toString()}`;
    }
    const loadDetail = (id = selected.value) => id ? api(detailPath(id)) : Promise.resolve(null);
    async function applyEvidenceFilters() {
      if (!selected.value) return;
      evidenceOffset.value = 0;
      detail.value = await loadDetail();
    }
    async function evidencePage(delta) {
      if (!selected.value || !detail.value) return;
      evidenceOffset.value = Math.max(0, evidenceOffset.value + delta * evidenceLimit);
      detail.value = await loadDetail();
    }
    function clearSelection() {
      selected.value = null;
      detail.value = null;
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

    const directiveMode = computed(() => (detail.value && detail.value.directive)
      ? detail.value.directive.mode
      : null);

    async function directive(mode) {
      if (!selected.value) return;
      const id = selected.value;
      try {
        await api("/api/directive", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ node_id: id, mode }),
        });
        // A directive changes what is offered, so the frontier and this node's
        // own card are the only things that can have moved. Reloading the whole
        // view would rebuild the narrative and the layout for nothing.
        detail.value = await loadDetail(id);
        await Promise.all([loadGraph(), loadFrontier()]);
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
    watch(goalId, async () => {
      // A dismissal is about one goal, not about the session.
      bannerDismissed.value = false;
      clearSelection();
      await refreshAll();
      await nextTick();
      fit();
    });

    onMounted(async () => {
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
      meta, graph, frontier, doubt, conflicts, claims, narrative, timeline, goalId, activeGoal,
      tab, selected, detail, evidenceKind, evidenceQuery, evidenceOffset, evidenceLimit,
      sinceTick, sincePct, markSince, clearSince, inWindow, cost,
      live, working, scrub, scrubLabel, playing, tip, error, missing,
      lastTick, atLive, goLive, panelWidth, resizing, startResize, zoomBy,
      activity, binOfScrub, cursorPct, stamp, directiveMode, clearSelection, bannerDismissed,
      edgeLines, radius, nodeOpacity, px, py, pick, hover, directive, fit,
      refreshAll, renderMd, at, applyEvidenceFilters, evidencePage, nodeLabel,
    };
  },
}).mount("#app");
