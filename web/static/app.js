// The passkey dialog: the server's options in, the device's answer out (WebAuthn). Nothing else runs here.
// The options are fetched when the page opens, so the tap goes straight to Face ID / Windows Hello: Safari opens
// the dialog only right on a tap, and a request to the server in between used to cost it (07.11, the iPhone).
"use strict";

const FRESH_MS = 240000;            // the server keeps a request for 5 minutes

function fromB64url(s) {
  const pad = "=".repeat((4 - (s.length % 4)) % 4);
  const bin = atob((s + pad).replace(/-/g, "+").replace(/_/g, "/"));
  return Uint8Array.from(bin, (c) => c.charCodeAt(0)).buffer;
}

function toB64url(buf) {
  let bin = "";
  new Uint8Array(buf).forEach((b) => { bin += String.fromCharCode(b); });
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST", credentials: "same-origin",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
  });
  const text = await res.text();
  let json = {};
  try { json = JSON.parse(text); } catch (e) { json = { error: text }; }
  if (!res.ok) throw new Error(json.error || res.statusText);
  return json;
}

function say(text) {
  const el = document.getElementById("msg");
  if (el) el.textContent = text;
}

const HINT = " Если страница открыта из Telegram — открой её в Safari (адрес вручную или «Открыть в Safari»); " +
  "на iPhone должна быть включена «Связка ключей iCloud» (Настройки → твоё имя → iCloud → Пароли).";

function report(where, e) {                                     // the server's log learns why a dialog failed
  post("/webauthn/client-error", { where: where, name: (e && e.name) || "", message: String((e && e.message) || e).slice(0, 300) })
    .catch(() => {});
}

function why(e) {
  if (e && e.name === "NotAllowedError") return "отменено или время вышло — нажми ещё раз." + HINT;
  if (e && e.name === "InvalidStateError") return "на этом устройстве ключ для сайта уже есть";
  return (e && (e.name ? e.name + ": " : "") + e.message) || String(e);
}

function prepare(kind) {
  const url = kind === "register" ? "/webauthn/register/options" : "/webauthn/login/options";
  const state = { at: 0, options: null };
  const load = async () => {
    const o = await post(url);
    o.challenge = fromB64url(o.challenge);
    if (o.user) o.user.id = fromB64url(o.user.id);
    (o.excludeCredentials || []).forEach((c) => { c.id = fromB64url(c.id); });
    (o.allowCredentials || []).forEach((c) => { c.id = fromB64url(c.id); });
    state.options = o;
    state.at = Date.now();
  };
  if (!window.PublicKeyCredential) {
    say("Этот браузер не умеет passkey." + HINT);
    report(kind, { name: "NoWebAuthn", message: navigator.userAgent });
  } else if (PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable) {
    PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable().then((ok) => {
      if (!ok) {
        say("Face ID / Windows Hello для passkey здесь недоступен." + HINT);
        report(kind, { name: "NoPlatformAuthenticator", message: navigator.userAgent });
      }
    }).catch(() => {});
  }
  load().catch((e) => say("Не получилось подготовить вход: " + why(e)));
  const take = async () => {
    if (!state.options || Date.now() - state.at > FRESH_MS) {   // stale: fetch again, the next tap opens the dialog
      say("Готовлю запрос…");
      try { await load(); say("Нажми ещё раз."); } catch (e) { say("Не получилось: " + why(e)); }
      return null;
    }
    const o = state.options;
    state.options = null;                                       // one request, one dialog: the server keeps one
    return o;
  };
  take.again = () => load().catch(() => {});                    // after a failure: the next attempt's request
  return take;
}

async function register(take) {
  const o = await take();
  if (!o) return;
  try {
    const cred = await navigator.credentials.create({ publicKey: o });
    await post("/webauthn/register/verify", {
      id: cred.id, rawId: toB64url(cred.rawId), type: cred.type,
      authenticatorAttachment: cred.authenticatorAttachment || null,
      clientExtensionResults: cred.getClientExtensionResults ? cred.getClientExtensionResults() : {},
      response: {
        clientDataJSON: toB64url(cred.response.clientDataJSON),
        attestationObject: toB64url(cred.response.attestationObject),
        transports: cred.response.getTransports ? cred.response.getTransports() : [],
      },
    });
    location.href = "/";
  } catch (e) { say("Не получилось: " + why(e)); report("register", e); take.again(); }
}

async function login(take) {
  const o = await take();
  if (!o) return;
  try {
    const cred = await navigator.credentials.get({ publicKey: o });
    await post("/webauthn/login/verify", {
      id: cred.id, rawId: toB64url(cred.rawId), type: cred.type,
      authenticatorAttachment: cred.authenticatorAttachment || null,
      clientExtensionResults: cred.getClientExtensionResults ? cred.getClientExtensionResults() : {},
      response: {
        clientDataJSON: toB64url(cred.response.clientDataJSON),
        authenticatorData: toB64url(cred.response.authenticatorData),
        signature: toB64url(cred.response.signature),
        userHandle: cred.response.userHandle ? toB64url(cred.response.userHandle) : null,
      },
    });
    location.href = "/";
  } catch (e) { say("Не получилось: " + why(e)); report("login", e); take.again(); }
}


// ── the pages: bars sized from data- attributes (CSSOM, so the CSP needs no inline style), charts, theme, motion,
//    notifications ──

const CALM = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function store(key, value) {           // this device's choices; nothing breaks when storage is refused
  try {
    if (value === undefined) return localStorage.getItem("rbry:" + key);
    localStorage.setItem("rbry:" + key, value);
  } catch (e) { return null; }
  return null;
}

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function theme() {                     // the chart colours of the theme in force
  return {
    ink: css("--ink"), ink2: css("--ink-2"), muted: css("--muted"), grid: css("--chart-grid"), line: css("--hair-2"),
    up: css("--up"), down: css("--down"), upFill: css("--up-fill"), downFill: css("--down-fill"),
    hub: css("--hub"), berry: css("--berry"), band1: css("--band-1"), band2: css("--band-2"), label: css("--chart-label"),
    font: "'JetBrains Mono', ui-monospace, monospace",
  };
}

function sizes() {                     // bars grow from zero a frame after the page shows (CSS transition on width/left)
  const apply = () => {
    document.querySelectorAll("[data-w]").forEach((el) => { el.style.width = Math.max(0, Math.min(100, +el.dataset.w)) + "%"; });
    document.querySelectorAll("[data-left]").forEach((el) => { el.style.left = "calc(" + Math.max(0, Math.min(100, +el.dataset.left)) + "% - 1px)"; });
  };
  if (CALM) { apply(); return; }
  requestAnimationFrame(() => requestAnimationFrame(apply));
  setTimeout(apply, 900);              // a background tab without frames still ends with the right widths
}

function fmtCount(v, fmt) {
  if (fmt === "pct") return (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2) + "%";
  if (fmt === "usd-int") return "$" + Math.floor(Math.abs(v)).toLocaleString("ru-RU").replace(/ /g, " ");
  return "$" + Math.abs(v).toLocaleString("ru-RU", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function counters() {                  // numbers count up to their value; the one that moved since the last visit flashes
  document.querySelectorAll("[data-count]").forEach((el) => {
    const to = +el.dataset.count, fmt = el.dataset.fmt || "usd", final = el.textContent, key = el.dataset.key;
    if (key) {
      const before = store(key);
      store(key, String(to));
      if (before !== null && +before !== to) el.classList.add(+before < to ? "flash-up" : "flash-down");
    }
    if (CALM || !isFinite(to)) return;
    const from = fmt === "pct" ? 0 : to * 0.92, t0 = performance.now(), dur = 1100;
    const step = (now) => {
      const k = Math.min(1, (now - t0) / dur), ease = 1 - Math.pow(1 - k, 4);
      el.textContent = k < 1 ? fmtCount(from + (to - from) * ease, fmt) : final;
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
    setTimeout(() => { el.textContent = final; }, dur + 400);     // no frames (a background tab): the true value anyway
  });
}

function reveal() {                    // sections below the fold rise in as they scroll into view
  if (CALM || !("IntersectionObserver" in window)) return;
  const items = [...document.querySelectorAll("main .sec, main .hud")].filter((el) => el.getBoundingClientRect().top > window.innerHeight);
  if (!items.length) return;
  const io = new IntersectionObserver((entries) => entries.forEach((en) => {
    if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
  }), { rootMargin: "0px 0px -8% 0px" });
  items.forEach((el) => { el.classList.add("rv"); io.observe(el); });
  setTimeout(() => items.forEach((el) => el.classList.add("in")), 4000);   // never leave anything hidden
}

function money(v) {
  const a = Math.abs(v);
  const s = a >= 1 ? a.toLocaleString("ru-RU", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : a.toPrecision(4);
  return (v < 0 ? "−$" : "$") + s;
}

function when(t) {                     // the chart's time is already Berlin's, written as if UTC
  const d = new Date(t * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return p(d.getUTCDate()) + "." + p(d.getUTCMonth() + 1) + " " + p(d.getUTCHours()) + ":" + p(d.getUTCMinutes());
}

function baseChart(el, T, opts) {
  const LC = window.LightweightCharts;
  return LC.createChart(el, Object.assign({
    autoSize: true,
    layout: { background: { type: "solid", color: "transparent" }, textColor: T.muted, fontFamily: T.font,
              fontSize: 11, attributionLogo: false },
    grid: { vertLines: { visible: false }, horzLines: { color: T.grid } },
    rightPriceScale: { borderColor: T.line, scaleMargins: { top: 0.12, bottom: 0.08 } },
    timeScale: { borderColor: T.line, timeVisible: true, secondsVisible: false, rightOffset: 2 },
    crosshair: { mode: LC.CrosshairMode.Normal,
                 vertLine: { color: T.berry, width: 1, style: 3, labelBackgroundColor: T.label },
                 horzLine: { color: T.hub, width: 1, style: 3, labelBackgroundColor: T.label } },
    localization: { locale: "ru-RU", timeFormatter: when },
    handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
  }, opts || {}));
}

function readout(el, chart, fmt) {
  const box = el.querySelector(".readout");
  if (!box) return;
  chart.subscribeCrosshairMove((p) => {
    if (!p || !p.time || !p.seriesData || p.seriesData.size === 0) { box.textContent = ""; return; }
    box.textContent = fmt(p);
  });
}

function priceFormat(data) {
  const v = data.length ? data[data.length - 1].close : 1;
  const digits = v >= 1000 ? 1 : v >= 10 ? 2 : v >= 1 ? 4 : v >= 0.01 ? 5 : 7;
  return { type: "price", precision: digits, minMove: Math.pow(10, -digits) };
}

const DRAW = {
  candles(el, p, T) {
    const LC = window.LightweightCharts;
    const chart = baseChart(el, T);
    const fut = p.future || null;
    const levels = (p.lines || []).map((l) => l.price);
    const keep = (orig) => {                                  // the trade's levels and its corridor stay in sight
      const r = orig();
      if (!r || !r.priceRange || !levels.length) return r;
      return { priceRange: { minValue: Math.min(r.priceRange.minValue, ...levels), maxValue: Math.max(r.priceRange.maxValue, ...levels) },
               margins: r.margins };
    };
    const s = chart.addSeries(LC.CandlestickSeries, {
      upColor: T.up, downColor: T.down, borderVisible: false, wickUpColor: T.up, wickDownColor: T.down,
      priceFormat: priceFormat(p.data), priceLineVisible: false, autoscaleInfoProvider: keep });
    s.setData(fut ? p.data.concat(fut.whitespace.map((t) => ({ time: t }))) : p.data);
    if (p.markers && p.markers.length) LC.createSeriesMarkers(s, p.markers);
    const pal = { "#e8edf6": T.ink, "#f6465d": T.down, "#3dae5b": T.up, "#f5b941": css("--warn") };
    (p.lines || []).forEach((l) => s.createPriceLine({ price: l.price, color: pal[l.color] || l.color, lineWidth: 1,
                                                       lineStyle: l.style, axisLabelVisible: true, title: l.title }));
    if (fut) {
      fut.bands.forEach((b) => ["up", "down"].forEach((side) => {
        const line = chart.addSeries(LC.LineSeries, { color: b.k > 1 ? T.band2 : T.band1, lineWidth: 1, lineStyle: b.k > 1 ? 1 : 2,
          priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false, priceFormat: priceFormat(p.data) });
        line.setData(b[side]);
      }));
      if (fut.expect) {
        const ex = chart.addSeries(LC.LineSeries, { color: T.hub, lineWidth: 2, lineStyle: 2, priceLineVisible: false,
          lastValueVisible: true, crosshairMarkerVisible: false, priceFormat: priceFormat(p.data) });
        ex.setData([fut.expect.from, fut.expect.to]);
        LC.createSeriesMarkers(ex, [{ time: fut.expect.to.time, position: "inBar", color: T.hub, shape: "circle", text: fut.expect.text }]);
      }
    }
    readout(el, chart, (q) => {
      const c = q.seriesData.get(s);
      if (!c || c.open === undefined) return when(q.time) + "  (будущее)";
      const ch = c.open ? ((c.close / c.open - 1) * 100) : 0;
      return when(q.time) + "  О " + c.open + "  М " + c.high + "  Н " + c.low + "  З " + c.close + "  " +
             (ch >= 0 ? "+" : "−") + Math.abs(ch).toFixed(2) + "%";
    });
    chart.timeScale().fitContent();
    if (p.visible) { try { chart.timeScale().setVisibleRange(p.visible); } catch (e) {} }
    return chart;
  },
  balance(el, p, T) {
    const LC = window.LightweightCharts;
    const chart = baseChart(el, T);
    const s = chart.addSeries(LC.BaselineSeries, {
      baseValue: { type: "price", price: p.base },
      topLineColor: T.up, topFillColor1: T.upFill, topFillColor2: "rgba(0,0,0,0)",
      bottomLineColor: T.down, bottomFillColor1: "rgba(0,0,0,0)", bottomFillColor2: T.downFill,
      lineWidth: 2, priceFormat: { type: "price", precision: 2, minMove: 0.01 }, priceLineVisible: false });
    s.setData(p.data);
    s.createPriceLine({ price: p.base, color: T.muted, lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "старт" });
    readout(el, chart, (q) => { const v = q.seriesData.get(s); return v ? when(q.time) + "  " + money(v.value) +
      "  " + (v.value >= p.base ? "+" : "−") + Math.abs((v.value / p.base - 1) * 100).toFixed(2) + "%" : ""; });
    chart.timeScale().fitContent();
    return chart;
  },
  compare(el, p, T) {
    const LC = window.LightweightCharts;
    const chart = baseChart(el, T);
    const pal = { account: T.ink, nostop: css("--ember"), selector: T.hub, c1: css("--violet") };
    const all = p.series.map((x) => {
      const s = chart.addSeries(LC.LineSeries, { color: pal[x.key] || x.color, lineWidth: x.key === "account" ? 3 : 2,
        lineType: x.key === "c1" ? 1 : 0,
        priceFormat: { type: "custom", formatter: (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2) + "%" },
        lastValueVisible: true, priceLineVisible: false, title: "" });
      s.setData(x.data);
      return [x, s];
    });
    if (all.length) all[0][1].createPriceLine({ price: 0, color: T.muted, lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
    readout(el, chart, (q) => when(q.time) + "  " + all.map(([x, s]) => {
      const v = q.seriesData.get(s); return v ? x.label + " " + (v.value >= 0 ? "+" : "−") + Math.abs(v.value).toFixed(2) + "%" : "";
    }).filter(Boolean).join(" · "));
    chart.timeScale().fitContent();
    return chart;
  },
  line(el, p, T) {
    const LC = window.LightweightCharts;
    const chart = baseChart(el, T);
    const s = chart.addSeries(LC.AreaSeries, { lineColor: T.ink, topColor: css("--area-fill"),
      bottomColor: "rgba(0,0,0,0)", lineWidth: 2, priceLineVisible: false,
      priceFormat: { type: "price", precision: 0, minMove: 1 } });
    s.setData(p.data);
    readout(el, chart, (q) => { const v = q.seriesData.get(s); return v ? when(q.time) + "  " + p.label + " " + money(v.value) : ""; });
    chart.timeScale().fitContent();
    return chart;
  },
  bars(el, p, T) {
    const LC = window.LightweightCharts;
    const chart = baseChart(el, T, { rightPriceScale: { borderColor: T.line, scaleMargins: { top: 0.15, bottom: 0 } } });
    const ember = css("--ember");
    const s = chart.addSeries(LC.HistogramSeries, { color: ember, priceLineVisible: false, lastValueVisible: false,
      priceFormat: { type: "price", precision: 0, minMove: 1 } });
    s.setData(p.data.map((d) => ({ time: d.time, value: d.value,
      color: p.threshold != null && d.value >= p.threshold ? ember : css("--ember-soft") })));
    if (p.threshold != null) s.createPriceLine({ price: p.threshold, color: T.ink2, lineWidth: 1, lineStyle: 2,
                                                 axisLabelVisible: true, title: "нужно" });
    readout(el, chart, (q) => { const v = q.seriesData.get(s); return v ? when(q.time) + "  " + p.label + ": " + v.value : ""; });
    chart.timeScale().fitContent();
    return chart;
  },
};

function drawAll(wipe) {
  if (!window.LightweightCharts) return;
  const T = theme();
  document.querySelectorAll(".lw[data-chart]").forEach((el) => {
    let p;
    try { p = el.payload || JSON.parse(el.dataset.chart); } catch (e) { return; }
    const draw = DRAW[p.kind];
    if (!draw || !(p.data || p.series)) return;
    if (el.chart) { try { el.chart.remove(); } catch (e) {} el.chart = null; }
    try {
      el.chart = draw(el, p, T);
      el.payload = p;
      if (wipe && !CALM && el.animate) {          // the chart wipes in from the left, like a line being drawn
        const a = el.animate([{ clipPath: "inset(0 100% 0 0)" }, { clipPath: "inset(0 0 0 0)" }],
                             { duration: 1300, easing: "cubic-bezier(.16,1,.3,1)" });
        setTimeout(() => { try { a.finish(); } catch (e) {} }, 1600);
      }
    } catch (e) { el.textContent = "График не построился: " + e.message; }
  });
}

function ranges() {                    // 24ч / 7д / всё (balance), сделка / 3д / 7д / 30д (candles) for the card's chart
  document.querySelectorAll(".ranges").forEach((box) => {
    const scope = box.closest(".hud, .sec, section") || document;
    const el = scope.querySelector(".lw");
    box.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      box.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      if (!el || !el.chart || !el.payload) return;
      const p = el.payload, ts = el.chart.timeScale();
      if (b.dataset.span === "trade" && p.visible) { ts.setVisibleRange(p.visible); return; }
      const days = +b.dataset.days;
      const data = p.data || [];
      if (!days || !data.length) { ts.fitContent(); return; }
      const to = p.visible ? p.visible.to : data[data.length - 1].time;
      ts.setVisibleRange({ from: to - days * 86400, to: to });
    }));
  });
}

function coinFilter() {
  const input = document.getElementById("coin-filter");
  if (!input) return;
  const rows = document.querySelectorAll(".coin-row");
  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    rows.forEach((r) => { r.hidden = q && !r.dataset.name.includes(q); });
  });
}

// the theme: dark, light, or the system's; charts redrawn in the new colours
function applyTheme(choice) {
  const light = choice === "light" || (choice === "system" && window.matchMedia("(prefers-color-scheme: light)").matches);
  document.documentElement.setAttribute("data-theme", light ? "light" : "dark");
  document.querySelectorAll("[data-theme-set]").forEach((b) => b.classList.toggle("on", b.dataset.themeSet === choice));
  drawAll(false);
}

function themePicker() {
  const choice = store("theme") || "system";
  document.querySelectorAll("[data-theme-set]").forEach((b) => {
    b.classList.toggle("on", b.dataset.themeSet === choice);
    b.addEventListener("click", (ev) => { ev.preventDefault(); store("theme", b.dataset.themeSet); applyTheme(b.dataset.themeSet); });
  });
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
      if ((store("theme") || "system") === "system") applyTheme("system");
    });
  }
}

// notifications while the site is open: new entries, exits, skips, a mass panic, urgent stories, C1's decision
const KINDS = ["entry", "exit", "skip", "panic", "urgent", "c1"];

function notifyPrefs() {
  let p = {};
  try { p = JSON.parse(store("notify") || "{}"); } catch (e) {}
  KINDS.forEach((k) => { if (p[k] === undefined) p[k] = true; });
  if (p.sound === undefined) p.sound = false;
  return p;
}

function notifySettings() {
  const prefs = notifyPrefs();
  document.querySelectorAll("[data-notify]").forEach((box) => {
    box.checked = !!prefs[box.dataset.notify];
    box.addEventListener("change", () => { const p = notifyPrefs(); p[box.dataset.notify] = box.checked; store("notify", JSON.stringify(p)); });
  });
  const test = document.querySelector("[data-notify-test]");
  if (test) test.addEventListener("click", () => toast({ kind: "entry", title: "Вход: BTC лонг", text: "пример уведомления · паника · $20.00", href: "/" }));
}

function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.value = 880; g.gain.value = 0.05; o.connect(g); g.connect(ctx.destination); o.start();
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.25); o.stop(ctx.currentTime + 0.26);
  } catch (e) {}
}

function toast(ev) {
  const box = document.querySelector(".toasts");
  if (!box) return;
  const t = document.createElement("a");
  t.className = "toast k-" + ev.kind;
  t.href = ev.href || "#";
  const b = document.createElement("b"); b.textContent = ev.title;
  const s = document.createElement("span"); s.textContent = ev.text || "";
  t.appendChild(b); t.appendChild(s);
  box.appendChild(t);
  requestAnimationFrame(() => t.classList.add("in"));
  setTimeout(() => { t.classList.remove("in"); setTimeout(() => t.remove(), 500); }, 9000);
  if (notifyPrefs().sound) beep();
}

function feed() {
  if (!document.body.dataset.feed) return;
  const poll = async () => {
    if (document.visibilityState !== "visible") return;
    const since = +(store("feed-since") || 0);
    let res;
    try { res = await (await fetch("/api/feed?since=" + since, { credentials: "same-origin" })).json(); } catch (e) { return; }
    if (!since) { store("feed-since", String(res.now)); return; }     // the first visit: nothing old pops up
    const prefs = notifyPrefs();
    const fresh = (res.events || []).filter((x) => prefs[x.kind]).reverse().slice(-4);
    fresh.forEach((x, i) => setTimeout(() => toast(x), i * 700));
    const newest = Math.max(since, ...(res.events || []).map((x) => x.ts));
    store("feed-since", String(newest));
  };
  poll();
  setInterval(poll, 60000);
  document.addEventListener("visibilitychange", poll);
}

document.addEventListener("DOMContentLoaded", () => {
  const r = document.getElementById("passkey-register");
  if (r) { const take = prepare("register"); r.addEventListener("click", () => register(take)); }
  const l = document.getElementById("passkey-login");
  if (l) { const take = prepare("login"); l.addEventListener("click", () => login(take)); }
  sizes();
  counters();
  reveal();
  coinFilter();
  drawAll(true);
  ranges();
  themePicker();
  notifySettings();
  feed();
  document.querySelectorAll("details.menu").forEach((d) => d.addEventListener("toggle", () => {   // one menu open at a time
    if (d.open) document.querySelectorAll("details.menu").forEach((o) => { if (o !== d) o.open = false; });
  }));
  document.addEventListener("click", (ev) => {
    document.querySelectorAll("details.menu[open]").forEach((d) => { if (!d.contains(ev.target)) d.open = false; });
  });
});
