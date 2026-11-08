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

// ── the pages: bars sized from data- attributes (CSSOM, so the CSP needs no inline style), charts, search ──

const THEME = {
  ink: "#e8edf6", ink2: "#a6b2c6", muted: "#6c7a91", grid: "rgba(255,255,255,0.045)", line: "#24304a",
  up: "#3dae5b", down: "#f6465d", upFill: "rgba(61,174,91,0.28)", downFill: "rgba(246,70,93,0.22)",
  font: "'JetBrains Mono', ui-monospace, monospace",
};

function sizes() {
  document.querySelectorAll("[data-w]").forEach((el) => { el.style.width = Math.max(0, Math.min(100, +el.dataset.w)) + "%"; });
  document.querySelectorAll("[data-left]").forEach((el) => { el.style.left = "calc(" + Math.max(0, Math.min(100, +el.dataset.left)) + "% - 1.5px)"; });
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

function baseChart(el, opts) {
  const LC = window.LightweightCharts;
  return LC.createChart(el, Object.assign({
    autoSize: true,
    layout: { background: { type: "solid", color: "transparent" }, textColor: THEME.muted, fontFamily: THEME.font,
              fontSize: 11, attributionLogo: false },
    grid: { vertLines: { color: THEME.grid }, horzLines: { color: THEME.grid } },
    rightPriceScale: { borderColor: THEME.line, scaleMargins: { top: 0.12, bottom: 0.08 } },
    timeScale: { borderColor: THEME.line, timeVisible: true, secondsVisible: false, rightOffset: 2 },
    crosshair: { mode: LC.CrosshairMode.Normal,
                 vertLine: { color: "rgba(232,237,246,0.35)", labelBackgroundColor: "#121a28" },
                 horzLine: { color: "rgba(232,237,246,0.35)", labelBackgroundColor: "#121a28" } },
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
  candles(el, p) {
    const LC = window.LightweightCharts;
    const chart = baseChart(el);
    const s = chart.addSeries(LC.CandlestickSeries, {
      upColor: THEME.up, downColor: THEME.down, borderVisible: false, wickUpColor: THEME.up, wickDownColor: THEME.down,
      priceFormat: priceFormat(p.data), priceLineVisible: false,
      autoscaleInfoProvider: (orig) => {                    // the trade's levels stay in sight (a far target too)
        const r = orig();
        const lv = (p.lines || []).map((l) => l.price);
        if (!r || !r.priceRange || !lv.length) return r;
        return { priceRange: { minValue: Math.min(r.priceRange.minValue, ...lv), maxValue: Math.max(r.priceRange.maxValue, ...lv) },
                 margins: r.margins };
      } });
    s.setData(p.data);
    if (p.markers && p.markers.length) LC.createSeriesMarkers(s, p.markers);
    (p.lines || []).forEach((l) => s.createPriceLine({ price: l.price, color: l.color, lineWidth: 1, lineStyle: l.style,
                                                       axisLabelVisible: true, title: l.title }));
    readout(el, chart, (q) => {
      const c = q.seriesData.get(s);
      if (!c) return "";
      const ch = c.open ? ((c.close / c.open - 1) * 100) : 0;
      return when(q.time) + "  О " + c.open + "  М " + c.high + "  Н " + c.low + "  З " + c.close + "  " +
             (ch >= 0 ? "+" : "−") + Math.abs(ch).toFixed(2) + "%";
    });
    chart.timeScale().fitContent();
    return chart;
  },
  balance(el, p) {
    const LC = window.LightweightCharts;
    const chart = baseChart(el);
    const s = chart.addSeries(LC.BaselineSeries, {
      baseValue: { type: "price", price: p.base },
      topLineColor: THEME.up, topFillColor1: THEME.upFill, topFillColor2: "rgba(61,174,91,0.02)",
      bottomLineColor: THEME.down, bottomFillColor1: "rgba(246,70,93,0.02)", bottomFillColor2: THEME.downFill,
      lineWidth: 2, priceFormat: { type: "price", precision: 2, minMove: 0.01 }, priceLineVisible: false });
    s.setData(p.data);
    s.createPriceLine({ price: p.base, color: THEME.muted, lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "старт" });
    readout(el, chart, (q) => { const v = q.seriesData.get(s); return v ? when(q.time) + "  " + money(v.value) +
      "  " + (v.value >= p.base ? "+" : "−") + Math.abs((v.value / p.base - 1) * 100).toFixed(2) + "%" : ""; });
    chart.timeScale().fitContent();
    return chart;
  },
  compare(el, p) {
    const LC = window.LightweightCharts;
    const chart = baseChart(el);
    const all = p.series.map((x) => {
      const s = chart.addSeries(LC.LineSeries, { color: x.color, lineWidth: x.key === "account" ? 3 : 2,
        priceFormat: { type: "custom", formatter: (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2) + "%" },
        lastValueVisible: true, priceLineVisible: false, title: "" });
      s.setData(x.data);
      return [x, s];
    });
    if (all.length) all[0][1].createPriceLine({ price: 0, color: THEME.muted, lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
    readout(el, chart, (q) => when(q.time) + "  " + all.map(([x, s]) => {
      const v = q.seriesData.get(s); return v ? x.label + " " + (v.value >= 0 ? "+" : "−") + Math.abs(v.value).toFixed(2) + "%" : "";
    }).filter(Boolean).join(" · "));
    chart.timeScale().fitContent();
    return chart;
  },
  line(el, p) {
    const LC = window.LightweightCharts;
    const chart = baseChart(el);
    const s = chart.addSeries(LC.AreaSeries, { lineColor: p.color, topColor: "rgba(232,237,246,0.16)",
      bottomColor: "rgba(232,237,246,0.0)", lineWidth: 2, priceLineVisible: false,
      priceFormat: { type: "price", precision: 0, minMove: 1 } });
    s.setData(p.data);
    readout(el, chart, (q) => { const v = q.seriesData.get(s); return v ? when(q.time) + "  " + p.label + " " + money(v.value) : ""; });
    chart.timeScale().fitContent();
    return chart;
  },
  bars(el, p) {
    const LC = window.LightweightCharts;
    const chart = baseChart(el, { rightPriceScale: { borderColor: THEME.line, scaleMargins: { top: 0.15, bottom: 0 } } });
    const s = chart.addSeries(LC.HistogramSeries, { color: p.color, priceLineVisible: false, lastValueVisible: false,
      priceFormat: { type: "price", precision: 0, minMove: 1 } });
    s.setData(p.data.map((d) => ({ time: d.time, value: d.value,
      color: p.threshold != null && d.value >= p.threshold ? p.color : "rgba(217,115,45,0.45)" })));
    if (p.threshold != null) s.createPriceLine({ price: p.threshold, color: THEME.ink2, lineWidth: 1, lineStyle: 2,
                                                 axisLabelVisible: true, title: "нужно" });
    readout(el, chart, (q) => { const v = q.seriesData.get(s); return v ? when(q.time) + "  " + p.label + ": " + v.value : ""; });
    chart.timeScale().fitContent();
    return chart;
  },
};

function charts() {
  if (!window.LightweightCharts) return;
  document.querySelectorAll(".lw[data-chart]").forEach((el) => {
    let p;
    try { p = JSON.parse(el.dataset.chart); } catch (e) { return; }
    const draw = DRAW[p.kind];
    if (!draw || !(p.data || p.series)) return;
    try {
      const chart = draw(el, p);
      el.chart = chart;
      el.payload = p;
    } catch (e) { el.textContent = "График не построился: " + e.message; }
  });
  document.querySelectorAll(".ranges").forEach((box) => {     // 24ч / 7д / всё for the chart in the same card
    const el = box.closest(".card").querySelector(".lw");
    box.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      box.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      if (!el || !el.chart || !el.payload) return;
      const days = +b.dataset.days;
      const data = el.payload.data;
      if (!days || !data.length) { el.chart.timeScale().fitContent(); return; }
      const to = data[data.length - 1].time;
      el.chart.timeScale().setVisibleRange({ from: to - days * 86400, to: to });
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

document.addEventListener("DOMContentLoaded", () => {
  const r = document.getElementById("passkey-register");
  if (r) { const take = prepare("register"); r.addEventListener("click", () => register(take)); }
  const l = document.getElementById("passkey-login");
  if (l) { const take = prepare("login"); l.addEventListener("click", () => login(take)); }
  sizes();
  coinFilter();
  charts();
});
