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

function why(e) {
  if (e && e.name === "NotAllowedError") return "отменено или время вышло — нажми ещё раз";
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
  } catch (e) { say("Не получилось: " + why(e)); take.again(); }
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
  } catch (e) { say("Не получилось: " + why(e)); take.again(); }
}

document.addEventListener("DOMContentLoaded", () => {
  const r = document.getElementById("passkey-register");
  if (r) { const take = prepare("register"); r.addEventListener("click", () => register(take)); }
  const l = document.getElementById("passkey-login");
  if (l) { const take = prepare("login"); l.addEventListener("click", () => login(take)); }
});
