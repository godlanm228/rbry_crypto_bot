// The passkey dialog: the server's options in, the device's answer out (WebAuthn). Nothing else runs here.
"use strict";

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

async function register() {
  try {
    const o = await post("/webauthn/register/options");
    o.challenge = fromB64url(o.challenge);
    o.user.id = fromB64url(o.user.id);
    (o.excludeCredentials || []).forEach((c) => { c.id = fromB64url(c.id); });
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
  } catch (e) { say("Не получилось: " + e.message); }
}

async function login() {
  try {
    const o = await post("/webauthn/login/options");
    o.challenge = fromB64url(o.challenge);
    (o.allowCredentials || []).forEach((c) => { c.id = fromB64url(c.id); });
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
  } catch (e) { say("Не получилось: " + e.message); }
}

document.addEventListener("DOMContentLoaded", () => {
  const r = document.getElementById("passkey-register");
  if (r) r.addEventListener("click", register);
  const l = document.getElementById("passkey-login");
  if (l) l.addEventListener("click", login);
});
