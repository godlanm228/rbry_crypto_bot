// Before the first paint: the theme the owner chose on this device (or the system's), so a light page never
// flashes dark. A file, not an inline script: the CSP allows scripts from the site alone.
(function () {
  var t = "system";
  try { t = localStorage.getItem("rbry:theme") || "system"; } catch (e) {}
  var light = t === "light" || (t === "system" && window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches);
  document.documentElement.setAttribute("data-theme", light ? "light" : "dark");
})();
