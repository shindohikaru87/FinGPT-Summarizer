// Tiny API helper for your FastAPI backend.
function makeApi(baseUrl) {
  const fetchJson = async (url, params={}) => {
    const u = new URL(url, baseUrl);
    Object.entries(params).forEach(([k,v]) => {
      if (v !== undefined && v !== null) u.searchParams.set(k, String(v));
    });
    const r = await fetch(u.toString());
    if (!r.ok) {
      const txt = await r.text().catch(()=>r.statusText);
      throw new Error(`HTTP ${r.status} — ${txt}`);
    }
    return r.json();
  };

  return {
    search: (args) => fetchJson("/search", args),
    clusters: (args) => fetchJson("/clusters", args),
    health: () => fetchJson("/healthz"),
  };
}
