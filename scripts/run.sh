#!/usr/bin/env bash
set -euo pipefail

# ---- Config (override with env or flags) ----
PORT_API="${PORT_API:-8000}"
PORT_UI="${PORT_UI:-5173}"
SINCE_HOURS="${SINCE_HOURS:-720}"      # 30 days
SUM_LIMIT="${SUM_LIMIT:-400}"
EMB_LIMIT="${EMB_LIMIT:-1000}"
MODEL_SUM="${MODEL_SUM:-gpt-4o-mini}"

DO_INGEST=false       # placeholder; wire to your crawler if/when ready
DO_SUMMARIZE=false
DO_EMBED=true
NO_UI=false
NO_OPEN=false
INSTALL_DEPS=false

# ---- Helpers ----
die() { echo "Error: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
in_use() { lsof -i ":$1" >/dev/null 2>&1; }
find_free_port() {
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
}

usage() {
cat <<USAGE
Usage: $0 [options]

Options:
  --since-hours N         Default: ${SINCE_HOURS}
  --sum-limit N           Default: ${SUM_LIMIT}
  --emb-limit N           Default: ${EMB_LIMIT}
  --model-sum NAME        Default: ${MODEL_SUM}
  --with-ingest           Run ingestion step (placeholder hook)
  --no-summarize          Skip summarization
  --no-embed              Skip embeddings
  --no-ui                 Do not start static UI server
  --no-open               Do not open browser automatically
  --install-deps          Run 'poetry install' first
  --port-api N            Default: ${PORT_API}
  --port-ui N             Default: ${PORT_UI}
  -h, --help              Show this help

Environment overrides:
  OPENAI_API_KEY, PORT_API, PORT_UI, SINCE_HOURS, SUM_LIMIT, EMB_LIMIT, MODEL_SUM
USAGE
}

# ---- Parse flags ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --since-hours) SINCE_HOURS="$2"; shift 2 ;;
    --sum-limit)   SUM_LIMIT="$2"; shift 2 ;;
    --emb-limit)   EMB_LIMIT="$2"; shift 2 ;;
    --model-sum)   MODEL_SUM="$2"; shift 2 ;;
    --with-ingest) DO_INGEST=true; shift ;;
    --no-summarize) DO_SUMMARIZE=false; shift ;;
    --no-embed)     DO_EMBED=false; shift ;;
    --no-ui)        NO_UI=true; shift ;;
    --no-open)      NO_OPEN=true; shift ;;
    --install-deps) INSTALL_DEPS=true; shift ;;
    --port-api)     PORT_API="$2"; shift 2 ;;
    --port-ui)      PORT_UI="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *) die "Unknown option: $1";;
  esac
done

# ---- Project root ----
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---- Load .env if present via your bootstrap helper (optional) ----
if [[ -f "scripts/_bootstrap_env.py" || -f "scripts/_bootstrap_env/__init__.py" ]]; then
  :  # Python scripts already call load_env()
fi

# ---- Sanity checks ----
[[ -n "${OPENAI_API_KEY:-}" ]] || echo "Warning: OPENAI_API_KEY is not set (embedding will fallback to keyword-only)."
have poetry || die "Poetry is required. Please install: https://python-poetry.org/docs/"

# ---- Install deps (optional) ----
if $INSTALL_DEPS; then
  echo "==> Installing dependencies with Poetry..."
  poetry install --no-interaction --no-ansi
fi

# ---- Optional ingestion hook (customize to your pipeline if/when ready) ----
if $DO_INGEST; then
  echo "==> Ingestion: customize this block to call your crawler."
  # poetry run python -m scripts.crawl --max-new 200 --sources reuters-markets,marketwatch-latest
fi

# ---- Summarization ----
if $DO_SUMMARIZE; then
  echo "==> Summarizing articles (model=$MODEL_SUM, since=${SINCE_HOURS}h, limit=${SUM_LIMIT})..."
  poetry run python -m scripts.summarize \
    --model "$MODEL_SUM" \
    --since-hours "$SINCE_HOURS" \
    --limit "$SUM_LIMIT" || echo "Summarization step returned non-zero; continuing."
fi

# ---- Embeddings ----
if $DO_EMBED; then
  echo "==> Generating embeddings (since=${SINCE_HOURS}h, limit=${EMB_LIMIT})..."
  poetry run python -m scripts.embed \
    --since-hours "$SINCE_HOURS" \
    --limit "$EMB_LIMIT" || echo "Embedding step returned non-zero; continuing."
fi

# ---- Ensure minimal UI exists (static/search.html) ----
STATIC_DIR="static"
INDEX_HTML="${STATIC_DIR}/search.html"
if ! $NO_UI; then
  mkdir -p "$STATIC_DIR"
  if [[ ! -f "$INDEX_HTML" ]]; then
cat > "$INDEX_HTML" <<'HTML'
<!doctype html>
<meta charset="utf-8"/>
<title>FinGPT Search</title>
<style>
  body { font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto; margin: 24px; max-width: 980px; }
  header { display:flex; gap:8px; align-items:center; margin-bottom: 12px; }
  input[type=text]{ flex:1; padding:10px 12px; border:1px solid #ccc; border-radius:10px; }
  button{ padding:10px 14px; border-radius:10px; border:1px solid #ccc; background:#fafafa; cursor:pointer;}
  .hit { border: 1px solid #ddd; border-radius: 12px; padding: 12px; margin: 10px 0; background: #fff; }
  .meta { color: #666; font-size: 12px; margin-top: 4px; }
</style>
<header>
  <input id="q" type="text" placeholder="Type keywords (e.g., fed rates, earnings, oil)..."/>
  <button id="go">Search</button>
</header>
<div id="out"></div>
<script>
const API = `http://localhost:${location.search.match(/api=(\d+)/)?.[1] ?? 8000}`;
async function run(p=1) {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const url = new URL(API + '/search');
  url.searchParams.set('q', q);
  url.searchParams.set('page', p);
  url.searchParams.set('page_size', 20);
  const r = await fetch(url);
  const js = await r.json();
  const out = document.getElementById('out');
  out.innerHTML = `<p><b>${js.returned}</b> / ${js.total_candidates} results</p>`;
  js.hits.forEach(h => {
    const d = document.createElement('div');
    d.className = 'hit';
    d.innerHTML = `
      <div><a href="${h.url}" target="_blank"><b>${h.title || '(untitled)'}</b></a></div>
      <div class="meta">${h.source ?? ''} • ${h.published_at ?? 'N/A'} • score ${Number(h.score).toFixed(3)} ${h.cluster_label ? '• ' + h.cluster_label : ''}</div>
      ${h.summary_text ? `<div style="margin-top:6px; white-space:pre-wrap;">${h.summary_text}</div>` : ''}
    `;
    out.appendChild(d);
  });
}
document.getElementById('go').onclick = () => run(1);
document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') run(1); });
</script>
HTML
  fi
fi

# ---- Start API & UI ----
PIDS=()

# Auto-pick a free API port if requested is busy
if in_use "${PORT_API}"; then
  echo "⚠️  Port ${PORT_API} is in use; selecting a free API port..."
  PORT_API="$(find_free_port)"
fi

echo "==> Starting API on :${PORT_API} ..."
poetry run uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT_API}" --reload &
PIDS+=($!)

if ! $NO_UI; then
  # Auto-pick a free UI port if requested is busy
  if in_use "${PORT_UI}"; then
    echo "⚠️  Port ${PORT_UI} is in use; selecting a free UI port..."
    PORT_UI="$(find_free_port)"
  fi

  echo "==> Serving static UI on :${PORT_UI} ..."
  ( cd "$STATIC_DIR" && python3 -m http.server "${PORT_UI}" ) &
  PIDS+=($!)
fi

# ---- Open browser ----
if ! $NO_UI && ! $NO_OPEN; then
  URL="http://localhost:${PORT_UI}/search.html?api=${PORT_API}"
  echo "==> UI available at: $URL"
  if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$URL"
  elif have xdg-open; then
    xdg-open "$URL" >/dev/null 2>&1 || true
  fi
else
  if ! $NO_UI; then
    echo "==> UI available at: http://localhost:${PORT_UI}/search.html?api=${PORT_API}"
  fi
  echo "==> API available at: http://localhost:${PORT_API}/docs"
fi

# ---- UI + API availability ----
if ! $NO_UI; then
  # write config.js so the UI knows which API port to use
  cat > "${STATIC_DIR}/config.js" <<EOF
// generated by run.sh
window.FINGPT_API = "http://localhost:${PORT_API}";
EOF

  URL="http://localhost:${PORT_UI}/search.html"
  echo "==> UI available at: $URL"
fi

echo "==> API available at: http://localhost:${PORT_API}/docs"

# ---- Auto-open browser ----
if ! $NO_UI && ! $NO_OPEN; then
  if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$URL"
  elif have xdg-open; then
    xdg-open "$URL" >/dev/null 2>&1 || true
  fi
fi


# ---- Cleanup on exit ----
cleanup() {
  echo ""
  echo "==> Shutting down..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

echo "==> Running. Press Ctrl+C to stop."
wait
