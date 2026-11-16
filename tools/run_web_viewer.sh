#!/usr/bin/env bash
# Start local viewer/API. GPU worker is explicit opt-in.
set -euo pipefail

viewer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$viewer_root"

if [[ -x "$viewer_root/.cache/toolchains/node/bin/node" ]]; then
  export PATH="$viewer_root/.cache/toolchains/node/bin:$PATH"
fi
if [[ ! -x .venv-web/bin/python ]] || ! command -v node >/dev/null; then
  printf '%s\n' 'Prepare .venv-web and Node.js first; see docs/WEB_VIEWER.md.' >&2
  exit 1
fi
if [[ "${VIEWER_START_WORKER:-0}" != 0 && "${VIEWER_START_WORKER:-0}" != 1 ]]; then
  printf '%s\n' 'VIEWER_START_WORKER must be 0 (playback) or 1 (process queued inference).' >&2
  exit 1
fi
if [[ "${VIEWER_NEXT_DIST_DIR:-.next}" != .next || "${VIEWER_API_ORIGIN:-http://127.0.0.1:8000}" != http://127.0.0.1:8000 ]]; then
  printf '%s\n' 'Production launcher requires .next and API http://127.0.0.1:8000; use separate servers for QA.' >&2
  exit 1
fi
if [[ ! -f web/frontend/.next/BUILD_ID ]]; then
  printf '%s\n' 'Build the frontend first: cd web/frontend && npm ci && npm run build' >&2
  exit 1
fi

# Check saved rewrite without importing app/main or creating/migrating SQLite.
.venv-web/bin/python -B -c 'import sys; sys.dont_write_bytecode=True; from tools.check_web_readiness import build_info; build_info("web/frontend/.next","http://127.0.0.1:8000")'
export VIEWER_NEXT_DIST_DIR=.next VIEWER_API_ORIGIN=http://127.0.0.1:8000

viewer_backend_pid=''
viewer_frontend_pid=''
viewer_worker_pid=''
cleanup() {
  trap - EXIT
  if [[ -n "$viewer_worker_pid" ]]; then
    kill "$viewer_worker_pid" 2>/dev/null || true
    wait "$viewer_worker_pid" 2>/dev/null || true
  fi
  if [[ -n "$viewer_backend_pid" ]]; then
    kill "$viewer_backend_pid" 2>/dev/null || true
    wait "$viewer_backend_pid" 2>/dev/null || true
  fi
  if [[ -n "$viewer_frontend_pid" ]]; then
    kill "$viewer_frontend_pid" 2>/dev/null || true
    wait "$viewer_frontend_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

.venv-web/bin/python -m uvicorn web.backend.app.main:app --host 127.0.0.1 --port 8000 &
viewer_backend_pid=$!
NEXT_TELEMETRY_DISABLED=1 node web/frontend/node_modules/next/dist/bin/next start \
  web/frontend --hostname 127.0.0.1 --port 3000 &
viewer_frontend_pid=$!
viewer_processes=("$viewer_backend_pid" "$viewer_frontend_pid")
if [[ "${VIEWER_START_WORKER:-0}" == 1 ]]; then
  .venv-web/bin/python -u tools/run_inference_worker.py &
  viewer_worker_pid=$!
fi
if [[ -z "$viewer_worker_pid" ]]; then
  printf '%s\n' 'Worker OFF: new/existing queued jobs wait; saved-artifact playback does not run inference.'
fi
printf '%s\n' 'Viewer: http://localhost:3000 | Jobs: /jobs | API: http://127.0.0.1:8000/docs | Ctrl+C to stop'
wait -n "${viewer_processes[@]}"
