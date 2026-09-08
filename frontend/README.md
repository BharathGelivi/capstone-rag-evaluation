# X-RAG frontend

## Run it

Two servers, both needed. Requires the Python env set up (see the repo
root's [README](../README.md#quick-start)) and Node.js/npm installed.

```bash
# Terminal 1 -- backend (SSE chat + graph API), from the repo root
python run_api_ui.py            # http://127.0.0.1:8010

# Terminal 2 -- this frontend
npm install                     # first run only
npm run dev                     # http://localhost:5173
```

Open <http://localhost:5173>. **Chat** tab: pick a retrieval arm and ask a
question against the statutes or judgments corpus. **Graph** tab: toggle
between the RAG provenance graph and the claude-mem memory graph.

React + TypeScript + Vite chat UI for the X-RAG diagnostic platform. Talks to
a FastAPI backend over Server-Sent Events. See [CLAUDE.md](CLAUDE.md) for
architecture notes if you're changing this code, not just running it.

If the frontend can't reach the backend, check that `API_BASE` in
[src/lib/api.ts](src/lib/api.ts) still points at `http://127.0.0.1:8010` and
that the backend's CORS allowlist in `../src/api_ui.py` includes whatever
port Vite actually picked (it prints this on startup; 5173 is the default
but Vite falls back to 5174 etc. if that port is busy).

## Build

```bash
npm run build      # tsc -b && vite build, output in dist/
npm run preview    # serve the production build locally
```
