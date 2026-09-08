# X-RAG frontend

React + TypeScript + Vite chat UI for a legal-RAG research platform. This is
the **product UI** — talks to a FastAPI backend over SSE. There is also a
Streamlit app at `../ui/` (`ui/app.py`); that is a separate, older analytics
tool, not this product. Don't conflate the two or "fix" one by copying from
the other.

## Run it

```
npm run dev        # vite dev server, http://localhost:5173
```

Backend must be running separately, from the repo root:

```
python run_api_ui.py
```

(not raw `uvicorn src.api_ui:app` — the entrypoint also loads `.env` and
checks LLM credentials before starting.)

`API_BASE` in [src/lib/api.ts](src/lib/api.ts) is hardcoded to
`http://127.0.0.1:8010` — no env var, change it there if the backend port
changes. Backend CORS in `src/api_ui.py` is hardcoded to ports 5173/5174; add
a port there if Vite picks a different one.

## Architecture

- `App.tsx` — sidebar (sessions, nav) + tab switch between `Chat` and `Graph`.
  Session list is fetched from the backend, not stored client-side except
  `xrag_session_id` in localStorage as a fallback anonymous ID.
- `components/Chat.tsx` — the whole chat experience: composer, message list,
  streaming, the `DetailsPanel` (retrieval chunks / recalled memories / trace
  timings / eval scores tabs). This file is large and does a lot; that's
  deliberate density, not a mess to be split up without being asked.
- `components/Graph.tsx` — knowledge-graph view, separate concern.
- `lib/api.ts` — every backend call. `streamChat()` is the important one: it
  parses Server-Sent Events off a raw `fetch` + `ReadableStream` (no SSE
  library, no axios). Frames are separated by `\n\n`; reads can split
  mid-frame, so it buffers.
- `App.css` — one global stylesheet, plain CSS (no CSS-in-JS, no Tailwind, no
  component library). Class names are hand-written and BEM-ish
  (`.msg-bubble`, `.stage-indicator`, `.details-panel`). Match that style —
  don't introduce a styling system.

## The SSE event contract (backend → frontend)

`POST /ui/chat` streams one event per line (`data: {...}\n\n`), consumed by
`Chat.tsx`'s `applyEvent()`. Event shapes, in the order a normal turn emits
them:

1. `stage` — `{label}` progress text ("Searching memory…", "Retrieving
   documents…", etc.) for the thinking indicator. There is no fixed enum of
   stage names on the frontend — treat `label` as opaque text from
   `src/chat_service.py`, don't hardcode a switch over stage names.
2. `memory` — `{recalled, memory_time}` — recalled past Q&A pairs, arrives
   before generation starts.
3. `meta` — `{search_query, query_was_condensed, arm}` — query
   condensation info.
4. `chunks` — `{chunks, retrieval_time, retrieval_metadata}` — retrieved
   passages. **Arrives before the answer text.** `DetailsPanel` is shown as
   soon as `meta.chunks` or `meta.recalled_memories` exist, not gated on the
   answer being complete — keep it that way, that's what makes the wait feel
   useful instead of dead.
5. `token` — `{text}` — one delta of the streamed answer, repeated many
   times. Appended to the last assistant message's `content`.
6. `strategy` — `{trace_id}`.
7. `evaluation` — `{groundedness, faithfulness, answer_relevancy,
   context_precision, context_relevancy, citation_precision,
   citations_found, device}` — scored after the answer, may arrive late or
   not at all depending on backend config; every field on `DetailsPanel`'s
   eval tab must tolerate `undefined`.
8. `done` — `{total_time, generation_time, retrieval_time, memory_time,
   trace_id}` — terminal event on success.
9. `error` — `{message}` — terminal event on failure. `Chat.tsx` tags the
   assistant message with `failedQuestion` so a Retry button can resend it.

The source of truth for this contract is `src/chat_service.py`'s
`run_chat_turn()` generator (Python) — if a new event type or field is
added/renamed there, the `ChatEvent`/`MsgMeta` types in `lib/api.ts` and
`Chat.tsx` need a matching edit, and TypeScript won't catch a drift by
itself since `ChatEvent` uses `[key: string]: unknown`. **Grep
`chat_service.py` for `yield {` before trusting the TS types are current.**

## Streaming + cancellation

`streamChat()` takes an optional `AbortSignal`. `Chat.tsx` creates one
`AbortController` per `send()`, stores it in `abortRef`, and the composer's
send button swaps to a stop icon while `busy`. Aborting only stops the
*client* from reading further — the FastAPI backend (`src/api_ui.py`'s
`/ui/chat`, a sync `def` route) keeps running the blocking pipeline in
Starlette's threadpool until it finishes on its own; there is no server-side
cancellation. Don't assume abort frees backend/GPU resources.

## Known rough edges (leave as-is unless asked)

- `sessionId()` helper in `Chat.tsx` is currently dead code (unused now that
  `App.tsx` owns session lifecycle via the backend) — tsc flags it, it's
  harmless, not worth "cleaning up" as a drive-by.
- No mobile/narrow-viewport layout pass has been done.
- No test setup exists for this package (no Vitest/RTL configured) — don't
  add a testing framework speculatively; ask first if tests are wanted.
