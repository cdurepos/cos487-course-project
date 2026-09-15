# Search Party

Search interface for the COS 487 group project. React + Vite, with mock searches
only — no backend, no server, no network calls.

## Run it

Needs Node.js 18+.

```bash
npm install
npm run dev
```

## Where the results come from

`src/api/mockIndex.js` holds a small mock corpus and a term-overlap ranker.
`src/api/searchClient.js` wraps it with simulated latency and is the only file the
UI calls, so a real engine can replace it later without touching any component.
Each result has `id, title, snippet, authors, venue, year, docType, score, url`.

## Where each rubric item lives

| Criterion | Where |
|---|---|
| Informative feedback | Inline query completion (`SearchBar`), live latency readout (`App`), hit counts, end-of-results marker |
| Reduce short-term memory load | `useSearchHistory`, suggestion pool, off-centre landing column |
| Easy reversal | Clear button outside the field; back arrow aborts the in-flight request and keeps the query |
| Error handling | `ErrorDialog` — empty query, timeout, network, server. Zero results is an empty state, not an error |
| User control / shortcuts | Filters apply client-side to results already fetched; `/` `Tab` `Enter` `↑↓` `Esc` |
| Consistency | One token set in `styles.css`; one verb per action across the flow |
| Design for closure | Hit count, end-of-results line, explicit empty states |
| Accessibility (WCAG 2.2 AA) | See below — axe-core reports 0 violations on idle, results and side-panel states |

## Accessibility

- **Screen-reader feedback** — one always-mounted `role="status"` region in `App`
  announces "Searching…", the hit count, "Search cancelled", and filter changes.
  The grey inline completion has a text equivalent (`#search-suggestion` in `SearchBar`).
- **Focus is never dropped** — opening the side panel or history menu moves focus in;
  closing hands it back to the control that opened it. Cancel/back returns to the
  search box. The error dialog is a native `<dialog>` opened with `showModal()`, so
  focus is trapped and the page behind is inert.
- **Hidden means hidden** — the closed rail is `visibility: hidden`, so its controls
  are out of the tab order and the accessibility tree, not just transparent.
- **Keyboard** — `Shift+Tab` always moves backwards (only plain `Tab` accepts a
  suggestion); `↑ ↓` only walk results when focus is on the page or a result.
- **Contrast** — `--muted` text ≥ 4.5:1 on every background; control borders use
  `--line-strong` (≥ 3:1). Small text buttons have a 24px minimum target.
- **Semantics** — `role="search"`, `type="search"`, labelled `aside`, grouped chips,
  "(opens in a new tab)" on result links, and a spoken label on relevance scores.

## Demoing the error dialog

Set `MOCK_FAILURE_RATE` in `src/api/searchClient.js` above 0.
