import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { search, SearchError } from './api/searchClient';
import { useSearchHistory } from './hooks/useSearchHistory';
import { applyFilters, countActive, EMPTY_FILTERS } from './components/FilterPanel';
import { BackIcon, HistoryIcon, MenuIcon } from './components/Icons';
import SearchBar from './components/SearchBar';
import Sidebar from './components/Sidebar';
import HistoryMenu from './components/HistoryMenu';
import LoadingState from './components/LoadingState';
import ResultList from './components/ResultList';
import ErrorDialog from './components/ErrorDialog';

const RAIL_ID = 'search-rail';
const HISTORY_ID = 'history-menu';

const docs = (n) => `${n} document${n === 1 ? '' : 's'}`;

export default function App() {
  const [phase, setPhase] = useState('idle'); // idle | loading | results
  const [query, setQuery] = useState('');
  const [submitted, setSubmitted] = useState('');
  const [results, setResults] = useState([]);
  const [elapsed, setElapsed] = useState(null);
  const [errorCode, setErrorCode] = useState(null);
  const [announcement, setAnnouncement] = useState('');

  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [railOpen, setRailOpen] = useState(false);
  const [railPanel, setRailPanel] = useState(null);
  const [historyOpen, setHistoryOpen] = useState(false);

  const { history, remember, clear } = useSearchHistory();
  const inputRef = useRef(null);
  const abortRef = useRef(null);
  const railRef = useRef(null);
  const railReturnRef = useRef(null);
  const historyWrapRef = useRef(null);
  const historyButtonRef = useRef(null);
  const filtersSeen = useRef(false);

  const visible = useMemo(() => applyFilters(results, filters), [results, filters]);
  const activeFilters = countActive(filters);

  /* --------------------------------------------------------- lifecycle --- */

  const runSearch = useCallback(
    async (raw) => {
      const controller = new AbortController();
      abortRef.current = controller;

      setHistoryOpen(false);
      setErrorCode(null);
      setPhase('loading');
      setElapsed(0);
      setAnnouncement('Searching…');

      const started = performance.now();
      const tick = setInterval(
        () => setElapsed(Math.round(performance.now() - started)),
        50
      );

      try {
        const response = await search({ query: raw, filters, signal: controller.signal });
        const shown = applyFilters(response.results, filters).length;
        const total = response.results.length;
        setResults(response.results);
        setSubmitted(response.query);
        setElapsed(response.elapsedMs);
        remember(response.query);
        setQuery('');
        setPhase('results');
        setAnnouncement(
          total === 0
            ? `No documents match “${response.query}”.`
            : shown === total
              ? `${docs(total)} found for “${response.query}”.`
              : `${shown} of ${docs(total)} shown for “${response.query}” after filtering.`
        );
      } catch (err) {
        if (err?.name === 'AbortError') {
          setPhase('idle');
          setElapsed(null);
          setAnnouncement('Search cancelled.');
          return;
        }
        setErrorCode(err instanceof SearchError ? err.code : 'SERVER');
        setPhase(submitted ? 'results' : 'idle');
        setElapsed(null);
        setAnnouncement('');
      } finally {
        clearInterval(tick);
        abortRef.current = null;
      }
    },
    [filters, remember, submitted]
  );

  /* Filters apply without a reload, so say what changed. */
  useEffect(() => {
    if (!filtersSeen.current) {
      filtersSeen.current = true;
      return;
    }
    if (phase === 'results') {
      setAnnouncement(`${visible.length} of ${docs(results.length)} shown.`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters]);

  useEffect(() => {
    document.title = phase === 'results' && submitted ? `${submitted} – Search Party` : 'Search Party';
  }, [phase, submitted]);

  /** Back arrow: cancels a search in flight, or steps off the results page. */
  function goBack() {
    // The back button unmounts either way; put focus somewhere useful.
    requestAnimationFrame(() => inputRef.current?.focus());
    if (phase === 'loading') {
      abortRef.current?.abort();
      return;
    }
    setPhase('idle');
    setQuery(submitted);
    setElapsed(null);
  }

  function clearInput() {
    setQuery('');
    inputRef.current?.focus();
  }

  function pickFromHistory(entry) {
    setQuery(entry);
    setHistoryOpen(false);
    inputRef.current?.focus();
    runSearch(entry);
  }

  const closeHistory = useCallback(({ restoreFocus } = {}) => {
    setHistoryOpen(false);
    if (restoreFocus) historyButtonRef.current?.focus();
  }, []);

  /** Opening moves focus into the rail; closing hands it back to the opener. */
  function openRail(panel) {
    if (!railOpen) railReturnRef.current = document.activeElement;
    setRailOpen(true);
    setRailPanel(panel);
    requestAnimationFrame(() =>
      railRef.current?.querySelector(`[data-panel="${panel}"]`)?.focus()
    );
  }

  function closeRail() {
    const hadFocus = railRef.current?.contains(document.activeElement);
    setRailOpen(false);
    if (hadFocus) {
      const back = railReturnRef.current;
      (back?.isConnected ? back : inputRef.current)?.focus();
    }
  }

  /* ---------------------------------------------------------- shortcuts --- */

  useEffect(() => {
    function onKey(event) {
      const typing = ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName);

      if (event.key === '/' && !typing && !errorCode) {
        event.preventDefault();
        inputRef.current?.focus();
        return;
      }
      if (event.key === 'Escape') {
        // The dialog and the history menu handle their own Esc.
        if (errorCode || historyOpen) return;
        if (railOpen) return closeRail();
        // Only the search box itself — Esc in the year field must not wipe the query.
        if (document.activeElement === inputRef.current && query) setQuery('');
      }
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  });

  /* ------------------------------------------------------------- render --- */

  const showBack = phase !== 'idle';
  const latency =
    phase === 'loading' ? `${elapsed} ms` : elapsed == null ? '—' : `${elapsed} ms`;

  return (
    <div className={`shell ${railOpen ? 'rail-open' : ''}`}>
      <Sidebar
        id={RAIL_ID}
        ref={railRef}
        open={railOpen}
        onClose={closeRail}
        panel={railPanel}
        onPanelChange={setRailPanel}
        filters={filters}
        onFiltersChange={setFilters}
        onFiltersReset={() => setFilters(EMPTY_FILTERS)}
        resultCount={visible.length}
        hasResults={phase === 'results'}
      />

      <div className="main">
        <header className="topbar">
          {showBack ? (
            <button
              type="button"
              className="iconbtn"
              onClick={goBack}
              aria-label={phase === 'loading' ? 'Cancel this search' : 'Back to the search box'}
            >
              <BackIcon width={28} height={28} />
            </button>
          ) : (
            <button
              type="button"
              className="iconbtn"
              onClick={() => (railOpen ? closeRail() : openRail('filters'))}
              aria-expanded={railOpen}
              aria-controls={RAIL_ID}
              aria-label="Filters and user guide"
            >
              <MenuIcon width={28} height={28} />
            </button>
          )}

          <div className="topbar__end" ref={historyWrapRef}>
            <button
              ref={historyButtonRef}
              type="button"
              className={`iconbtn ${historyOpen ? 'is-on' : ''}`}
              onClick={() => setHistoryOpen((v) => !v)}
              aria-expanded={historyOpen}
              aria-haspopup="dialog"
              aria-controls={historyOpen ? HISTORY_ID : undefined}
              aria-label="Recent searches"
            >
              <HistoryIcon width={28} height={28} />
            </button>
            {historyOpen && (
              <HistoryMenu
                id={HISTORY_ID}
                history={history}
                onPick={pickFromHistory}
                onClear={clear}
                onClose={closeHistory}
                boundaryRef={historyWrapRef}
              />
            )}
          </div>
        </header>

        <main className={`stage ${phase === 'idle' ? 'stage--idle' : ''}`}>
          <h1 className="wordmark">Search Party</h1>

          <SearchBar
            ref={inputRef}
            value={query}
            onChange={setQuery}
            onSubmit={runSearch}
            onClear={clearInput}
            onToggleFilters={() =>
              railOpen && railPanel === 'filters' ? closeRail() : openRail('filters')
            }
            filtersOpen={railOpen && railPanel === 'filters'}
            filtersControls={RAIL_ID}
            activeFilterCount={activeFilters}
            history={history}
            placeholder={phase === 'results' ? 'Modify search' : 'Search the collection'}
          />

          <p className="latency">
            Response time: <span className="latency__value">{latency}</span>
            {phase === 'results' && (
              <span className="latency__hits">
                {visible.length} of {docs(results.length)}
                {activeFilters > 0 ? ' after filtering' : ''}
              </span>
            )}
          </p>

          {/* One polite live region, always mounted, so every state change is heard. */}
          <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
            {announcement}
          </p>

          {phase === 'loading' && <LoadingState onCancel={goBack} />}

          {phase === 'results' && (
            <ResultList
              results={visible}
              query={submitted}
              totalBeforeFilters={results.length}
              onResetFilters={() => setFilters(EMPTY_FILTERS)}
            />
          )}
        </main>
      </div>

      {errorCode && (
        <ErrorDialog
          code={errorCode}
          onDismiss={() => {
            setErrorCode(null);
            // Wait for the modal to close — the page behind it is inert until then.
            requestAnimationFrame(() => inputRef.current?.focus());
          }}
        />
      )}
    </div>
  );
}
