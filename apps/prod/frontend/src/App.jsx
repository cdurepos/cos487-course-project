import { useCallback, useEffect, useRef, useState } from 'react';
import { search, SearchError } from './api/search';
import { useSearchHistory } from './hooks/useSearchHistory';
import { DEFAULT_SETTINGS } from './components/Settings';
import { BackIcon, HistoryIcon, SettingsIcon } from './components/Icons';
import SearchBar from './components/SearchBar';
import Sidebar from './components/Sidebar';
import HistoryMenu from './components/HistoryMenu';
import LoadingState from './components/LoadingState';
import ResultList from './components/ResultList';
import ErrorDialog from './components/ErrorDialog';
import styles from './App.module.css';

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

  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [railOpen, setRailOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);

  const { history, remember, clear } = useSearchHistory();
  const inputRef = useRef(null);
  const abortRef = useRef(null);
  const railRef = useRef(null);
  const railReturnRef = useRef(null);
  const historyWrapRef = useRef(null);
  const historyButtonRef = useRef(null);
  const settingsSeen = useRef(false);

  const runSearch = useCallback(
    async (raw) => {
      abortRef.current?.abort();
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
        const response = await search({
          query: raw,
          settings,
          signal: controller.signal,
        });
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
            : `${docs(total)} found for “${response.query}”.`
        );
      } catch (err) {
        if (err?.name === 'AbortError') {
          if (abortRef.current !== controller) return;
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
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    [settings, remember, submitted]
  );

  useEffect(() => {
    if (!settingsSeen.current) {
      settingsSeen.current = true;
      return;
    }
    if (phase === 'results' && submitted) {
      runSearch(submitted);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings]);

  useEffect(() => {
    document.title = phase === 'results' && submitted ? `${submitted} – Search Party` : 'Search Party';
  }, [phase, submitted]);

  function goBack() {
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

  function openRail() {
    if (!railOpen) railReturnRef.current = document.activeElement;
    setRailOpen(true);
    requestAnimationFrame(() =>
      railRef.current?.querySelector('[data-panel="settings"]')?.focus()
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

  useEffect(() => {
    function onKey(event) {
      const typing = ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName);

      if (event.key === '/' && !typing && !errorCode) {
        event.preventDefault();
        inputRef.current?.focus();
        return;
      }
      if (event.key === 'Escape') {
        if (errorCode || historyOpen) return;
        if (railOpen) return closeRail();
        if (document.activeElement === inputRef.current && query) setQuery('');
      }
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  });

  const showBack = phase !== 'idle';
  const latency =
    phase === 'loading' ? `${elapsed} ms` : elapsed == null ? '—' : `${elapsed} ms`;

  return (
    <div className={`${styles.shell} ${railOpen ? styles.railOpen : ''}`}>
      <Sidebar
        id={RAIL_ID}
        ref={railRef}
        open={railOpen}
        onClose={closeRail}
        settings={settings}
        onSettingsChange={setSettings}
        onSettingsReset={() => setSettings(DEFAULT_SETTINGS)}
      />

      <div className={styles.main}>
        <header className={styles.topbar}>
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
              className={`iconbtn ${railOpen ? 'is-on' : ''}`}
              onClick={() => (railOpen ? closeRail() : openRail())}
              aria-expanded={railOpen}
              aria-controls={RAIL_ID}
              aria-label="Settings and user guide"
            >
              <SettingsIcon width={28} height={28} />
            </button>
          )}

          <div className={styles.topbarEnd} ref={historyWrapRef}>
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

        <main className={`${styles.stage} ${phase === 'idle' ? styles.stageIdle : ''}`}>
          <h1 className={styles.wordmark}>Search Party</h1>

          <SearchBar
            ref={inputRef}
            value={query}
            onChange={setQuery}
            onSubmit={runSearch}
            onClear={clearInput}
            placeholder={phase === 'results' ? 'Modify search' : 'Search the collection'}
          />

          <p className={styles.latency}>
            Response time: <span className={styles.latencyValue}>{latency}</span>
            {phase === 'results' && (
              <span className={styles.latencyHits}>{docs(results.length)}</span>
            )}
          </p>

          <p className="srOnly" role="status" aria-live="polite" aria-atomic="true">
            {announcement}
          </p>

          {phase === 'loading' && <LoadingState onCancel={goBack} />}

          {phase === 'results' && <ResultList results={results} query={submitted} />}
        </main>
      </div>

      {errorCode && (
        <ErrorDialog
          code={errorCode}
          onDismiss={() => {
            setErrorCode(null);
            requestAnimationFrame(() => inputRef.current?.focus());
          }}
        />
      )}
    </div>
  );
}
