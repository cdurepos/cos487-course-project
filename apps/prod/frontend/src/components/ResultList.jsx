import { useEffect, useRef } from 'react';

export default function ResultList({ results, query, totalBeforeFilters, onResetFilters }) {
  const listRef = useRef(null);
  const linkRefs = useRef([]);

  /* Arrow keys walk the list — only from the page body or from a result, so
     they never hijack arrows inside the side panel, menus or form controls. */
  useEffect(() => {
    function onKey(event) {
      if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const active = document.activeElement;
      const onBody = !active || active === document.body;
      if (!onBody && !listRef.current?.contains(active)) return;

      const links = linkRefs.current.filter(Boolean);
      if (!links.length) return;

      event.preventDefault();
      const current = links.indexOf(active);
      const step = event.key === 'ArrowDown' ? 1 : -1;
      const next = current === -1 ? 0 : (current + step + links.length) % links.length;
      links[next].focus();
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [results]);

  if (results.length === 0) {
    const filtered = totalBeforeFilters > 0;
    return (
      <div className="empty">
        <h2 className="empty__head">
          {filtered
            ? 'Your filters hid every result.'
            : `Nothing in the collection matches “${query}”.`}
        </h2>
        <p className="empty__body">
          {filtered
            ? `${totalBeforeFilters} documents matched before filtering.`
            : 'Try fewer words, or a broader term.'}
        </p>
        {filtered && (
          <button type="button" className="textbtn" onClick={onResetFilters}>
            Reset filters
          </button>
        )}
      </div>
    );
  }

  return (
    <>
      <ol className="results" ref={listRef} aria-label={`Results for “${query}”`}>
        {results.map((result, i) => (
          <li key={result.id}>
            <article className="card">
              <h2 className="card__title">
                <a
                  ref={(el) => (linkRefs.current[i] = el)}
                  href={result.url}
                  target="_blank"
                  rel="noreferrer"
                >
                  {result.title}
                  <span className="sr-only"> (opens in a new tab)</span>
                </a>
              </h2>
              <p className="card__snippet">{result.snippet}</p>
              <p className="card__meta">
                <span>
                  {result.authors}
                  {result.venue ? `, ${result.venue}` : ''} {result.year ?? ''}
                </span>
                <span className="card__score">
                  <span className="sr-only">Relevance score </span>
                  {result.score.toFixed(2)}
                </span>
              </p>
            </article>
          </li>
        ))}
      </ol>
      <p className="results__end">End of results for “{query}”.</p>
    </>
  );
}
