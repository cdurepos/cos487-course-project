import { useEffect, useRef } from 'react';
import styles from './ResultList.module.css';

export default function ResultList({ results, query }) {
  const listRef = useRef(null);
  const linkRefs = useRef([]);

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
    return (
      <div className={styles.empty}>
        <h2 className={styles.emptyHead}>Nothing in the collection matches “{query}”.</h2>
        <p className={styles.emptyBody}>Try fewer words, or a broader term.</p>
      </div>
    );
  }

  return (
    <>
      <ol className={styles.list} ref={listRef} aria-label={`Results for “${query}”`}>
        {results.map((result, i) => (
          <li key={result.id}>
            <article className={styles.card}>
              <h2 className={styles.title}>
                {result.url && result.url !== '#' ? (
                  <a
                    ref={(el) => (linkRefs.current[i] = el)}
                    href={result.url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {result.title}
                    <span className="srOnly"> (opens in a new tab)</span>
                  </a>
                ) : (
                  <span ref={(el) => (linkRefs.current[i] = el)} tabIndex={0}>
                    {result.title}
                  </span>
                )}
              </h2>
              {result.snippet ? <p className={styles.snippet}>{result.snippet}</p> : null}
              <p className={styles.meta}>
                <span>{result.id}</span>
                <span className={styles.score}>
                  <span className="srOnly">Relevance score </span>
                  {result.score.toFixed(4)}
                </span>
              </p>
            </article>
          </li>
        ))}
      </ol>
      <p className={styles.end}>End of results for “{query}”.</p>
    </>
  );
}
