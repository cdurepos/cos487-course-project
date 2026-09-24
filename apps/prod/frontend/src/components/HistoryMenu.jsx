import { useEffect, useRef } from 'react';
import styles from './HistoryMenu.module.css';

/**
 * `boundaryRef` wraps both this menu and the button that toggles it, so a click
 * on the toggle is not mistaken for a click outside.
 */
export default function HistoryMenu({ id, history, onPick, onClear, onClose, boundaryRef }) {
  const ref = useRef(null);

  useEffect(() => {
    const first = ref.current?.querySelector('button');
    (first ?? ref.current)?.focus();
  }, []);

  useEffect(() => {
    function onPointer(event) {
      const boundary = boundaryRef?.current ?? ref.current;
      if (!boundary?.contains(event.target)) onClose({ restoreFocus: false });
    }
    function onKey(event) {
      if (event.key === 'Escape') onClose({ restoreFocus: true });
    }
    document.addEventListener('mousedown', onPointer);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointer);
      document.removeEventListener('keydown', onKey);
    };
  }, [onClose, boundaryRef]);

  function clearAll() {
    onClear();
    ref.current?.focus();
  }

  return (
    <div
      id={id}
      className={styles.root}
      ref={ref}
      role="dialog"
      aria-labelledby={`${id}-title`}
      tabIndex={-1}
    >
      <h2 className={styles.title} id={`${id}-title`}>
        Recent searches
      </h2>

      {history.length === 0 ? (
        <p className={styles.empty}>Searches you run will show up here.</p>
      ) : (
        <ul className={styles.list}>
          {history.map((query) => (
            <li key={query}>
              <button type="button" className={styles.item} onClick={() => onPick(query)}>
                {query}
              </button>
            </li>
          ))}
        </ul>
      )}

      {history.length > 0 && (
        <button type="button" className={styles.clear} onClick={clearAll}>
          Clear history
        </button>
      )}
    </div>
  );
}
