import { useEffect, useRef } from 'react';

/**
 * `boundaryRef` wraps both this menu and the button that toggles it, so a click
 * on the toggle is not mistaken for a click outside (which closed the menu on
 * mousedown and then reopened it on click).
 */
export default function HistoryMenu({ id, history, onPick, onClear, onClose, boundaryRef }) {
  const ref = useRef(null);

  /* Move focus in on open so keyboard and screen-reader users land in the menu. */
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
    /* The clear button is about to unmount; keep focus inside the menu. */
    ref.current?.focus();
  }

  return (
    <div
      id={id}
      className="history"
      ref={ref}
      role="dialog"
      aria-labelledby={`${id}-title`}
      tabIndex={-1}
    >
      <h2 className="history__title" id={`${id}-title`}>
        Recent searches
      </h2>

      {history.length === 0 ? (
        <p className="history__empty">Searches you run will show up here.</p>
      ) : (
        <ul className="history__list">
          {history.map((query) => (
            <li key={query}>
              <button type="button" className="history__item" onClick={() => onPick(query)}>
                {query}
              </button>
            </li>
          ))}
        </ul>
      )}

      {history.length > 0 && (
        <button type="button" className="history__clear" onClick={clearAll}>
          Clear history
        </button>
      )}
    </div>
  );
}
