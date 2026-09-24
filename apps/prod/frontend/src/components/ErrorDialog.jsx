import { useEffect, useRef } from 'react';
import styles from './ErrorDialog.module.css';

const COPY = {
  EMPTY_QUERY: {
    title: 'The search box is empty',
    body: 'Type a few words describing what you are looking for, then press Enter.',
    action: 'Back to the search box',
  },
  TIMEOUT: {
    title: 'The search timed out',
    body: 'The index took longer than ten seconds to answer. Your query is still in the box — run it again, or narrow it first.',
    action: 'Back to the search box',
  },
  NETWORK: {
    title: 'The search service is unreachable',
    body: 'Nothing was returned by the server. Check that it is running, then try again.',
    action: 'Back to the search box',
  },
  SERVER: {
    title: 'The search failed',
    body: 'The index returned an error for that query. Your query is still in the box — try again, or rephrase it.',
    action: 'Back to the search box',
  },
};

/**
 * Native <dialog> opened with showModal(): the browser traps focus inside it,
 * makes the page behind inert, and closes it on Esc.
 */
export default function ErrorDialog({ code, onDismiss }) {
  const dialogRef = useRef(null);
  const copy = COPY[code] ?? COPY.SERVER;

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog.open) dialog.showModal();
    return () => {
      if (dialog.open) dialog.close();
    };
  }, []);

  const close = () => dialogRef.current?.close();

  return (
    <dialog
      ref={dialogRef}
      className={styles.root}
      role="alertdialog"
      aria-labelledby="dialog-title"
      aria-describedby="dialog-body"
      onClose={onDismiss}
      onKeyDown={(e) => {
        if (e.key === 'Escape') {
          e.preventDefault();
          close();
        }
      }}
      onMouseDown={(e) => e.target === e.currentTarget && close()}
    >
      <div className={styles.inner}>
        <h2 className={styles.title} id="dialog-title">
          {copy.title}
        </h2>
        <p className={styles.body} id="dialog-body">
          {copy.body}
        </p>
        <button type="button" className={styles.action} onClick={close} autoFocus>
          {copy.action}
        </button>
      </div>
    </dialog>
  );
}
