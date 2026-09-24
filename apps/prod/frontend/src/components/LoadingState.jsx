import styles from './LoadingState.module.css';

/**
 * The spinner says "something is happening". How long it has been happening is
 * reported by the one latency readout under the search bar, not duplicated here.
 */
export default function LoadingState({ onCancel }) {
  return (
    <div className={styles.root}>
      <div className={styles.spinner} aria-hidden="true">
        {Array.from({ length: 12 }, (_, i) => (
          <span key={i} className={styles.spoke} style={{ '--i': i }} />
        ))}
      </div>
      <button type="button" className="textbtn" onClick={onCancel}>
        Stop this search
      </button>
    </div>
  );
}
