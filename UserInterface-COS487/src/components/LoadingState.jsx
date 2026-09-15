/**
 * The spinner says "something is happening". How long it has been happening is
 * reported by the one latency readout under the search bar, not duplicated here.
 * Screen readers hear "Searching…" from the app-wide status region instead.
 */
export default function LoadingState({ onCancel }) {
  return (
    <div className="loading">
      <div className="spinner" aria-hidden="true">
        {Array.from({ length: 12 }, (_, i) => (
          <span key={i} style={{ '--i': i }} />
        ))}
      </div>
      <button type="button" className="textbtn" onClick={onCancel}>
        Stop this search
      </button>
    </div>
  );
}
