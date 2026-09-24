import styles from './Settings.module.css';

const RETRIEVAL_TYPES = [
  { value: 'bm25', label: 'BM25' },
  { value: 'tfidf', label: 'TF-IDF' },
];

const LEVELS = [
  { value: 'paper', label: 'Paper' },
  { value: 'paragraph', label: 'Paragraph' },
];

/** Defaults for everyday search. */
export const DEFAULT_SETTINGS = {
  type: 'bm25',
  level: 'paper',
  stem: true,
  k: 10,
};

/**
 * Retrieval settings: method, index level, stemming, and how many hits to ask
 * for. Changing these re-runs the search from App.
 */
export default function Settings({ settings, onChange, onReset }) {
  return (
    <div className={styles.root}>
      <div className={styles.group} role="group" aria-labelledby="retrieval-type-label">
        <p className={styles.label} id="retrieval-type-label">
          Retrieval method
        </p>
        <div className={styles.chips}>
          {RETRIEVAL_TYPES.map((option) => (
            <button
              key={option.value}
              type="button"
              className={`${styles.chip} ${settings.type === option.value ? styles.chipOn : ''}`}
              aria-pressed={settings.type === option.value}
              onClick={() => onChange({ ...settings, type: option.value })}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.group} role="group" aria-labelledby="index-level-label">
        <p className={styles.label} id="index-level-label">
          Index level
        </p>
        <div className={styles.chips}>
          {LEVELS.map((option) => (
            <button
              key={option.value}
              type="button"
              className={`${styles.chip} ${settings.level === option.value ? styles.chipOn : ''}`}
              aria-pressed={settings.level === option.value}
              onClick={() => onChange({ ...settings, level: option.value })}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.group}>
        <label className={styles.label} htmlFor="use-stemming">
          Stemming
        </label>
        <select
          id="use-stemming"
          className={styles.select}
          value={settings.stem ? 'yes' : 'no'}
          onChange={(e) => onChange({ ...settings, stem: e.target.value === 'yes' })}
        >
          <option value="yes">On</option>
          <option value="no">Off</option>
        </select>
      </div>

      <div className={styles.group}>
        <label className={styles.label} htmlFor="result-count">
          Results to return
        </label>
        <input
          id="result-count"
          className={styles.field}
          type="number"
          inputMode="numeric"
          min="1"
          max="100"
          value={settings.k}
          onChange={(e) => {
            const next = Number(e.target.value);
            onChange({
              ...settings,
              k: Number.isFinite(next) ? next : DEFAULT_SETTINGS.k,
            });
          }}
        />
      </div>

      <div className={styles.foot}>
        <button type="button" className="textbtn" onClick={onReset}>
          Reset to defaults
        </button>
      </div>
    </div>
  );
}
