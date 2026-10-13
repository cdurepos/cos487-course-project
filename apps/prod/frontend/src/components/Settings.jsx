import { useEffect, useState } from 'react';
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

// The API accepts 1-100 results per search.
const MIN_K = 1;
const MAX_K = 100;
const clampK = (n) => Math.min(MAX_K, Math.max(MIN_K, n));

/**
 * Retrieval settings: method, index level, stemming, and how many hits to ask
 * for. Changing these re-runs the search from App.
 */
export default function Settings({ settings, onChange, onReset }) {
  // The field can sit empty or half-typed without sending a bad value to the
  // API; settings.k only ever holds a number from 1 to 100.
  const [kDraft, setKDraft] = useState(String(settings.k));
  useEffect(() => setKDraft(String(settings.k)), [settings.k]);

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
          min={MIN_K}
          max={MAX_K}
          value={kDraft}
          onChange={(e) => {
            setKDraft(e.target.value);
            const next = Number.parseInt(e.target.value, 10);
            if (Number.isFinite(next)) {
              onChange({ ...settings, k: clampK(next) });
            }
          }}
          onBlur={() => setKDraft(String(settings.k))}
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
