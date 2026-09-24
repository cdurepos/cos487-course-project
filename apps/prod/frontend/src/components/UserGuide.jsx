import styles from './UserGuide.module.css';

const SHORTCUTS = [
  ['/', 'Jump to the search box'],
  ['Enter', 'Run the search'],
  ['↑ ↓', 'Move through results'],
  ['Esc', 'Clear the box, or close what is open'],
];

export default function UserGuide() {
  return (
    <div className={styles.root}>
      <p className={styles.lead}>
        Type a few words describing what you are looking for, then press Enter.
      </p>

      <dl className={styles.keys}>
        {SHORTCUTS.map(([key, meaning]) => (
          <div className={styles.row} key={key}>
            <dt>
              <kbd className={styles.kbd}>{key}</kbd>
            </dt>
            <dd>{meaning}</dd>
          </div>
        ))}
      </dl>

      <p className={styles.note}>
        Everyday search uses BM25 over papers with stemming. Use Settings to switch
        method, index level, stemming, or result count. The clock icon holds recent
        searches; the back arrow cancels a search in progress.
      </p>
    </div>
  );
}
