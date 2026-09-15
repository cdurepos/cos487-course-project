const SHORTCUTS = [
  ['/', 'Jump to the search box'],
  ['Tab', 'Accept the suggested ending'],
  ['Enter', 'Run the search'],
  ['↑ ↓', 'Move through results'],
  ['Esc', 'Clear the box, or close what is open'],
];

export default function UserGuide() {
  return (
    <div className="guide">
      <p className="guide__lead">
        Type a few words and press Enter. The grey text after your cursor is a suggested ending —
        press Tab to take it.
      </p>

      <dl className="guide__keys">
        {SHORTCUTS.map(([key, meaning]) => (
          <div className="guide__row" key={key}>
            <dt>
              <kbd>{key}</kbd>
            </dt>
            <dd>{meaning}</dd>
          </div>
        ))}
      </dl>

      <p className="guide__note">
        Filters narrow the results you already have, so changing one does not run the search again.
        The clock icon holds your recent searches; the back arrow cancels a search in progress and
        keeps what you typed.
      </p>
    </div>
  );
}
