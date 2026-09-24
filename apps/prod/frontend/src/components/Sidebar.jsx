import { forwardRef, useState } from 'react';
import { GuideIcon, ClearIcon, ChevronDown } from './Icons';
import Settings from './Settings';
import UserGuide from './UserGuide';
import styles from './Sidebar.module.css';

/**
 * Settings are always shown when the rail is open. The user guide is the only
 * expandable section.
 *
 * When closed, the rail is `visibility: hidden`, which takes every control out
 * of the tab order and the accessibility tree — not just out of sight.
 */
const Sidebar = forwardRef(function Sidebar(
  {
    id,
    open,
    onClose,
    settings,
    onSettingsChange,
    onSettingsReset,
  },
  ref
) {
  const [guideOpen, setGuideOpen] = useState(false);

  return (
    <aside
      id={id}
      ref={ref}
      className={`${styles.rail} ${open ? styles.open : ''}`}
      aria-label="Settings"
    >
      <div className={styles.head}>
        <button
          type="button"
          className="iconbtn iconbtn--quiet"
          onClick={onClose}
          aria-label="Hide the side panel"
          data-panel="settings"
        >
          <ClearIcon width={18} height={18} />
        </button>
      </div>

      <div className={styles.body}>
        <h2 className={styles.heading}>Settings</h2>
        <Settings
          settings={settings}
          onChange={onSettingsChange}
          onReset={onSettingsReset}
        />

        <button
          type="button"
          className={`${styles.tab} ${guideOpen ? styles.tabOn : ''}`}
          onClick={() => setGuideOpen((v) => !v)}
          aria-expanded={guideOpen}
          aria-controls="rail-guide"
        >
          <GuideIcon width={22} height={22} />
          <span>User guide</span>
          <ChevronDown
            width={18}
            height={18}
            className={`${styles.chevron} ${guideOpen ? styles.chevronOpen : ''}`}
          />
        </button>

        <div
          id="rail-guide"
          className={`${styles.fold} ${guideOpen ? styles.foldOpen : ''}`}
        >
          <div className={styles.foldInner}>
            <UserGuide />
          </div>
        </div>
      </div>
    </aside>
  );
});

export default Sidebar;
