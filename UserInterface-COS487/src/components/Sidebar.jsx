import { forwardRef } from 'react';
import { FilterIcon, GuideIcon, ClearIcon } from './Icons';
import FilterPanel from './FilterPanel';
import UserGuide from './UserGuide';

/**
 * One rail, two panels. Opening a panel keeps the other collapsed so the rail
 * never scrolls into a wall of controls.
 *
 * When closed, the rail is `visibility: hidden` (see styles.css), which takes
 * every control out of the tab order and the accessibility tree — not just out
 * of sight.
 */
const Sidebar = forwardRef(function Sidebar(
  {
    id,
    open,
    onClose,
    panel,
    onPanelChange,
    filters,
    onFiltersChange,
    onFiltersReset,
    resultCount,
    hasResults,
  },
  ref
) {
  return (
    <aside
      id={id}
      ref={ref}
      className={`rail ${open ? 'is-open' : ''}`}
      aria-label="Search options"
    >
      <div className="rail__head">
        <button
          type="button"
          className="iconbtn iconbtn--quiet"
          onClick={onClose}
          aria-label="Hide the side panel"
        >
          <ClearIcon width={18} height={18} />
        </button>
      </div>

      <div className="rail__nav">
        <button
          type="button"
          className={`rail__tab ${panel === 'filters' ? 'is-on' : ''}`}
          onClick={() => onPanelChange(panel === 'filters' ? null : 'filters')}
          aria-expanded={panel === 'filters'}
          aria-controls="rail-filters"
          data-panel="filters"
        >
          <FilterIcon width={26} height={26} />
          <span>Filters</span>
        </button>

        <div id="rail-filters" hidden={panel !== 'filters'}>
          {panel === 'filters' && (
            <FilterPanel
              filters={filters}
              onChange={onFiltersChange}
              onReset={onFiltersReset}
              resultCount={resultCount}
              hasResults={hasResults}
            />
          )}
        </div>

        <button
          type="button"
          className={`rail__tab ${panel === 'guide' ? 'is-on' : ''}`}
          onClick={() => onPanelChange(panel === 'guide' ? null : 'guide')}
          aria-expanded={panel === 'guide'}
          aria-controls="rail-guide"
          data-panel="guide"
        >
          <GuideIcon width={26} height={26} />
          <span>User guide</span>
        </button>

        <div id="rail-guide" hidden={panel !== 'guide'}>
          {panel === 'guide' && <UserGuide />}
        </div>
      </div>
    </aside>
  );
});

export default Sidebar;
