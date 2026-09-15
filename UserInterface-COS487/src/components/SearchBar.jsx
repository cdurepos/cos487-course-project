import { forwardRef, useMemo } from 'react';
import { ClearIcon, FilterIcon, SearchIcon } from './Icons';
import { SUGGESTIONS } from '../api/mockIndex';

/**
 * The clear button sits outside the field, to the left, on purpose: far enough
 * from the typing hand that nobody wipes a long query by accident.
 *
 * The grey tail after the cursor is a real inline completion — Tab or the right
 * arrow accepts it.
 */
const SearchBar = forwardRef(function SearchBar(
  {
    value,
    onChange,
    onSubmit,
    onClear,
    onToggleFilters,
    filtersOpen,
    filtersControls,
    activeFilterCount = 0,
    history = [],
    placeholder = 'Search the collection',
  },
  ref
) {
  const completion = useMemo(() => suggest(value, history), [value, history]);

  function handleKeyDown(event) {
    if (event.key === 'Enter') {
      event.preventDefault();
      onSubmit(completion && event.shiftKey ? value + completion : value);
      return;
    }
    if (!completion) return;

    /* Shift+Tab must still move focus backwards, completion or not. */
    const atEnd = event.target.selectionStart === value.length;
    const tab = event.key === 'Tab' && !event.shiftKey;
    if (tab || (event.key === 'ArrowRight' && atEnd)) {
      event.preventDefault();
      onChange(value + completion);
    }
  }

  return (
    <div className="searchbar" role="search">
      <button
        type="button"
        className="searchbar__clear"
        onClick={onClear}
        disabled={!value}
        aria-label="Clear the search box"
      >
        <ClearIcon />
      </button>

      <div className="field">
        <div className="field__ghost" aria-hidden="true">
          <span className="field__ghost-typed">{value}</span>
          <span className="field__ghost-tail">{completion}</span>
        </div>

        <input
          ref={ref}
          className="field__input"
          type="search"
          enterKeyHint="search"
          value={value}
          placeholder={value ? '' : placeholder}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          autoComplete="off"
          spellCheck="false"
          aria-label="Search query"
          aria-describedby="search-suggestion"
        />
        {/* The grey tail is aria-hidden; this is its screen-reader equivalent. */}
        <span id="search-suggestion" className="sr-only" aria-live="polite">
          {completion ? `Suggestion: ${value}${completion}. Press Tab to accept.` : ''}
        </span>

        <div className="field__actions">
          <button
            type="button"
            className={`iconbtn ${filtersOpen ? 'is-on' : ''}`}
            onClick={onToggleFilters}
            aria-expanded={filtersOpen}
            aria-controls={filtersControls}
            aria-label={
              activeFilterCount
                ? `Filters, ${activeFilterCount} applied`
                : 'Filters'
            }
          >
            <FilterIcon />
            {activeFilterCount > 0 && <span className="iconbtn__count">{activeFilterCount}</span>}
          </button>

          <button
            type="button"
            className="iconbtn"
            onClick={() => onSubmit(value)}
            aria-label="Run the search"
          >
            <SearchIcon />
          </button>
        </div>
      </div>
    </div>
  );
});

/** Longest-prefix match against past queries first, then the canned list. */
function suggest(value, history) {
  const typed = value.toLowerCase();
  if (typed.length < 3) return '';

  const pool = [...history.map((h) => h.toLowerCase()), ...SUGGESTIONS];
  const match = pool.find((candidate) => candidate.startsWith(typed) && candidate !== typed);
  return match ? match.slice(value.length) : '';
}

export default SearchBar;
