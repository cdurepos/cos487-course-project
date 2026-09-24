import { forwardRef } from 'react';
import { ClearIcon, SearchIcon } from './Icons';
import styles from './SearchBar.module.css';

/**
 * The clear button sits outside the field, to the left, on purpose: far enough
 * from the typing hand that nobody wipes a long query by accident.
 */
const SearchBar = forwardRef(function SearchBar(
  {
    value,
    onChange,
    onSubmit,
    onClear,
    placeholder = 'Search the collection',
  },
  ref
) {
  function handleKeyDown(event) {
    if (event.key === 'Enter') {
      event.preventDefault();
      onSubmit(value);
    }
  }

  return (
    <div className={styles.searchbar} role="search">
      <button
        type="button"
        className={styles.clear}
        onClick={onClear}
        disabled={!value}
        aria-label="Clear the search box"
      >
        <ClearIcon />
      </button>

      <div className={styles.field}>
        <input
          ref={ref}
          className={styles.input}
          type="search"
          enterKeyHint="search"
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          autoComplete="off"
          spellCheck="false"
          aria-label="Search query"
        />

        <div className={styles.actions}>
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

export default SearchBar;
