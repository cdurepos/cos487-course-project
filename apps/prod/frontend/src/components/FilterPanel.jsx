const DOC_TYPES = [
  { value: 'paper', label: 'Papers' },
  { value: 'dataset', label: 'Datasets' },
  { value: 'code', label: 'Code' },
  { value: 'thesis', label: 'Theses' },
];

const SORTS = [
  { value: 'relevance', label: 'Relevance' },
  { value: 'newest', label: 'Newest first' },
  { value: 'oldest', label: 'Oldest first' },
];

export const EMPTY_FILTERS = { docTypes: [], fromYear: '', sort: 'relevance' };

export function countActive(filters) {
  let n = filters.docTypes.length;
  if (filters.fromYear) n += 1;
  if (filters.sort !== 'relevance') n += 1;
  return n;
}

/** Filtering happens on results already in hand, so nothing reloads. */
export function applyFilters(results, filters) {
  let out = results;

  if (filters.docTypes.length) {
    out = out.filter((r) => filters.docTypes.includes(r.docType));
  }
  if (filters.fromYear) {
    out = out.filter((r) => !r.year || r.year >= Number(filters.fromYear));
  }
  if (filters.sort === 'newest') {
    out = [...out].sort((a, b) => (b.year ?? 0) - (a.year ?? 0));
  } else if (filters.sort === 'oldest') {
    out = [...out].sort((a, b) => (a.year ?? 0) - (b.year ?? 0));
  }
  return out;
}

export default function FilterPanel({ filters, onChange, onReset, resultCount, hasResults }) {
  function toggleType(value) {
    const next = filters.docTypes.includes(value)
      ? filters.docTypes.filter((t) => t !== value)
      : [...filters.docTypes, value];
    onChange({ ...filters, docTypes: next });
  }

  return (
    <div className="filters">
      <div className="filters__group" role="group" aria-labelledby="doc-type-label">
        <p className="filters__label" id="doc-type-label">
          Document type
        </p>
        <div className="chips">
          {DOC_TYPES.map((type) => (
            <button
              key={type.value}
              type="button"
              className={`chip ${filters.docTypes.includes(type.value) ? 'is-on' : ''}`}
              aria-pressed={filters.docTypes.includes(type.value)}
              onClick={() => toggleType(type.value)}
            >
              {type.label}
            </button>
          ))}
        </div>
      </div>

      <div className="filters__group">
        <label className="filters__label" htmlFor="from-year">
          Published from
        </label>
        <input
          id="from-year"
          className="filters__year"
          type="number"
          inputMode="numeric"
          min="1990"
          max="2026"
          placeholder="Any year"
          value={filters.fromYear}
          onChange={(e) => onChange({ ...filters, fromYear: e.target.value })}
        />
      </div>

      <div className="filters__group">
        <label className="filters__label" htmlFor="sort-by">
          Order by
        </label>
        <select
          id="sort-by"
          className="filters__select"
          value={filters.sort}
          onChange={(e) => onChange({ ...filters, sort: e.target.value })}
        >
          {SORTS.map((sort) => (
            <option key={sort.value} value={sort.value}>
              {sort.label}
            </option>
          ))}
        </select>
      </div>

      <div className="filters__foot">
        {hasResults && (
          <span className="filters__count">
            {resultCount} shown — filtered here, nothing reloads
          </span>
        )}
        <button type="button" className="textbtn" onClick={onReset}>
          Reset filters
        </button>
      </div>
    </div>
  );
}
