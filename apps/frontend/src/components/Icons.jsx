/** Icons are drawn to match the wireframe glyphs: thin strokes, no fills. */

const base = {
  width: 24,
  height: 24,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.5,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true,
};

export const MenuIcon = (p) => (
  <svg {...base} {...p}>
    <circle cx="5" cy="12" r="1.4" fill="currentColor" stroke="none" />
    <circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none" />
    <circle cx="19" cy="12" r="1.4" fill="currentColor" stroke="none" />
  </svg>
);

export const HistoryIcon = (p) => (
  <svg {...base} {...p}>
    <path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1" />
    <path d="M3.2 4.6v4.2h4.2" />
    <path d="M12 7.6V12l3 1.8" />
  </svg>
);

export const SearchIcon = (p) => (
  <svg {...base} {...p}>
    <circle cx="10.5" cy="10.5" r="6.5" />
    <path d="M15.4 15.4 21 21" />
  </svg>
);

export const FilterIcon = (p) => (
  <svg {...base} {...p}>
    <path d="M3.5 5h17l-6.6 7.6V20l-3.8-2.4v-5z" />
  </svg>
);

export const ClearIcon = (p) => (
  <svg {...base} {...p} strokeWidth={2}>
    <path d="M5.5 5.5 18.5 18.5" />
    <path d="M18.5 5.5 5.5 18.5" />
  </svg>
);

export const BackIcon = (p) => (
  <svg {...base} {...p}>
    <path d="M20 12H4" />
    <path d="M10 6 4 12l6 6" />
  </svg>
);

export const GuideIcon = (p) => (
  <svg {...base} {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 11v5.5" />
    <circle cx="12" cy="7.8" r="0.9" fill="currentColor" stroke="none" />
  </svg>
);
