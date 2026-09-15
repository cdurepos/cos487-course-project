import { useCallback, useEffect, useState } from 'react';

const KEY = 'search-party.history';
const LIMIT = 8;

function read() {
  try {
    const raw = window.localStorage.getItem(KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.slice(0, LIMIT) : [];
  } catch {
    return [];
  }
}

/** Recent queries, newest first. Storage failures degrade to session-only memory. */
export function useSearchHistory() {
  const [history, setHistory] = useState(read);

  useEffect(() => {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(history));
    } catch {
      /* private mode or full quota — keep the in-memory copy */
    }
  }, [history]);

  const remember = useCallback((query) => {
    const entry = query.trim();
    if (!entry) return;
    setHistory((prev) => [entry, ...prev.filter((q) => q !== entry)].slice(0, LIMIT));
  }, []);

  const clear = useCallback(() => setHistory([]), []);

  return { history, remember, clear };
}
