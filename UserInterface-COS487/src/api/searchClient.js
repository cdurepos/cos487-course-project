/**
 * The only file that knows where results come from. Right now that is the
 * mock corpus in mockIndex.js — no server, no network.
 *
 * Contract the UI relies on (keep it if a real engine is wired in later):
 *   search({ query, filters, signal }) -> { query, results, totalHits, elapsedMs }
 *   throws SearchError on failure
 *   rejects with an AbortError when `signal` fires
 */

import { rankDocuments } from './mockIndex';

/** Set above 0 to demo the error dialog, e.g. 0.3 for a failure every few searches. */
const MOCK_FAILURE_RATE = 0;

export class SearchError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'SearchError';
    this.code = code; // EMPTY_QUERY | TIMEOUT | NETWORK | SERVER
  }
}

export async function search({ query, signal }) {
  const trimmed = query.trim();
  if (!trimmed) {
    throw new SearchError('EMPTY_QUERY', 'Enter a few words to search for.');
  }

  const started = performance.now();
  const results = await mockSearch(trimmed, signal);

  return {
    query: trimmed,
    results,
    totalHits: results.length,
    elapsedMs: Math.round(performance.now() - started),
  };
}

/** Simulated latency so the loading state and cancel button can be seen. */
function mockSearch(query, signal) {
  const latency = 700 + Math.random() * 900;

  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(abortError());

    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      if (Math.random() < MOCK_FAILURE_RATE) {
        reject(new SearchError('SERVER', 'The index did not respond to that query.'));
      } else {
        resolve(rankDocuments(query));
      }
    }, latency);

    function onAbort() {
      clearTimeout(timer);
      reject(abortError());
    }
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

function abortError() {
  return new DOMException('Search cancelled', 'AbortError');
}
