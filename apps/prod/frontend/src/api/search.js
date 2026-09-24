/**
 * The only file that knows where results come from.
 *
 * Contract the UI relies on:
 *   search({ query, settings, signal }) -> { query, results, totalHits, elapsedMs }
 *   throws SearchError on failure
 *   rejects with an AbortError when `signal` fires
 */

export class SearchError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'SearchError';
    this.code = code; // EMPTY_QUERY | TIMEOUT | NETWORK | SERVER
  }
}

const TIMEOUT_MS = 60_000;

export async function search({ query, settings, signal }) {
  const trimmed = query.trim();
  if (!trimmed) {
    throw new SearchError('EMPTY_QUERY', 'Enter a few words to search for.');
  }

  const started = performance.now();
  const params = new URLSearchParams({
    q: trimmed,
    type: settings?.type ?? 'bm25',
    level: settings?.level ?? 'paper',
    stem: String(settings?.stem ?? true),
    k: String(settings?.k ?? 10),
  });

  const payload = await getJson(`/search?${params}`, signal);

  return {
    query: payload.query ?? trimmed,
    results: (payload.results ?? []).map(toResult),
    totalHits: payload.totalHits ?? payload.results?.length ?? 0,
    elapsedMs: Math.round(performance.now() - started),
  };
}

async function getJson(url, signal) {
  const controller = new AbortController();
  const relay = () => controller.abort();
  let timedOut = false;

  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, TIMEOUT_MS);
  signal?.addEventListener('abort', relay, { once: true });

  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new SearchError('SERVER', `The search service answered with ${response.status}.`);
    }
    return await response.json();
  } catch (err) {
    if (err instanceof SearchError) throw err;
    if (timedOut) throw new SearchError('TIMEOUT', 'The search service took too long to answer.');
    if (signal?.aborted) throw err;
    if (err instanceof SyntaxError) {
      throw new SearchError('SERVER', 'The search service did not return JSON.');
    }
    throw new SearchError('NETWORK', 'The search service could not be reached.');
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', relay);
  }
}

/** Map an API hit into the shape ResultList renders. */
function toResult(hit) {
  return {
    id: String(hit.id),
    title: hit.title || String(hit.id),
    snippet: hit.snippet || '',
    authors: hit.authors || '',
    venue: hit.venue || '',
    year: hit.year ?? null,
    docType: hit.docType || '',
    score: Number(hit.score) || 0,
    url: hit.url || '#',
  };
}
