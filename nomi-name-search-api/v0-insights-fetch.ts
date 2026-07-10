/**
 * Drop-in fetch helpers for v0-name-card-system (nomistories.com).
 * Copy to: lib/nomi-api.ts in the Next.js project.
 *
 * API: https://nomi-name-search-api.onrender.com
 * Spec: docs/v0_insights_integration.md
 */

export const API_BASE =
  (typeof process !== 'undefined' &&
    process.env.NEXT_PUBLIC_NOMI_API_BASE) ||
  'https://nomi-name-search-api.onrender.com';

export const INSIGHTS_TIMEOUT_MS = 8_000;

export interface NameCardData {
  name: string;
  name_strip: string;
  language: string;
  meaning: string;
  phonetic_spelling?: string;
  audio_url?: string;
  pronunciation_by?: string;
  cultural_context?: string;
  themes?: string[];
  story?: {
    preview_text?: string;
    attribution?: string;
    title?: string;
    [key: string]: unknown;
  };
}

export interface NameLookupResponse {
  name_strip: string;
  results: NameCardData[];
  total: number;
}

export interface InsightsResponse {
  name: string;
  language: string;
  meaning: string;
  insight: string;
  rag_used: boolean;
  rag_excerpts?: string;
  rag_language_key?: string | null;
  attributions?: string[];
  model?: string | null;
}

async function fetchWithTimeout(
  url: string,
  timeoutMs: number,
): Promise<Response | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    });
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export async function fetchName(
  nameStrip: string,
  language?: string,
): Promise<NameLookupResponse | null> {
  const lang = language ? `?language=${encodeURIComponent(language)}` : '';
  const res = await fetchWithTimeout(
    `${API_BASE}/name/${encodeURIComponent(nameStrip)}${lang}`,
    15_000,
  );
  if (!res?.ok) return null;
  return res.json();
}

/** Returns null on timeout, HTTP error, or missing insight — never throws. */
export async function fetchInsights(
  nameStrip: string,
  language: string,
): Promise<InsightsResponse | null> {
  const params = new URLSearchParams({
    name: nameStrip,
    language: language || '',
  });
  const res = await fetchWithTimeout(
    `${API_BASE}/insights?${params}`,
    INSIGHTS_TIMEOUT_MS,
  );
  if (!res?.ok) return null;
  try {
    const data: InsightsResponse = await res.json();
    const insight = (data.insight ?? '').trim();
    if (!insight) return null;
    return {
      ...data,
      insight,
      rag_used: Boolean(data.rag_used),
      rag_excerpts: data.rag_excerpts ?? '',
      attributions: data.attributions ?? [],
    };
  } catch {
    return null;
  }
}

/** Load name card + insights in parallel; retries insights with resolved language. */
export async function loadNameWithInsights(
  nameStrip: string,
  languageHint = '',
): Promise<{
  card: NameCardData | null;
  insights: InsightsResponse | null;
}> {
  const [nameResult, insightsResult] = await Promise.all([
    fetchName(nameStrip, languageHint || undefined),
    fetchInsights(nameStrip, languageHint),
  ]);

  const card = nameResult?.results?.[0] ?? null;
  let insights = insightsResult;

  if (!insights && card?.language) {
    insights = await fetchInsights(nameStrip, card.language);
  }

  return { card, insights };
}
