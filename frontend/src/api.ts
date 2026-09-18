export interface QueryResponse {
  answer: string;
  sources: string[];
  contexts: string[];
}

export interface QueryResult extends QueryResponse {
  latencyMs: number;
}

export interface DocumentInfo {
  source: string;
  chunks: number;
}

export async function askQuestion(
  question: string,
  fetchImpl: typeof fetch = fetch,
): Promise<QueryResult> {
  const started = performance.now();
  const response = await fetchImpl("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const latencyMs = performance.now() - started;

  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }

  const body = (await response.json()) as QueryResponse;
  return { ...body, latencyMs };
}

export async function listDocuments(fetchImpl: typeof fetch = fetch): Promise<DocumentInfo[]> {
  const response = await fetchImpl("/documents");
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  const body = (await response.json()) as { documents: DocumentInfo[] };
  return body.documents;
}

export async function uploadDocument(
  file: File,
  fetchImpl: typeof fetch = fetch,
): Promise<DocumentInfo> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetchImpl("/documents", { method: "POST", body: form });
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return (await response.json()) as DocumentInfo;
}

export async function deleteDocument(
  source: string,
  fetchImpl: typeof fetch = fetch,
): Promise<void> {
  const response = await fetchImpl(`/documents/${encodeURIComponent(source)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
}

export interface Author {
  given: string;
  family: string;
}

export interface CitationMetadata {
  title: string;
  authors: Author[];
  year: number | null;
  venue: string | null;
  document_type: string;
  url: string | null;
  doi: string | null;
}

export interface Citation {
  source: string;
  metadata: CitationMetadata;
  ieee: string;
  apa: string;
}

export async function getCitation(
  source: string,
  fetchImpl: typeof fetch = fetch,
): Promise<Citation> {
  const response = await fetchImpl(`/documents/${encodeURIComponent(source)}/citation`);
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return (await response.json()) as Citation;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    // non-JSON error body; fall through to the generic message
  }
  return `Request failed (${response.status})`;
}

export function formatLatency(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}
