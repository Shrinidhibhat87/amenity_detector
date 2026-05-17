/**
 * Typed fetch helpers for the FastAPI backend.
 *
 * Server Components (RSC) call these directly — fetch runs on the Node.js
 * runtime where `window` is undefined, so getBaseUrl() returns API_BASE_URL
 * (internal Docker DNS or http://localhost:8000 for local dev).
 *
 * Client Components would use the NEXT_PUBLIC_API_BASE_URL (the public host
 * port the browser can reach). In P-B all data fetching is server-side.
 *
 * The `next: { revalidate }` option on fetch is a Next.js extension that
 * enables ISR — the response is cached for N seconds before a background
 * re-fetch is triggered without blocking the next request.
 */

import { z } from 'zod';
import { PropertyDetail, PropertySummary } from './schemas';

// Next.js extends the global fetch with a `next` option for ISR.
type NextFetchInit = RequestInit & {
  next?: { revalidate?: number | false; tags?: string[] };
};

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export function getBaseUrl(): string {
  if (typeof window === 'undefined') {
    // Server-side (RSC, sitemap, route handlers): use internal Docker DNS.
    return process.env['API_BASE_URL'] ?? 'http://localhost:8000';
  }
  // Client-side: browser cannot resolve the Docker service name.
  return process.env['NEXT_PUBLIC_API_BASE_URL'] ?? 'http://localhost:8000';
}

/** Returns the full URL for serving a stored property image. */
export function getImageUrl(imageId: string): string {
  return `${getBaseUrl()}/api/v1/images/${imageId}`;
}

async function apiFetch<T>(
  schema: { parse: (data: unknown) => T },
  path: string,
  init?: NextFetchInit,
): Promise<T> {
  const url = `${getBaseUrl()}${path}`;
  const res = await fetch(url, init);

  if (!res.ok) {
    throw new ApiError(res.status, `${res.status} ${res.statusText} — ${url}`);
  }

  const json: unknown = await res.json();
  return schema.parse(json);
}

export async function listProperties(opts?: {
  offset?: number;
  limit?: number;
}): Promise<PropertySummary[]> {
  const params = new URLSearchParams();
  if (opts?.offset != null) params.set('offset', String(opts.offset));
  if (opts?.limit != null) params.set('limit', String(opts.limit));
  const qs = params.size > 0 ? `?${params.toString()}` : '';
  return apiFetch(z.array(PropertySummary), `/api/v1/properties/${qs}`, {
    next: { revalidate: 60 },
  });
}

export async function getProperty(id: string): Promise<PropertyDetail> {
  return apiFetch(PropertyDetail, `/api/v1/properties/${id}`, {
    next: { revalidate: 60, tags: [`property-${id}`] },
  });
}
