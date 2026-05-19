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
import {
  ImageDetectionResponse,
  ModelInfo,
  PropertyCreateResponse,
  PropertyDetail,
  PropertyImage,
  PropertySummary,
  type DescribeRequest,
  type DescribeResponse,
  type ImageUpdateRequest,
  type PropertyCreateRequest,
  type PropertyUpdateRequest,
} from './schemas';
import { DescribeResponse as DescribeResponseSchema } from './schemas';

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

/**
 * Next.js 14.2+/15 refuses to proxy images whose upstream resolves to a
 * private IP (SSRF guard), and `remotePatterns` does not bypass that check.
 * In dev the API lives at http://localhost:8000 / http://api:8000, both of
 * which trip the guard — so we pass `unoptimized` on those images to bypass
 * the optimiser entirely. Production hosts (real domains) still get the
 * optimised path.
 */
export function shouldUnoptimizeApiImages(): boolean {
  const base = getBaseUrl();
  try {
    const { hostname } = new URL(base);
    return (
      hostname === 'localhost' ||
      hostname === '127.0.0.1' ||
      hostname === '::1' ||
      hostname === 'api'
    );
  } catch {
    return false;
  }
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

// ── Mutations ────────────────────────────────────────────────────────────────
// All mutations run client-side (no cache) and bypass ISR.

export async function listModels(): Promise<ModelInfo[]> {
  return apiFetch(z.array(ModelInfo), '/api/v1/models/', { cache: 'no-store' });
}

export async function createProperty(
  body: PropertyCreateRequest,
): Promise<PropertyCreateResponse> {
  return apiFetch(PropertyCreateResponse, '/api/v1/properties/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    cache: 'no-store',
  });
}

export interface UploadImageArgs {
  propertyId: string;
  file: File;
  modelName: string;
  /** Abort signal — used to enforce a per-image timeout. */
  signal?: AbortSignal;
}

export async function uploadImage(args: UploadImageArgs): Promise<ImageDetectionResponse> {
  const fd = new FormData();
  fd.append('file', args.file);
  fd.append('model_name', args.modelName);
  // FormData sets its own Content-Type with the multipart boundary —
  // do NOT set Content-Type manually, the browser must control it.
  return apiFetch(
    ImageDetectionResponse,
    `/api/v1/properties/${args.propertyId}/images`,
    args.signal != null
      ? { method: 'POST', body: fd, cache: 'no-store', signal: args.signal }
      : { method: 'POST', body: fd, cache: 'no-store' },
  );
}

export async function patchImage(
  propertyId: string,
  imageId: string,
  patch: ImageUpdateRequest,
): Promise<PropertyImage> {
  return apiFetch(PropertyImage, `/api/v1/properties/${propertyId}/images/${imageId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
    cache: 'no-store',
  });
}

export async function describeProperty(
  propertyId: string,
  body: DescribeRequest,
): Promise<DescribeResponse> {
  return apiFetch(DescribeResponseSchema, `/api/v1/properties/${propertyId}/describe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    cache: 'no-store',
  });
}

export async function patchProperty(
  propertyId: string,
  patch: PropertyUpdateRequest,
): Promise<PropertyDetail> {
  return apiFetch(PropertyDetail, `/api/v1/properties/${propertyId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
    cache: 'no-store',
  });
}
