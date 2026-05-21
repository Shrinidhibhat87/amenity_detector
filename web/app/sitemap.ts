import type { MetadataRoute } from 'next';
import { listProperties } from '@/lib/api';
import { buildSitemap } from '@/lib/sitemap-builder';

// Skip build-time prerender — the API isn't reachable during `docker build`.
// Sitemap regenerates per request; Googlebot crawls are infrequent enough
// that the overhead is negligible.
export const dynamic = 'force-dynamic';

// NEXT_PUBLIC_SITE_URL is set in docker-compose.yml / .env.
// Falls back to localhost for local dev.
const SITE_URL = process.env['NEXT_PUBLIC_SITE_URL'] ?? 'http://localhost:3000';

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const properties = await listProperties({ limit: 100 });
  return buildSitemap(properties, SITE_URL, new Date());
}
