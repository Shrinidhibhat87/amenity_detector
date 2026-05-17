import type { MetadataRoute } from 'next';
import { listProperties } from '@/lib/api';

// Skip build-time prerender — the API isn't reachable during `docker build`.
// Sitemap regenerates per request; Googlebot crawls are infrequent enough
// that the overhead is negligible.
export const dynamic = 'force-dynamic';

// NEXT_PUBLIC_SITE_URL is set in docker-compose.yml / .env.
// Falls back to localhost for local dev.
const SITE_URL = process.env['NEXT_PUBLIC_SITE_URL'] ?? 'http://localhost:3001';

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const properties = await listProperties({ limit: 100 });

  const listingUrls: MetadataRoute.Sitemap = properties.map((p) => ({
    url: `${SITE_URL}/properties/${p.id}`,
    lastModified: new Date(p.created_at),
    changeFrequency: 'weekly',
    priority: 0.8,
  }));

  return [
    { url: SITE_URL, lastModified: new Date(), changeFrequency: 'daily', priority: 1 },
    {
      url: `${SITE_URL}/browse`,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 0.9,
    },
    ...listingUrls,
  ];
}
