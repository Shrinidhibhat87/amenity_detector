import type { MetadataRoute } from 'next';

const SITE_URL = process.env['NEXT_PUBLIC_SITE_URL'] ?? 'http://localhost:3000';

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: '*',
        allow: [
          '/',
          '/browse',
          '/properties/',
          '/search',
          // /llms.txt and /api/feed.jsonl are the two surfaces AI agents
          // depend on. Listing them explicitly stops a future Disallow on
          // /api/ from accidentally cutting agents off.
          '/llms.txt',
          '/api/feed.jsonl',
        ],
        disallow: [
          // Dev-only component showcase
          '/kitchen-sink',
          // Owner-only upload + amenity-review wizard
          '/detect/',
          // Internal NL query parser; not useful to indexers
          '/api/parse',
        ],
      },
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
  };
}
