import type { MetadataRoute } from 'next';

const SITE_URL = process.env['NEXT_PUBLIC_SITE_URL'] ?? 'http://localhost:3001';

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: '*',
        allow: ['/', '/browse', '/properties/'],
        // Kitchen-sink is a dev-only component showcase; detect wizard is owner-only.
        disallow: ['/kitchen-sink', '/detect/'],
      },
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
  };
}
