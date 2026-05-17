/**
 * /llms.txt — machine-readable site description for LLM agents and AI crawlers.
 *
 * Follows the emerging llms.txt spec (https://llmstxt.org) so agents
 * understand the purpose and structure of this site without scraping HTML.
 */

import { NextResponse } from 'next/server';
import { listProperties } from '@/lib/api';

const SITE_URL = process.env['NEXT_PUBLIC_SITE_URL'] ?? 'http://localhost:3001';

export async function GET(): Promise<NextResponse> {
  const properties = await listProperties({ limit: 100 });

  const listingLines = properties.map((p) => {
    const parts: string[] = [p.name];
    const loc = [p.locality, p.country_code].filter(Boolean).join(', ');
    if (loc.length > 0) parts.push(`(${loc})`);
    return `- ${parts.join(' ')}: ${SITE_URL}/properties/${p.id}`;
  });

  const text = [
    '# Amenity Detector',
    '',
    '> Detect amenities in property photos using vision-language models.',
    '> Owners upload photos; AI detects amenities room by room. Buyers search by natural language.',
    '',
    '## Key pages',
    '',
    `- Home: ${SITE_URL}`,
    `- Browse all listings: ${SITE_URL}/browse`,
    `- Structured JSONL feed: ${SITE_URL}/api/feed.jsonl`,
    `- Sitemap: ${SITE_URL}/sitemap.xml`,
    '',
    '## Listings',
    '',
    ...listingLines,
  ].join('\n');

  return new NextResponse(text, {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
}
