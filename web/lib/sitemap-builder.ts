/**
 * Sitemap builder for the public property index.
 *
 * Extracted from app/sitemap.ts so the rules are unit-testable without
 * standing up a real HTTP backend.
 *
 * Rules:
 * - Static pages (home, browse, search) appear first with priority weighted
 *   by importance: home=1.0, browse=0.9, search=0.7. lastmod=now because
 *   their content is regenerated per request.
 * - Listings without any images are treated as drafts and excluded. A URL
 *   in the sitemap is a promise to Google that there is indexable content
 *   at that URL; an image-less listing usually means the owner abandoned
 *   the upload step.
 * - Listings with a description, listing_type, and at least one image are
 *   weighted higher (priority 0.8) than sparse listings (priority 0.6).
 * - lastModified uses created_at since the schema has no updated_at column.
 *   That is a known limitation — when an owner edits a listing, the
 *   sitemap will not reflect it until updated_at is added in a future
 *   schema change.
 */

import type { MetadataRoute } from 'next';
import type { PropertySummary } from './schemas';
import { propertyUrl } from './property-url';

function isReady(p: PropertySummary): boolean {
  return p.image_count > 0;
}

function priorityFor(p: PropertySummary): number {
  const full = p.description != null && p.listing_type != null && p.image_count > 0;
  return full ? 0.8 : 0.6;
}

export function buildSitemap(
  properties: PropertySummary[],
  siteUrl: string,
  now: Date,
): MetadataRoute.Sitemap {
  const staticEntries: MetadataRoute.Sitemap = [
    { url: siteUrl, lastModified: now, changeFrequency: 'daily', priority: 1 },
    {
      url: `${siteUrl}/browse`,
      lastModified: now,
      changeFrequency: 'daily',
      priority: 0.9,
    },
    {
      url: `${siteUrl}/search`,
      lastModified: now,
      changeFrequency: 'weekly',
      priority: 0.7,
    },
  ];

  const listingEntries: MetadataRoute.Sitemap = properties
    .filter(isReady)
    .map((p) => ({
      url: propertyUrl(p, siteUrl),
      lastModified: new Date(p.created_at),
      changeFrequency: 'weekly',
      priority: priorityFor(p),
    }));

  return [...staticEntries, ...listingEntries];
}
