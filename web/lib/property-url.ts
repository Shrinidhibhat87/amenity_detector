/**
 * Canonical URL helpers for property listings.
 *
 * A property has both a database UUID and an immutable URL-safe slug.
 * Public-facing surfaces — sitemap, JSON-LD, canonical link, og:url,
 * BreadcrumbList items, Browse cards — should always emit the slug
 * URL when one exists, falling back to the UUID URL only for legacy
 * pre-Phase 9 rows that have slug = NULL.
 *
 * Centralising the choice here keeps the "which URL is canonical?"
 * decision in one place, so a future change (e.g. moving to a different
 * path prefix) lives in one file.
 */

import type { PropertyDetail, PropertySummary } from './schemas';

type Sluggable = Pick<PropertyDetail | PropertySummary, 'id' | 'slug'>;

/** Returns the canonical site-relative path for a property. */
export function propertyPath(p: Sluggable): string {
  const slug = p.slug;
  if (slug != null && slug.length > 0) {
    return `/properties/${slug}`;
  }
  return `/properties/${p.id}`;
}

/** Returns the canonical absolute URL for a property. */
export function propertyUrl(p: Sluggable, siteUrl: string): string {
  return `${siteUrl}${propertyPath(p)}`;
}
