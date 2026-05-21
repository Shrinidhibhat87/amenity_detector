import { describe, expect, it } from 'vitest';
import { buildSitemap } from '../../lib/sitemap-builder';
import type { PropertySummary } from '../../lib/schemas';

const SITE_URL = 'https://example.com';
const NOW = new Date('2026-05-21T10:00:00Z');

const baseSummary: PropertySummary = {
  id: 'prop-1',
  name: 'Sachsenhausen Apartment',
  description: null,
  model_used: null,
  extra_info: null,
  created_at: '2026-05-01T10:00:00',
  image_count: 0,
  first_image_id: null,
  slug: null,
  listing_type: null,
  price: null,
  currency: null,
  price_period: null,
  num_bedrooms: null,
  num_bathrooms: null,
  area_sqm: null,
  property_type: null,
  furnishing: null,
  available_from: null,
  locality: null,
  postal_code: null,
  country_code: null,
  latitude: null,
  longitude: null,
  owner_email: null,
};

describe('buildSitemap', () => {
  it('emits Home, Browse, and Search as static entries with lastmod=now', () => {
    const entries = buildSitemap([], SITE_URL, NOW);
    const urls = entries.map((e) => e.url);
    expect(urls).toContain(SITE_URL);
    expect(urls).toContain(`${SITE_URL}/browse`);
    expect(urls).toContain(`${SITE_URL}/search`);
    const home = entries.find((e) => e.url === SITE_URL);
    expect(home?.priority).toBe(1);
    expect(home?.lastModified).toEqual(NOW);
  });

  it('skips listings with no images (drafts)', () => {
    const entries = buildSitemap([baseSummary], SITE_URL, NOW);
    expect(entries.some((e) => e.url.includes('/properties/'))).toBe(false);
  });

  it('includes listings with at least one image', () => {
    const ready: PropertySummary = {
      ...baseSummary,
      image_count: 3,
      first_image_id: 'img-1',
    };
    const entries = buildSitemap([ready], SITE_URL, NOW);
    const listing = entries.find((e) => e.url === `${SITE_URL}/properties/prop-1`);
    expect(listing).toBeDefined();
    expect(listing?.lastModified).toEqual(new Date('2026-05-01T10:00:00'));
    expect(listing?.changeFrequency).toBe('weekly');
  });

  it('weights priority by listing completeness — full metadata > sparse', () => {
    const full: PropertySummary = {
      ...baseSummary,
      id: 'full',
      image_count: 2,
      first_image_id: 'i1',
      description: 'A spacious 3BHK with parquet floors.',
      listing_type: 'rent',
    };
    const sparse: PropertySummary = {
      ...baseSummary,
      id: 'sparse',
      image_count: 1,
      first_image_id: 'i2',
    };
    const entries = buildSitemap([full, sparse], SITE_URL, NOW);
    const fullEntry = entries.find((e) => e.url.endsWith('/full'));
    const sparseEntry = entries.find((e) => e.url.endsWith('/sparse'));
    expect(fullEntry?.priority).toBe(0.8);
    expect(sparseEntry?.priority).toBe(0.6);
  });

  it('keeps static-page entries ordered before listing entries', () => {
    const ready: PropertySummary = {
      ...baseSummary,
      image_count: 1,
      first_image_id: 'img-1',
    };
    const entries = buildSitemap([ready], SITE_URL, NOW);
    const firstListingIdx = entries.findIndex((e) => e.url.includes('/properties/'));
    const homeIdx = entries.findIndex((e) => e.url === SITE_URL);
    const browseIdx = entries.findIndex((e) => e.url === `${SITE_URL}/browse`);
    expect(homeIdx).toBeLessThan(firstListingIdx);
    expect(browseIdx).toBeLessThan(firstListingIdx);
  });
});
