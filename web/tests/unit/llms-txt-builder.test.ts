import { describe, expect, it } from 'vitest';
import { buildLlmsTxt } from '../../lib/llms-txt-builder';
import type { PropertySummary } from '../../lib/schemas';

const SITE_URL = 'https://example.com';

const base: PropertySummary = {
  id: 'prop-1',
  name: 'Sachsenhausen Apartment',
  status: 'published' as const,
  description: null,
  model_used: null,
  extra_info: null,
  created_at: '2026-05-01T10:00:00',
  image_count: 1,
  first_image_id: 'img-1',
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

describe('buildLlmsTxt', () => {
  it('opens with the site title and a one-sentence summary', () => {
    const text = buildLlmsTxt([], SITE_URL);
    expect(text.startsWith('# Amenity Detector')).toBe(true);
    expect(text).toMatch(/^> .+$/m);
  });

  it('lists the public entry points an agent should know', () => {
    const text = buildLlmsTxt([], SITE_URL);
    expect(text).toContain(`${SITE_URL}/browse`);
    expect(text).toContain(`${SITE_URL}/search`);
    expect(text).toContain(`${SITE_URL}/api/feed.jsonl`);
    expect(text).toContain(`${SITE_URL}/sitemap.xml`);
  });

  it('describes the structured data shape so agents do not need to parse HTML', () => {
    const text = buildLlmsTxt([], SITE_URL);
    expect(text).toMatch(/JSON-LD/);
    expect(text).toMatch(/Accommodation/);
    expect(text).toMatch(/RealEstateListing/);
  });

  it('groups listings into Rentals and For sale sections', () => {
    const rent: PropertySummary = { ...base, id: 'r1', name: 'Rent A', listing_type: 'rent' };
    const sale: PropertySummary = { ...base, id: 's1', name: 'Sale A', listing_type: 'sale' };
    const text = buildLlmsTxt([rent, sale], SITE_URL);
    const rentIdx = text.indexOf('## Rentals');
    const saleIdx = text.indexOf('## For sale');
    expect(rentIdx).toBeGreaterThan(-1);
    expect(saleIdx).toBeGreaterThan(-1);
    // Each listing should appear under its own section
    const rentSection = text.slice(rentIdx, saleIdx > rentIdx ? saleIdx : text.length);
    expect(rentSection).toContain('Rent A');
    expect(rentSection).not.toContain('Sale A');
  });

  it('groups untyped listings into Other listings', () => {
    const text = buildLlmsTxt([base], SITE_URL);
    expect(text).toContain('## Other listings');
    expect(text).toContain('Sachsenhausen Apartment');
  });

  it('includes location next to each listing when available', () => {
    const withLoc: PropertySummary = {
      ...base,
      listing_type: 'rent',
      locality: 'Sachsenhausen',
      country_code: 'DE',
    };
    const text = buildLlmsTxt([withLoc], SITE_URL);
    expect(text).toContain('(Sachsenhausen, DE)');
  });

  it('skips drafts (image_count === 0) so agents do not see incomplete listings', () => {
    const draft: PropertySummary = { ...base, id: 'd1', name: 'Draft', image_count: 0 };
    const text = buildLlmsTxt([draft], SITE_URL);
    expect(text).not.toContain('Draft');
  });
});
