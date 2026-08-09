import { describe, expect, it } from 'vitest';
import { propertyPath, propertyUrl } from '../../lib/property-url';
import type { PropertyDetail, PropertySummary } from '../../lib/schemas';

const SITE_URL = 'https://example.com';

const minProp: PropertyDetail = {
  id: 'prop-uuid-1',
  name: 'Example',
  status: 'published' as const,
  description: null,
  model_used: null,
  extra_info: null,
  created_at: '2026-05-01T10:00:00',
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
  images: [],
};

describe('propertyPath', () => {
  it('returns the slug-based path when the property has a slug', () => {
    expect(propertyPath({ ...minProp, slug: 'example-abc123' })).toBe(
      '/properties/example-abc123',
    );
  });

  it('falls back to the UUID-based path when slug is missing', () => {
    expect(propertyPath({ ...minProp, slug: null })).toBe('/properties/prop-uuid-1');
  });

  it('falls back to UUID for an empty-string slug too', () => {
    expect(propertyPath({ ...minProp, slug: '' })).toBe('/properties/prop-uuid-1');
  });

  it('works against a PropertySummary just as it does against PropertyDetail', () => {
    const summary: PropertySummary = {
      ...minProp,
      slug: 'summary-slug-xyz',
      image_count: 1,
      first_image_id: 'i1',
    };
    expect(propertyPath(summary)).toBe('/properties/summary-slug-xyz');
  });
});

describe('propertyUrl', () => {
  it('prefixes the absolute site URL onto the slug path', () => {
    const url = propertyUrl({ ...minProp, slug: 'example-abc123' }, SITE_URL);
    expect(url).toBe(`${SITE_URL}/properties/example-abc123`);
  });

  it('prefixes the absolute site URL onto the UUID path when slug is missing', () => {
    expect(propertyUrl(minProp, SITE_URL)).toBe(`${SITE_URL}/properties/prop-uuid-1`);
  });
});
