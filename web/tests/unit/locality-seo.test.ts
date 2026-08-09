/**
 * Tests for the locality enrichment SEO wiring:
 *   - JSON-LD `geo` backfills from the locality insight when the listing has no
 *     coordinate of its own (and the listing's own geo still wins when present).
 *   - llms.txt advertises the neighbourhood data + the mandatory OSM attribution
 *     so AI agents know it is there and how it is licensed.
 *   - The PropertyDetail Zod schema parses a nested locality_insight.
 */

import { describe, expect, it } from 'vitest';
import { buildAccommodation } from '../../lib/json-ld';
import { buildLlmsTxt } from '../../lib/llms-txt-builder';
import { PropertyDetail, type LocalityInsight } from '../../lib/schemas';

const SITE_URL = 'https://amenity.example.com';

function insight(overrides: Partial<LocalityInsight> = {}): LocalityInsight {
  return {
    display_name: '60311 Frankfurt am Main, Germany',
    latitude: 50.1109,
    longitude: 8.6821,
    radius_m: 1000,
    blurb: 'Central spot with a school and two parks nearby.',
    category_counts: { school: 1, park: 2 },
    pois: [],
    attribution: '© OpenStreetMap contributors',
    ...overrides,
  };
}

function detail(overrides: Record<string, unknown> = {}): PropertyDetail {
  return {
    id: 'p1',
    name: 'Frankfurt flat',
    status: 'published' as const,
    description: null,
    model_used: null,
    extra_info: null,
    created_at: '2026-06-14T10:00:00',
    slug: 'frankfurt-flat-abc123',
    listing_type: 'rent',
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
    locality_insight: insight(),
    ...overrides,
  } as PropertyDetail;
}

describe('JSON-LD geo backfill from locality insight', () => {
  it('uses the insight coordinate when the listing has none', () => {
    const ld = buildAccommodation(detail(), SITE_URL);
    expect(ld.geo).toEqual({ '@type': 'GeoCoordinates', latitude: 50.1109, longitude: 8.6821 });
  });

  it("prefers the listing's own coordinate over the insight", () => {
    const ld = buildAccommodation(detail({ latitude: 48.1, longitude: 11.5 }), SITE_URL);
    expect(ld.geo).toEqual({ '@type': 'GeoCoordinates', latitude: 48.1, longitude: 11.5 });
  });

  it('omits geo when neither listing nor insight has a coordinate', () => {
    const ld = buildAccommodation(
      detail({ locality_insight: insight({ latitude: null, longitude: null }) }),
      SITE_URL,
    );
    expect(ld.geo).toBeUndefined();
  });
});

describe('llms.txt neighbourhood advertisement', () => {
  it('mentions neighbourhood data and the OSM attribution', () => {
    const txt = buildLlmsTxt([], SITE_URL);
    expect(txt).toContain('neighbourhood');
    expect(txt).toContain('© OpenStreetMap contributors');
  });
});

describe('PropertyDetail schema with nested locality_insight', () => {
  it('parses a nested insight, stripping extra POI tag fields', () => {
    const parsed = PropertyDetail.parse({
      id: 'p1',
      name: 'Frankfurt flat',
      description: null,
      model_used: null,
      extra_info: null,
      created_at: '2026-06-14T10:00:00',
      images: [],
      locality_insight: {
        display_name: 'Frankfurt',
        latitude: '50.1109', // Decimal-as-string from Pydantic
        longitude: '8.6821',
        radius_m: 1000,
        blurb: 'Nice area.',
        category_counts: { school: 1 },
        pois: [
          {
            category: 'school',
            name: 'Goethe-Schule',
            latitude: 50.111,
            longitude: 8.682,
            distance_m: 120,
            osm_type: 'node',
            osm_id: 1,
            tags: { amenity: 'school' }, // extra field — should be stripped
          },
        ],
        attribution: '© OpenStreetMap contributors',
      },
    });

    expect(parsed.locality_insight?.latitude).toBe(50.1109);
    expect(parsed.locality_insight?.pois[0]).not.toHaveProperty('tags');
  });

  it('accepts a property with no locality_insight', () => {
    const parsed = PropertyDetail.parse({
      id: 'p2',
      name: 'Bare',
      description: null,
      model_used: null,
      extra_info: null,
      created_at: '2026-06-14T10:00:00',
      images: [],
    });
    expect(parsed.locality_insight ?? null).toBeNull();
  });
});
