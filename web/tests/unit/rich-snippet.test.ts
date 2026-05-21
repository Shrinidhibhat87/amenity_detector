/**
 * Rich-snippet validation tests.
 *
 * The per-builder unit tests in json-ld.test.ts assert that each helper
 * emits the fields we asked for. This file is one layer higher: it
 * exercises buildPropertyJsonLd on realistic fixtures and asserts that
 * the resulting payload satisfies the *external* constraints we promise
 * search engines and AI agents — Google's documented rich-result
 * requirements and a few defensive checks against silent-failure modes.
 *
 * Catches:
 * - Required-field drift (someone removes name / image / offers)
 * - null/undefined/NaN leaking into JSON.stringify output (Google's
 *   structured-data validator fails on these)
 * - Out-of-order BreadcrumbList positions (silently drops the trail)
 * - Non-absolute URLs (Google needs canonical absolute URLs)
 */

import { describe, expect, it } from 'vitest';
import { buildPropertyJsonLd } from '../../lib/json-ld';
import type { PropertyDetail } from '../../lib/schemas';

const SITE_URL = 'https://amenity.example.com';

function realisticRent(): PropertyDetail {
  return {
    id: 'rent-001',
    name: 'Sachsenhausen 3BHK Apartment',
    description: 'Top-floor flat with parquet floors and a south-facing balcony.',
    model_used: 'openai/gpt-4o-mini',
    extra_info: null,
    created_at: '2026-05-01T10:00:00',
    slug: 'sachsenhausen-3bhk-abc123',
    listing_type: 'rent',
    price: 1500,
    currency: 'EUR',
    price_period: 'monthly',
    num_bedrooms: 3,
    num_bathrooms: 2,
    area_sqm: 82.5,
    property_type: 'apartment',
    furnishing: 'semi_furnished',
    available_from: '2026-06-01',
    locality: 'Sachsenhausen',
    postal_code: '60594',
    country_code: 'DE',
    latitude: 50.103,
    longitude: 8.681,
    owner_email: null,
    images: [
      {
        id: 'img-1',
        file_path: '/1.jpg',
        room_type: 'living_room',
        amenities: [
          { id: 'a1', amenity_name: 'fireplace', room_type: 'living_room', is_present: true, confidence: 0.95 },
          { id: 'a2', amenity_name: 'parquet', room_type: 'living_room', is_present: true, confidence: 0.88 },
        ],
        alt_text: 'Sunlit living room with fireplace',
        caption: 'Living room',
        is_primary: true,
        display_order: 0,
      },
      {
        id: 'img-2',
        file_path: '/2.jpg',
        room_type: 'kitchen',
        amenities: [
          { id: 'a3', amenity_name: 'refrigerator', room_type: 'kitchen', is_present: true, confidence: 0.92 },
        ],
        alt_text: null,
        caption: null,
        is_primary: false,
        display_order: 1,
      },
    ],
  };
}

function realisticSale(): PropertyDetail {
  return {
    ...realisticRent(),
    id: 'sale-001',
    name: 'Bornheim 4BHK House',
    listing_type: 'sale',
    price: 685000,
    currency: 'EUR',
    price_period: null,
    locality: 'Bornheim',
    property_type: 'house',
  };
}

// ── No-leak helpers ──────────────────────────────────────────────────────────

function containsNullishOrNaN(serialized: string): boolean {
  return /:\s*null\b/.test(serialized) || serialized.includes('NaN');
}

// ── Rent path ────────────────────────────────────────────────────────────────

describe('rich-snippet rent payload', () => {
  const entries = buildPropertyJsonLd(realisticRent(), SITE_URL);
  const primary = entries[0];
  const breadcrumb = entries[1];

  if (primary == null || breadcrumb == null) throw new Error('expected 2 entries');

  it('uses an Accommodation subtype (Apartment) for a rent listing', () => {
    expect(primary['@type']).toBe('Apartment');
  });

  it('has the Google-recommended Accommodation fields populated', () => {
    expect(primary).toMatchObject({
      name: 'Sachsenhausen 3BHK Apartment',
      url: `${SITE_URL}/properties/rent-001`,
      numberOfRooms: 3,
    });
    const addr = (primary as { address?: { addressLocality?: string; addressCountry?: string } }).address;
    expect(addr?.addressLocality).toBe('Sachsenhausen');
    expect(addr?.addressCountry).toBe('DE');
    const image = (primary as { image?: string[] }).image;
    expect(image?.length).toBeGreaterThanOrEqual(1);
    if (image != null) {
      for (const url of image) expect(url).toMatch(/^https:\/\//);
    }
  });

  it('emits amenityFeature for the detected amenities, deduped', () => {
    const features = (primary as { amenityFeature?: Array<{ name: string }> }).amenityFeature;
    expect(features?.map((f) => f.name)).toEqual(['fireplace', 'parquet', 'refrigerator']);
  });

  it('serialises with no null or NaN leaking through', () => {
    const json = JSON.stringify(primary);
    expect(containsNullishOrNaN(json)).toBe(false);
  });
});

// ── Sale path ────────────────────────────────────────────────────────────────

describe('rich-snippet sale payload', () => {
  const [primary] = buildPropertyJsonLd(realisticSale(), SITE_URL);
  if (primary == null) throw new Error('expected primary entry');

  it('uses RealEstateListing for a sale listing', () => {
    expect(primary['@type']).toBe('RealEstateListing');
  });

  it('has an Offer with price + priceCurrency — Google requires both', () => {
    const offer = (primary as { offers?: { price: number; priceCurrency: string } }).offers;
    expect(offer?.price).toBe(685000);
    expect(offer?.priceCurrency).toBe('EUR');
  });

  it('serialises with no null or NaN leaking through', () => {
    const json = JSON.stringify(primary);
    expect(containsNullishOrNaN(json)).toBe(false);
  });
});

// ── Breadcrumb path ──────────────────────────────────────────────────────────

describe('rich-snippet BreadcrumbList', () => {
  it('has positions 1, 2, 3 in order with absolute item URLs', () => {
    const [, breadcrumb] = buildPropertyJsonLd(realisticRent(), SITE_URL);
    if (breadcrumb == null || breadcrumb['@type'] !== 'BreadcrumbList') {
      throw new Error('expected BreadcrumbList as the second entry');
    }
    const items = breadcrumb.itemListElement;
    expect(items.map((i) => i.position)).toEqual([1, 2, 3]);
    for (const i of items) {
      expect(i.item).toMatch(/^https:\/\//);
    }
  });
});

// ── Payload sanity ───────────────────────────────────────────────────────────

describe('rich-snippet payload sanity', () => {
  it('keeps a realistic listing JSON payload under 8KB', () => {
    const entries = buildPropertyJsonLd(realisticRent(), SITE_URL);
    const total = entries.reduce((acc, e) => acc + JSON.stringify(e).length, 0);
    expect(total).toBeLessThan(8 * 1024);
  });
});
