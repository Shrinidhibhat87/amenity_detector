import { describe, expect, it } from 'vitest';
import {
  buildAccommodation,
  buildBreadcrumbList,
  buildPropertyJsonLd,
  buildRealEstateListing,
} from '../../lib/json-ld';
import type { PropertyDetail } from '../../lib/schemas';

const SITE_URL = 'https://example.com';

// Minimal PropertyDetail fixture with every nullable field set to null.
// Individual tests spread over this and override only what they exercise.
const minDetail: PropertyDetail = {
  id: 'prop-1',
  name: 'Sachsenhausen Apartment',
  status: 'published' as const,
  description: null,
  model_used: null,
  extra_info: null,
  created_at: '2026-05-01T10:00:00',
  slug: 'sachsenhausen-apartment-abc123',
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

// ── buildAccommodation (rent path) ───────────────────────────────────────────

describe('buildAccommodation', () => {
  it('emits a schema.org Accommodation with the core required fields', () => {
    const p: PropertyDetail = {
      ...minDetail,
      listing_type: 'rent',
      num_bedrooms: 3,
      num_bathrooms: 2,
      area_sqm: 82.5,
      property_type: 'apartment',
      locality: 'Sachsenhausen',
      postal_code: '60594',
      country_code: 'DE',
    };
    const ld = buildAccommodation(p, SITE_URL);
    expect(ld['@context']).toBe('https://schema.org');
    expect(ld['@type']).toBe('Apartment');
    expect(ld.name).toBe('Sachsenhausen Apartment');
    expect(ld.numberOfRooms).toBe(3);
    expect(ld.numberOfBathroomsTotal).toBe(2);
    expect(ld.floorSize).toEqual({
      '@type': 'QuantitativeValue',
      value: 82.5,
      unitCode: 'MTK',
    });
    expect(ld.address).toEqual({
      '@type': 'PostalAddress',
      addressLocality: 'Sachsenhausen',
      postalCode: '60594',
      addressCountry: 'DE',
    });
    expect(ld.url).toBe(`${SITE_URL}/properties/${p.slug}`);
  });

  it('uses generic Accommodation when property_type is missing', () => {
    const p: PropertyDetail = { ...minDetail, listing_type: 'rent' };
    const ld = buildAccommodation(p, SITE_URL);
    expect(ld['@type']).toBe('Accommodation');
  });

  it('omits address block when no locality/postal/country present', () => {
    const ld = buildAccommodation({ ...minDetail, listing_type: 'rent' }, SITE_URL);
    expect(ld.address).toBeUndefined();
  });

  it('emits geo coordinates only when both lat and lng are present', () => {
    const ld = buildAccommodation(
      { ...minDetail, listing_type: 'rent', latitude: 50.103, longitude: 8.681 },
      SITE_URL,
    );
    expect(ld.geo).toEqual({
      '@type': 'GeoCoordinates',
      latitude: 50.103,
      longitude: 8.681,
    });
  });

  it('skips geo block when latitude is set but longitude is missing', () => {
    const ld = buildAccommodation(
      { ...minDetail, listing_type: 'rent', latitude: 50.103, longitude: null },
      SITE_URL,
    );
    expect(ld.geo).toBeUndefined();
  });

  it('emits an image array of absolute URLs in display order', () => {
    const p: PropertyDetail = {
      ...minDetail,
      listing_type: 'rent',
      images: [
        {
          id: 'img-2',
          file_path: '/2.jpg',
          room_type: 'kitchen',
          amenities: [],
          alt_text: null,
          caption: null,
          is_primary: false,
          display_order: 1,
        },
        {
          id: 'img-1',
          file_path: '/1.jpg',
          room_type: 'living_room',
          amenities: [],
          alt_text: null,
          caption: null,
          is_primary: true,
          display_order: 0,
        },
      ],
    };
    const ld = buildAccommodation(p, SITE_URL);
    expect(ld.image).toEqual([
      `${SITE_URL}/api/images/img-1`,
      `${SITE_URL}/api/images/img-2`,
    ]);
  });

  it('never emits null fields — only set keys are present', () => {
    const ld = buildAccommodation({ ...minDetail, listing_type: 'rent' }, SITE_URL);
    expect(Object.values(ld)).not.toContain(null);
  });

  it('emits LocationFeatureSpecification entries for present amenities, deduped', () => {
    const ld = buildAccommodation(
      {
        ...minDetail,
        listing_type: 'rent',
        images: [
          {
            id: 'img-1',
            file_path: '/1.jpg',
            room_type: 'kitchen',
            amenities: [
              { id: 'a1', amenity_name: 'refrigerator', room_type: 'kitchen', is_present: true, confidence: 0.9 },
              { id: 'a2', amenity_name: 'oven', room_type: 'kitchen', is_present: true, confidence: 0.8 },
              { id: 'a3', amenity_name: 'sink', room_type: 'kitchen', is_present: false, confidence: 0.7 },
            ],
            alt_text: null,
            caption: null,
            is_primary: true,
            display_order: 0,
          },
          {
            id: 'img-2',
            file_path: '/2.jpg',
            room_type: 'living_room',
            amenities: [
              // Duplicate name across images should appear only once
              { id: 'a4', amenity_name: 'refrigerator', room_type: 'kitchen', is_present: true, confidence: 0.6 },
              { id: 'a5', amenity_name: 'fireplace', room_type: 'living_room', is_present: true, confidence: 0.95 },
            ],
            alt_text: null,
            caption: null,
            is_primary: false,
            display_order: 1,
          },
        ],
      },
      SITE_URL,
    );
    expect(ld.amenityFeature).toEqual([
      { '@type': 'LocationFeatureSpecification', name: 'refrigerator', value: true },
      { '@type': 'LocationFeatureSpecification', name: 'oven', value: true },
      { '@type': 'LocationFeatureSpecification', name: 'fireplace', value: true },
    ]);
  });

  it('falls back to a UUID-based url when the property has no slug', () => {
    const ld = buildAccommodation({ ...minDetail, slug: null }, SITE_URL);
    expect(ld.url).toBe(`${SITE_URL}/properties/${minDetail.id}`);
  });

  it('omits amenityFeature when no present amenities exist', () => {
    const ld = buildAccommodation(
      {
        ...minDetail,
        listing_type: 'rent',
        images: [
          {
            id: 'img-1',
            file_path: '/1.jpg',
            room_type: 'kitchen',
            amenities: [
              { id: 'a1', amenity_name: 'sink', room_type: 'kitchen', is_present: false, confidence: 0.1 },
            ],
            alt_text: null,
            caption: null,
            is_primary: false,
            display_order: 0,
          },
        ],
      },
      SITE_URL,
    );
    expect(ld.amenityFeature).toBeUndefined();
  });
});

// ── buildRealEstateListing (sale path) ───────────────────────────────────────

describe('buildRealEstateListing', () => {
  it('emits a RealEstateListing with Offer containing price + currency', () => {
    const p: PropertyDetail = {
      ...minDetail,
      listing_type: 'sale',
      price: 450000,
      currency: 'EUR',
      num_bedrooms: 4,
      locality: 'Bornheim',
      country_code: 'DE',
    };
    const ld = buildRealEstateListing(p, SITE_URL);
    expect(ld['@type']).toBe('RealEstateListing');
    expect(ld.name).toBe('Sachsenhausen Apartment');
    expect(ld.url).toBe(`${SITE_URL}/properties/${p.slug}`);
    expect(ld.offers).toEqual({
      '@type': 'Offer',
      price: 450000,
      priceCurrency: 'EUR',
      availability: 'https://schema.org/InStock',
      url: `${SITE_URL}/properties/${p.slug}`,
    });
  });

  it('omits offers when the property has no price', () => {
    const p: PropertyDetail = { ...minDetail, listing_type: 'sale' };
    const ld = buildRealEstateListing(p, SITE_URL);
    expect(ld.offers).toBeUndefined();
  });

  it('falls back to EUR when currency is missing but price is set', () => {
    const p: PropertyDetail = { ...minDetail, listing_type: 'sale', price: 300000 };
    const ld = buildRealEstateListing(p, SITE_URL);
    expect(ld.offers?.priceCurrency).toBe('EUR');
  });
});

// ── buildBreadcrumbList ──────────────────────────────────────────────────────

describe('buildBreadcrumbList', () => {
  it('emits three items in order: Home, Browse, Property', () => {
    const ld = buildBreadcrumbList(
      { ...minDetail, name: 'Frankfurt Loft' },
      SITE_URL,
    );
    expect(ld['@type']).toBe('BreadcrumbList');
    expect(ld.itemListElement).toHaveLength(3);
    expect(ld.itemListElement[0]).toEqual({
      '@type': 'ListItem',
      position: 1,
      name: 'Home',
      item: SITE_URL,
    });
    expect(ld.itemListElement[1]).toEqual({
      '@type': 'ListItem',
      position: 2,
      name: 'Browse',
      item: `${SITE_URL}/browse`,
    });
    expect(ld.itemListElement[2]?.name).toBe('Frankfurt Loft');
    expect(ld.itemListElement[2]?.item).toBe(
      `${SITE_URL}/properties/sachsenhausen-apartment-abc123`,
    );
  });
});

// ── buildPropertyJsonLd (router) ─────────────────────────────────────────────

describe('buildPropertyJsonLd', () => {
  it('emits Accommodation + BreadcrumbList for rent listings', () => {
    const ld = buildPropertyJsonLd(
      { ...minDetail, listing_type: 'rent' },
      SITE_URL,
    );
    const types = ld.map((entry) => entry['@type']);
    expect(types).toContain('Accommodation');
    expect(types).toContain('BreadcrumbList');
    expect(types).not.toContain('RealEstateListing');
  });

  it('emits RealEstateListing + BreadcrumbList for sale listings', () => {
    const ld = buildPropertyJsonLd(
      { ...minDetail, listing_type: 'sale' },
      SITE_URL,
    );
    const types = ld.map((entry) => entry['@type']);
    expect(types).toContain('RealEstateListing');
    expect(types).toContain('BreadcrumbList');
    expect(types).not.toContain('Accommodation');
  });

  it('defaults to Accommodation when listing_type is null', () => {
    const ld = buildPropertyJsonLd({ ...minDetail }, SITE_URL);
    const types = ld.map((entry) => entry['@type']);
    expect(types).toContain('Accommodation');
  });
});
