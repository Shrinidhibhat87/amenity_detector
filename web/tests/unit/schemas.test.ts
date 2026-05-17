import { describe, it, expect } from 'vitest';

import {
  PropertySummary,
  PropertyDetail,
  DetectedAmenity,
  PropertyImage,
  PropertyCreateRequest,
  Health,
  SearchFilter,
} from '@/lib/schemas';

// JS reference shape we want to validate. Mirrors what FastAPI returns
// from GET /api/v1/properties/{id} after Phase 9.
const detailFixture = {
  id: 'p_abc123',
  name: 'Bendel 26',
  description: 'Bright 3-room apartment in Frankenberger Viertel.',
  model_used: 'openai/gpt-4o-mini',
  extra_info: null,
  created_at: '2026-05-10T18:32:11.000Z',
  slug: 'bendel-26-abc123',
  listing_type: 'rent',
  price: '980',
  currency: 'EUR',
  price_period: 'monthly',
  num_bedrooms: 2,
  num_bathrooms: 1,
  area_sqm: '78.0',
  property_type: 'apartment',
  furnishing: 'semi_furnished',
  available_from: '2026-06-01',
  locality: 'Frankenberger Viertel',
  postal_code: '52066',
  country_code: 'DE',
  latitude: '50.776',
  longitude: '6.084',
  owner_email: 'owner@example.com',
  images: [
    {
      id: 'img_1',
      file_path: 'storage/images/img_1.jpg',
      room_type: 'kitchen',
      amenities: [
        {
          id: 'a_1',
          amenity_name: 'refrigerator',
          room_type: 'kitchen',
          is_present: true,
          confidence: 0.92,
        },
      ],
      alt_text: 'Bright modern kitchen with stainless fridge and tiled splashback',
      caption: 'Kitchen, daytime',
      is_primary: true,
      display_order: 0,
    },
  ],
};

describe('PropertyDetail', () => {
  it('accepts a Phase 9 detail payload from the API', () => {
    const parsed = PropertyDetail.parse(detailFixture);
    expect(parsed.id).toBe('p_abc123');
    expect(parsed.images).toHaveLength(1);
    expect(parsed.images[0]!.amenities[0]!.amenity_name).toBe('refrigerator');
    // Decimal-as-string from Pydantic should coerce to number on the client.
    expect(typeof parsed.price).toBe('number');
    expect(parsed.price).toBe(980);
  });

  it('accepts a legacy row with all Phase 9 fields null', () => {
    const legacy = {
      ...detailFixture,
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
    expect(() => PropertyDetail.parse(legacy)).not.toThrow();
  });

  it('rejects an unknown listing_type', () => {
    const bad = { ...detailFixture, listing_type: 'lease' };
    expect(() => PropertyDetail.parse(bad)).toThrow();
  });

  it('rejects out-of-range latitude', () => {
    const bad = { ...detailFixture, latitude: '120' };
    expect(() => PropertyDetail.parse(bad)).toThrow();
  });
});

describe('PropertySummary', () => {
  it('accepts list-view shape with image_count', () => {
    const summary = {
      id: 'p_abc123',
      name: 'Bendel 26',
      description: null,
      model_used: 'openai/gpt-4o-mini',
      extra_info: null,
      created_at: '2026-05-10T18:32:11.000Z',
      image_count: 4,
      first_image_id: 'img_1',
      slug: 'bendel-26-abc123',
      listing_type: 'rent',
      price: '980',
      currency: 'EUR',
      price_period: 'monthly',
      num_bedrooms: 2,
      num_bathrooms: 1,
      area_sqm: '78',
      property_type: 'apartment',
      furnishing: 'semi_furnished',
      available_from: null,
      locality: 'Frankenberger Viertel',
      postal_code: '52066',
      country_code: 'DE',
      latitude: '50.776',
      longitude: '6.084',
      owner_email: null,
    };
    const parsed = PropertySummary.parse(summary);
    expect(parsed.image_count).toBe(4);
    expect(parsed.first_image_id).toBe('img_1');
  });
});

describe('DetectedAmenity + PropertyImage', () => {
  it('accepts confidence as null (manually-added amenity)', () => {
    const a = DetectedAmenity.parse({
      id: 'a_2',
      amenity_name: 'sofa',
      room_type: 'living_room',
      is_present: true,
      confidence: null,
    });
    expect(a.confidence).toBeNull();
  });

  it('defaults is_primary and display_order when omitted on PropertyImage', () => {
    const img = PropertyImage.parse({
      id: 'img_x',
      file_path: 'storage/images/img_x.jpg',
      room_type: null,
      amenities: [],
    });
    expect(img.is_primary).toBe(false);
    expect(img.display_order).toBe(0);
    expect(img.alt_text).toBeNull();
  });
});

describe('PropertyCreateRequest', () => {
  it('accepts the minimal body {name, model_name}', () => {
    const parsed = PropertyCreateRequest.parse({
      name: 'Bendel 26',
      model_name: 'openai/gpt-4o-mini',
    });
    expect(parsed.name).toBe('Bendel 26');
  });

  it('upper-cases country_code to match FastAPI StringConstraints', () => {
    const parsed = PropertyCreateRequest.parse({
      name: 'Bendel 26',
      model_name: 'openai/gpt-4o-mini',
      country_code: 'de',
    });
    expect(parsed.country_code).toBe('DE');
  });

  it('rejects num_bedrooms > 50', () => {
    expect(() =>
      PropertyCreateRequest.parse({
        name: 'x',
        model_name: 'y',
        num_bedrooms: 51,
      }),
    ).toThrow();
  });
});

describe('Health', () => {
  it('accepts the /health shape', () => {
    expect(() =>
      Health.parse({ status: 'ok', database: 'ok', version: '1.0.0' }),
    ).not.toThrow();
  });
});

describe('SearchFilter', () => {
  it('applies defaults when empty', () => {
    const parsed = SearchFilter.parse({});
    expect(parsed.query).toBe('');
    expect(parsed.rooms).toBe('any');
    expect(parsed.amenities).toEqual([]);
    expect(parsed.sort).toBe('recent');
  });

  it('rejects unknown sort', () => {
    expect(() => SearchFilter.parse({ sort: 'cheapest' })).toThrow();
  });
});
