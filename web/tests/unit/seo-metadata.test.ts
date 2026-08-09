import { describe, expect, it } from 'vitest';
import type { Metadata } from 'next';
import { buildPropertyMetadata } from '../../lib/seo-metadata';
import type { PropertyDetail } from '../../lib/schemas';

const SITE_URL = 'https://example.com';

// Twitter is a discriminated union where the bare TwitterMetadata variant
// has no `card` field. The builder always sets `card`, so narrow to the
// variants that have it before reading.
type TwitterWithCard = Extract<NonNullable<Metadata['twitter']>, { card: string }>;
function twitter(m: Metadata): TwitterWithCard {
  if (m.twitter == null || !('card' in m.twitter)) {
    throw new Error('expected twitter card to be set');
  }
  return m.twitter as TwitterWithCard;
}

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

describe('buildPropertyMetadata', () => {
  it('emits a title containing the property name and the brand', () => {
    const m = buildPropertyMetadata(minDetail, SITE_URL);
    expect(m.title).toBe('Sachsenhausen Apartment — Amenity Detector');
  });

  it('uses the property description when present', () => {
    const m = buildPropertyMetadata(
      { ...minDetail, description: 'Top-floor flat with balcony in Sachsenhausen.' },
      SITE_URL,
    );
    expect(m.description).toBe('Top-floor flat with balcony in Sachsenhausen.');
    expect(m.openGraph?.description).toBe('Top-floor flat with balcony in Sachsenhausen.');
  });

  it('falls back to a synthesised description from listing_type + location', () => {
    const m = buildPropertyMetadata(
      { ...minDetail, listing_type: 'rent', locality: 'Bornheim', country_code: 'DE' },
      SITE_URL,
    );
    expect(m.description).toBe('For rent in Bornheim, DE');
  });

  it('sets the canonical URL to the public property URL', () => {
    const m = buildPropertyMetadata(minDetail, SITE_URL);
    expect(m.alternates?.canonical).toBe(`${SITE_URL}/properties/${minDetail.slug}`);
  });

  it('emits an Open Graph image array using the first image, with alt text', () => {
    const m = buildPropertyMetadata(
      {
        ...minDetail,
        images: [
          {
            id: 'img-1',
            file_path: '/1.jpg',
            room_type: 'living_room',
            amenities: [],
            alt_text: 'Sunlit living room with parquet floor',
            caption: null,
            is_primary: true,
            display_order: 0,
          },
        ],
      },
      SITE_URL,
    );
    expect(m.openGraph?.images).toEqual([
      {
        url: `${SITE_URL}/api/images/img-1`,
        alt: 'Sunlit living room with parquet floor',
      },
    ]);
  });

  it('falls back to the property name as image alt when alt_text is null', () => {
    const m = buildPropertyMetadata(
      {
        ...minDetail,
        images: [
          {
            id: 'img-1',
            file_path: '/1.jpg',
            room_type: null,
            amenities: [],
            alt_text: null,
            caption: null,
            is_primary: false,
            display_order: 0,
          },
        ],
      },
      SITE_URL,
    );
    const ogImage = m.openGraph?.images;
    if (!Array.isArray(ogImage) || ogImage.length === 0) {
      throw new Error('expected one OG image');
    }
    const first = ogImage[0];
    if (
      first == null ||
      typeof first === 'string' ||
      first instanceof URL ||
      !('alt' in first)
    ) {
      throw new Error('expected an OG image descriptor with alt');
    }
    expect(first.alt).toBe('Sachsenhausen Apartment');
  });

  it('omits images entirely when the property has no images', () => {
    const m = buildPropertyMetadata(minDetail, SITE_URL);
    expect(m.openGraph?.images).toBeUndefined();
  });

  it('emits a summary_large_image Twitter card with the same image', () => {
    const m = buildPropertyMetadata(
      {
        ...minDetail,
        images: [
          {
            id: 'img-1',
            file_path: '/1.jpg',
            room_type: null,
            amenities: [],
            alt_text: null,
            caption: null,
            is_primary: true,
            display_order: 0,
          },
        ],
      },
      SITE_URL,
    );
    const t = twitter(m);
    expect(t.card).toBe('summary_large_image');
    expect(t.images).toEqual([`${SITE_URL}/api/images/img-1`]);
  });

  it('emits a basic summary Twitter card when no images are present', () => {
    const m = buildPropertyMetadata(minDetail, SITE_URL);
    const t = twitter(m);
    expect(t.card).toBe('summary');
    expect(t.images).toBeUndefined();
  });

  it('sets the Open Graph url to the canonical URL', () => {
    const m = buildPropertyMetadata(minDetail, SITE_URL);
    expect(m.openGraph?.url).toBe(`${SITE_URL}/properties/${minDetail.slug}`);
  });
});
