import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  ApiError,
  applyClientFilters,
  createProperty,
  describeProperty,
  getImageUrl,
  getProperty,
  listModels,
  listProperties,
  patchImage,
  patchProperty,
  searchProperties,
  uploadImage,
} from '../../lib/api';

// jsdom has `window`, so getBaseUrl() branches to NEXT_PUBLIC_API_BASE_URL.
const BASE = 'http://test-api:8000';
vi.stubEnv('NEXT_PUBLIC_API_BASE_URL', BASE);

const mockFetch = vi.fn<typeof fetch>();
vi.stubGlobal('fetch', mockFetch);

// Minimal PropertySummary fixture — all nullable fields set to null.
const minSummary = {
  id: 'prop-1',
  name: 'Ocean View Apartment',
  created_at: '2024-06-01T10:00:00',
  description: null,
  model_used: null,
  extra_info: null,
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

function okResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => body,
  } as unknown as Response;
}

function errResponse(status: number): Response {
  return {
    ok: false,
    status,
    statusText: `HTTP ${status}`,
    json: async () => null,
  } as unknown as Response;
}

beforeEach(() => {
  mockFetch.mockReset();
});

// ── getImageUrl ────────────────────────────────────────────────────────────────

describe('getImageUrl', () => {
  it('builds the correct URL from the base URL', () => {
    expect(getImageUrl('img-abc-123')).toBe(`${BASE}/api/v1/images/img-abc-123`);
  });
});

// ── listProperties ─────────────────────────────────────────────────────────────

describe('listProperties', () => {
  it('parses and returns a valid array on 200', async () => {
    mockFetch.mockResolvedValueOnce(okResponse([minSummary]));
    const result = await listProperties();
    expect(result).toHaveLength(1);
    expect(result[0]?.name).toBe('Ocean View Apartment');
    expect(result[0]?.image_count).toBe(0);
  });

  it('sends offset and limit as query params', async () => {
    mockFetch.mockResolvedValueOnce(okResponse([]));
    await listProperties({ offset: 20, limit: 10 });
    const calledUrl = (mockFetch.mock.calls[0] as [string])[0];
    expect(calledUrl).toContain('offset=20');
    expect(calledUrl).toContain('limit=10');
  });

  it('omits query string when no options given', async () => {
    mockFetch.mockResolvedValueOnce(okResponse([]));
    await listProperties();
    const calledUrl = (mockFetch.mock.calls[0] as [string])[0];
    expect(calledUrl).not.toContain('?');
  });

  it('throws ApiError on a non-2xx response', async () => {
    mockFetch.mockResolvedValueOnce(errResponse(500));
    await expect(listProperties()).rejects.toThrow(ApiError);
  });

  it('ApiError carries the HTTP status code', async () => {
    mockFetch.mockResolvedValueOnce(errResponse(404));
    await expect(listProperties()).rejects.toMatchObject({ status: 404 });
  });

  it('coerces decimal-as-string price to a number', async () => {
    mockFetch.mockResolvedValueOnce(okResponse([{ ...minSummary, price: '1200.50' }]));
    const [prop] = await listProperties();
    expect(typeof prop?.price).toBe('number');
    expect(prop?.price).toBeCloseTo(1200.5);
  });

  it('accepts listing_type enum values', async () => {
    mockFetch.mockResolvedValueOnce(okResponse([{ ...minSummary, listing_type: 'rent' }]));
    const [prop] = await listProperties();
    expect(prop?.listing_type).toBe('rent');
  });
});

// ── getProperty ────────────────────────────────────────────────────────────────

describe('getProperty', () => {
  const minDetail = { ...minSummary, images: [] };

  it('parses PropertyDetail on 200', async () => {
    mockFetch.mockResolvedValueOnce(okResponse(minDetail));
    const prop = await getProperty('prop-1');
    expect(prop.id).toBe('prop-1');
    expect(prop.images).toEqual([]);
  });

  it('requests the correct URL', async () => {
    mockFetch.mockResolvedValueOnce(okResponse(minDetail));
    await getProperty('prop-abc');
    const calledUrl = (mockFetch.mock.calls[0] as [string])[0];
    expect(calledUrl).toBe(`${BASE}/api/v1/properties/prop-abc`);
  });

  it('throws ApiError on 404', async () => {
    mockFetch.mockResolvedValueOnce(errResponse(404));
    await expect(getProperty('missing')).rejects.toThrow(ApiError);
  });

  it('ApiError on 404 has status 404', async () => {
    mockFetch.mockResolvedValueOnce(errResponse(404));
    await expect(getProperty('missing')).rejects.toMatchObject({ status: 404 });
  });
});

// ── Mutations ──────────────────────────────────────────────────────────────────

describe('listModels', () => {
  it('parses ModelInfo array', async () => {
    mockFetch.mockResolvedValueOnce(
      okResponse([{ name: 'gpt-4o-mini', available: true, description: 'GPT' }]),
    );
    const models = await listModels();
    expect(models[0]?.name).toBe('gpt-4o-mini');
    expect(models[0]?.available).toBe(true);
  });
});

describe('createProperty', () => {
  const minDetail = { ...minSummary, images: [] };

  it('POSTs JSON body and parses response', async () => {
    mockFetch.mockResolvedValueOnce(
      okResponse({ property_id: 'p-9', message: 'created', property: minDetail }),
    );
    const res = await createProperty({ name: 'X', model_name: 'gpt-4o-mini' });
    expect(res.property_id).toBe('p-9');

    const init = (mockFetch.mock.calls[0] as [string, RequestInit])[1];
    expect(init.method).toBe('POST');
    expect(init.body).toBe(JSON.stringify({ name: 'X', model_name: 'gpt-4o-mini' }));
  });

  it('hits /api/v1/properties/', async () => {
    mockFetch.mockResolvedValueOnce(
      okResponse({ property_id: 'p', message: '', property: minDetail }),
    );
    await createProperty({ name: 'X', model_name: 'gpt' });
    const url = (mockFetch.mock.calls[0] as [string])[0];
    expect(url).toBe(`${BASE}/api/v1/properties/`);
  });
});

describe('uploadImage', () => {
  const minImage = {
    id: 'img-1',
    file_path: '/storage/img-1.jpg',
    room_type: 'kitchen',
    amenities: [],
    alt_text: null,
    caption: null,
    is_primary: false,
    display_order: 0,
  };

  it('POSTs multipart FormData (no Content-Type header)', async () => {
    mockFetch.mockResolvedValueOnce(
      okResponse({ property_id: 'p-1', image: minImage }),
    );
    const file = new File(['data'], 'photo.jpg', { type: 'image/jpeg' });
    await uploadImage({ propertyId: 'p-1', file, modelName: 'gpt-4o-mini' });

    const init = (mockFetch.mock.calls[0] as [string, RequestInit])[1];
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    // Crucial: must NOT set Content-Type — the browser fills boundary itself.
    expect(init.headers).toBeUndefined();
  });

  it('forwards an AbortSignal when provided', async () => {
    mockFetch.mockResolvedValueOnce(
      okResponse({ property_id: 'p-1', image: minImage }),
    );
    const ctrl = new AbortController();
    const file = new File(['x'], 'a.jpg', { type: 'image/jpeg' });
    await uploadImage({ propertyId: 'p-1', file, modelName: 'gpt', signal: ctrl.signal });

    const init = (mockFetch.mock.calls[0] as [string, RequestInit])[1];
    expect(init.signal).toBe(ctrl.signal);
  });
});

describe('patchImage', () => {
  const minImage = {
    id: 'img-1',
    file_path: '/storage/img-1.jpg',
    room_type: null,
    amenities: [],
    alt_text: null,
    caption: null,
    is_primary: true,
    display_order: 0,
  };

  it('sends PATCH with JSON body', async () => {
    mockFetch.mockResolvedValueOnce(okResponse(minImage));
    await patchImage('p-1', 'img-1', { is_primary: true, alt_text: 'kitchen' });

    const init = (mockFetch.mock.calls[0] as [string, RequestInit])[1];
    expect(init.method).toBe('PATCH');
    expect(init.body).toBe(JSON.stringify({ is_primary: true, alt_text: 'kitchen' }));
  });
});

describe('describeProperty', () => {
  it('POSTs to /describe and parses { description }', async () => {
    mockFetch.mockResolvedValueOnce(okResponse({ description: 'Lovely place.' }));
    const res = await describeProperty('p-1', {
      amenities: [{ amenity_name: 'WiFi', room_type: 'living_room', is_present: true }],
      model_name: 'gpt',
    });
    expect(res.description).toBe('Lovely place.');
    const url = (mockFetch.mock.calls[0] as [string])[0];
    expect(url).toBe(`${BASE}/api/v1/properties/p-1/describe`);
  });
});

describe('patchProperty', () => {
  const minDetail = { ...minSummary, images: [] };

  it('accepts description in patch body', async () => {
    mockFetch.mockResolvedValueOnce(okResponse({ ...minDetail, description: 'Updated' }));
    const res = await patchProperty('p-1', { description: 'Updated' });
    expect(res.description).toBe('Updated');

    const init = (mockFetch.mock.calls[0] as [string, RequestInit])[1];
    expect(init.method).toBe('PATCH');
    expect(init.body).toBe(JSON.stringify({ description: 'Updated' }));
  });
});

// ── applyClientFilters ───────────────────────────────────────────────────────

describe('applyClientFilters', () => {
  const rentApt = {
    ...minSummary,
    id: 'rent-1',
    listing_type: 'rent' as const,
    price: 1200,
    currency: 'EUR',
    num_bedrooms: 2,
  };
  const saleHouse = {
    ...minSummary,
    id: 'sale-1',
    listing_type: 'sale' as const,
    price: 350000,
    currency: 'EUR',
    num_bedrooms: 4,
  };
  const noMeta = { ...minSummary, id: 'meta-less' };

  it('returns input unchanged when filters are empty', () => {
    const out = applyClientFilters([rentApt, saleHouse, noMeta], {});
    expect(out).toHaveLength(3);
  });

  it('filters by listing_type', () => {
    const out = applyClientFilters([rentApt, saleHouse], { listing_type: 'rent' });
    expect(out.map((p) => p.id)).toEqual(['rent-1']);
  });

  it('drops properties with no listing_type when filter is set', () => {
    const out = applyClientFilters([rentApt, noMeta], { listing_type: 'rent' });
    expect(out.map((p) => p.id)).toEqual(['rent-1']);
  });

  it('filters by num_bedrooms with equality', () => {
    const out = applyClientFilters([rentApt, saleHouse], { num_bedrooms: 2 });
    expect(out.map((p) => p.id)).toEqual(['rent-1']);
  });

  it('filters by price_max (inclusive)', () => {
    const out = applyClientFilters([rentApt, saleHouse], { price_max: 1200 });
    expect(out.map((p) => p.id)).toEqual(['rent-1']);
  });

  it('matches currency when both filter and property have it', () => {
    const out = applyClientFilters(
      [rentApt, { ...rentApt, id: 'rent-2', currency: 'USD' }],
      { price_max: 2000, currency: 'EUR' },
    );
    expect(out.map((p) => p.id)).toEqual(['rent-1']);
  });

  it('does not filter on currency when filter has none', () => {
    const out = applyClientFilters(
      [rentApt, { ...rentApt, id: 'rent-2', currency: 'USD' }],
      { price_max: 2000 },
    );
    expect(out.map((p) => p.id).sort()).toEqual(['rent-1', 'rent-2']);
  });

  it('combines all dimensions (AND)', () => {
    const out = applyClientFilters([rentApt, saleHouse, noMeta], {
      listing_type: 'rent',
      num_bedrooms: 2,
      price_max: 1500,
      currency: 'EUR',
    });
    expect(out.map((p) => p.id)).toEqual(['rent-1']);
  });
});

// ── searchProperties ─────────────────────────────────────────────────────────

describe('searchProperties', () => {
  it('hits /search?amenities=... when amenities are provided', async () => {
    mockFetch.mockResolvedValueOnce(okResponse([minSummary]));
    await searchProperties({ amenities: ['pool', 'wifi'], filters: {} });
    const url = (mockFetch.mock.calls[0] as [string])[0];
    expect(url).toBe(`${BASE}/api/v1/properties/search?amenities=pool%2Cwifi`);
  });

  it('falls back to listProperties when amenities is empty', async () => {
    mockFetch.mockResolvedValueOnce(okResponse([minSummary]));
    await searchProperties({ amenities: [], filters: {} });
    const url = (mockFetch.mock.calls[0] as [string])[0];
    expect(url).toContain('/api/v1/properties/');
    expect(url).not.toContain('/search');
  });

  it('applies client-side filters to the server result', async () => {
    const rent = { ...minSummary, id: 'r', listing_type: 'rent' as const };
    const sale = { ...minSummary, id: 's', listing_type: 'sale' as const };
    mockFetch.mockResolvedValueOnce(okResponse([rent, sale]));
    const out = await searchProperties({
      amenities: ['pool'],
      filters: { listing_type: 'rent' },
    });
    expect(out.map((p) => p.id)).toEqual(['r']);
  });

  it('throws ApiError when the upstream /search returns non-2xx', async () => {
    mockFetch.mockResolvedValueOnce(errResponse(400));
    await expect(
      searchProperties({ amenities: ['x'], filters: {} }),
    ).rejects.toThrow(ApiError);
  });
});
