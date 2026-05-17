import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, getImageUrl, getProperty, listProperties } from '../../lib/api';

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
