import type { Page, Route } from '@playwright/test';

/**
 * A stub of the FastAPI backend, installed per test with `page.route`.
 *
 * It keeps just enough server state to make the lifecycle observable: the
 * status a property carries and whether publish was ever called. Tests assert
 * against `calls` to prove, for example, that abandoning a draft publishes
 * nothing.
 */
export interface MockApi {
  calls: { publish: number; create: number; upload: number };
  status: string;
}

const PROPERTY_ID = '11111111-2222-4333-8444-555555555555';

function propertyBody(status: string, description: string | null) {
  return {
    id: PROPERTY_ID,
    name: 'E2E Apartment',
    status,
    description,
    model_used: 'openai/gpt-4o-mini',
    extra_info: null,
    created_at: '2026-08-09T10:00:00Z',
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
    postal_code: '60311',
    country_code: 'DE',
    latitude: null,
    longitude: null,
    owner_email: null,
    images: [],
    locality_insight: null,
  };
}

function imageBody(index: number) {
  return {
    id: `img-${index}`,
    file_path: `${PROPERTY_ID}/photo-${index}.jpg`,
    room_type: 'kitchen',
    amenities: [
      {
        id: `am-${index}-1`,
        amenity_name: 'refrigerator',
        room_type: 'kitchen',
        is_present: true,
        confidence: 0.9,
      },
      {
        id: `am-${index}-2`,
        amenity_name: 'oven',
        room_type: 'kitchen',
        is_present: true,
        confidence: 0.8,
      },
    ],
    alt_text: 'A kitchen',
    caption: null,
    is_primary: false,
    display_order: 0,
  };
}

const LOCALITY_STREAM = [
  'data: {"type":"category","category":"supermarket","done":1,"total":1,"count":3}\n\n',
  `data: {"type":"result","data":${JSON.stringify({
    display_name: 'Frankfurt',
    latitude: 50.11,
    longitude: 8.68,
    radius_m: 3000,
    blurb: 'Central, well connected, three supermarkets close by.',
    category_counts: { supermarket: 3 },
    pois: [],
    attribution: '© OpenStreetMap contributors',
  })}}\n\n`,
].join('');

export async function mockApi(page: Page): Promise<MockApi> {
  const api: MockApi = {
    calls: { publish: 0, create: 0, upload: 0 },
    status: 'draft',
  };
  let description: string | null = null;

  const json = (route: Route, body: unknown, status = 200) =>
    route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

  await page.route('**/api/v1/**', async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();

    if (path === '/api/v1/models/') {
      return json(route, [
        { name: 'openai/gpt-4o-mini', available: true, description: 'Fast multimodal model' },
      ]);
    }

    if (path === '/api/v1/properties/' && method === 'POST') {
      api.calls.create += 1;
      api.status = 'draft';
      return json(
        route,
        {
          property_id: PROPERTY_ID,
          message: 'created',
          property: propertyBody(api.status, null),
        },
        201,
      );
    }

    if (path.endsWith('/images') && method === 'POST') {
      api.calls.upload += 1;
      api.status = 'processing';
      return json(route, { property_id: PROPERTY_ID, image: imageBody(api.calls.upload) }, 201);
    }

    if (path.endsWith('/locality/stream')) {
      return route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: LOCALITY_STREAM,
      });
    }

    if (path.endsWith('/describe') && method === 'POST') {
      api.status = 'ready_for_review';
      return json(route, {
        description: 'A bright kitchen with a fridge and an oven.',
      });
    }

    if (path.endsWith('/publish') && method === 'POST') {
      api.calls.publish += 1;
      api.status = 'published';
      return json(route, propertyBody(api.status, description));
    }

    if (path === `/api/v1/properties/${PROPERTY_ID}` && method === 'PATCH') {
      const body = route.request().postDataJSON() as { description?: string };
      if (body.description != null) {
        description = body.description;
        api.status = 'completed';
      }
      return json(route, propertyBody(api.status, description));
    }

    if (path.startsWith('/api/v1/properties/') && method === 'GET') {
      return json(route, propertyBody(api.status, description));
    }

    return json(route, {});
  });

  return api;
}

export { PROPERTY_ID };
