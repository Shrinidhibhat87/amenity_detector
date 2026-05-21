/**
 * /api/feed.jsonl — newline-delimited JSON feed of all listings.
 *
 * Each line is a self-contained JSON object representing one property.
 * This format is easy for LLM agents and data pipelines to consume
 * incrementally without loading a full JSON array into memory.
 */

import { NextResponse } from 'next/server';
import { listProperties } from '@/lib/api';

const SITE_URL = process.env['NEXT_PUBLIC_SITE_URL'] ?? 'http://localhost:3000';

export async function GET(): Promise<NextResponse> {
  const properties = await listProperties({ limit: 100 });

  const lines = properties.map((p) =>
    JSON.stringify({
      id: p.id,
      url: `${SITE_URL}/properties/${p.id}`,
      name: p.name,
      listing_type: p.listing_type,
      price: p.price,
      currency: p.currency,
      price_period: p.price_period,
      num_bedrooms: p.num_bedrooms,
      num_bathrooms: p.num_bathrooms,
      area_sqm: p.area_sqm,
      property_type: p.property_type,
      furnishing: p.furnishing,
      locality: p.locality,
      postal_code: p.postal_code,
      country_code: p.country_code,
      image_count: p.image_count,
      created_at: p.created_at,
    }),
  );

  return new NextResponse(lines.join('\n'), {
    headers: { 'Content-Type': 'application/jsonl; charset=utf-8' },
  });
}
