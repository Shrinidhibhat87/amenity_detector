/**
 * /api/parse — natural-language query parser endpoint.
 *
 * Accepts a POST body of `{ query: string }` and returns the structured
 * ParsedQuery defined in `lib/nl-parser.ts`. Wrapping the parser in a route
 * handler instead of calling it from the browser means Phase 11 can swap the
 * implementation for an LLM call (LiteLLM proxy per project memory) without
 * touching any caller — the network shape stays the same and API keys stay
 * server-side.
 *
 * GET is intentionally unsupported: queries can be long enough to bump up
 * against URL length limits, and the body would be visible in logs / browser
 * history. Stick to POST.
 */

import { NextResponse } from 'next/server';
import { z } from 'zod';
import { parseQuery, ParsedQuerySchema } from '@/lib/nl-parser';

const RequestSchema = z.object({
  query: z.string().max(500),
});

export async function POST(request: Request): Promise<NextResponse> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: 'Request body must be valid JSON.' },
      { status: 400 },
    );
  }

  const parsed = RequestSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: 'Invalid request body.', issues: parsed.error.issues },
      { status: 400 },
    );
  }

  const result = parseQuery(parsed.data.query);
  // Round-trip through the schema so a future change to the parser cannot
  // emit a shape that violates the published contract.
  return NextResponse.json(ParsedQuerySchema.parse(result));
}
