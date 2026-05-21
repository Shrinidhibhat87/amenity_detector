/**
 * /llms.txt — machine-readable site description for LLM agents and AI crawlers.
 *
 * Follows the emerging llms.txt spec (https://llmstxt.org) so agents
 * understand the purpose and structure of this site without scraping HTML.
 *
 * The body is built in lib/llms-txt-builder.ts so the format is
 * unit-testable without a real backend.
 */

import { NextResponse } from 'next/server';
import { listProperties } from '@/lib/api';
import { buildLlmsTxt } from '@/lib/llms-txt-builder';

const SITE_URL = process.env['NEXT_PUBLIC_SITE_URL'] ?? 'http://localhost:3000';

export async function GET(): Promise<NextResponse> {
  const properties = await listProperties({ limit: 100 });
  const text = buildLlmsTxt(properties, SITE_URL);

  return new NextResponse(text, {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
}
