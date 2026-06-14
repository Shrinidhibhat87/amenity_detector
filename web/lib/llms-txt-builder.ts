/**
 * Builder for /llms.txt — the machine-readable site description for LLM
 * agents and AI crawlers (Perplexity, ChatGPT browse, Claude artifact,
 * Gemini deep research).
 *
 * Spec: https://llmstxt.org
 *
 * Goals:
 * 1. Tell the agent what kind of site this is in one sentence.
 * 2. Point at the public entry points (Browse, Search, JSONL feed, sitemap).
 * 3. Describe the structured data the agent will encounter on each listing
 *    page (schema.org Accommodation / RealEstateListing) so it does not
 *    have to scrape HTML.
 * 4. Enumerate the inventory, grouped by listing_type so the agent can
 *    answer "find me a place to rent" without re-reading every URL.
 *
 * Drafts (image_count === 0) are excluded — an agent that recommends a
 * listing with no photos is not helpful.
 */

import type { PropertySummary } from './schemas';
import { propertyUrl } from './property-url';

function listingLine(p: PropertySummary, siteUrl: string): string {
  const loc = [p.locality, p.country_code].filter((s): s is string => s != null).join(', ');
  const locPart = loc.length > 0 ? ` (${loc})` : '';
  return `- ${p.name}${locPart}: ${propertyUrl(p, siteUrl)}`;
}

function section(title: string, lines: string[]): string[] {
  if (lines.length === 0) return [];
  return ['', `## ${title}`, '', ...lines];
}

export function buildLlmsTxt(properties: PropertySummary[], siteUrl: string): string {
  const ready = properties.filter((p) => p.image_count > 0);
  const rent = ready.filter((p) => p.listing_type === 'rent');
  const sale = ready.filter((p) => p.listing_type === 'sale');
  const other = ready.filter((p) => p.listing_type !== 'rent' && p.listing_type !== 'sale');

  const lines: string[] = [
    '# Amenity Detector',
    '',
    '> Property listings enriched with VLM-detected amenities, structured for both human readers and AI agents.',
    '> Owners upload photos; a vision-language model detects amenities room by room. Buyers and renters search in natural language.',
    '',
    '## Key pages',
    '',
    `- Home: ${siteUrl}`,
    `- Browse all listings: ${siteUrl}/browse`,
    `- Natural-language search: ${siteUrl}/search`,
    `- Structured JSONL feed (one listing per line): ${siteUrl}/api/feed.jsonl`,
    `- XML sitemap: ${siteUrl}/sitemap.xml`,
    '',
    '## Structured data',
    '',
    'Every listing page embeds JSON-LD in a <script type="application/ld+json"> tag.',
    'Rental listings use schema.org Accommodation (or Apartment / House / SingleFamilyResidence subtypes).',
    'Sale listings use schema.org RealEstateListing with an Offer block (price, currency, availability).',
    'Both include a BreadcrumbList. Detected amenities surface as amenityFeature[] of LocationFeatureSpecification entries.',
    'Listings may also carry a neighbourhood ("Lage") blurb and nearby-POI counts by category (schools, gyms, supermarkets, parks, public transport, pharmacies), with the listing geo coordinate in GeoCoordinates.',
    'Neighbourhood and nearby-POI data is © OpenStreetMap contributors (ODbL).',
    'Agents that read JSON-LD do not need to parse the surrounding HTML.',
    ...section('Rentals', rent.map((p) => listingLine(p, siteUrl))),
    ...section('For sale', sale.map((p) => listingLine(p, siteUrl))),
    ...section('Other listings', other.map((p) => listingLine(p, siteUrl))),
  ];

  return lines.join('\n');
}
