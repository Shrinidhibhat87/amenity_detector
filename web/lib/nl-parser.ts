/**
 * Natural-language search parser — mock implementation for P-D.
 *
 * Turns a free-text query like "3BHK rent under 1500 EUR with fireplace in
 * living room" into a structured ParsedQuery the search page can pass to the
 * existing /api/v1/properties/search endpoint (amenities AND-filter) plus a
 * set of filters the client applies on the returned summaries.
 *
 * This is a regex + heuristic mock so the frontend can ship before Phase 11
 * lands. The Zod schema below is the contract Phase 11 must honour when the
 * real LLM-backed parser replaces this file — keep the output shape stable.
 *
 * Strategy:
 *   1. lowercase + trim a working copy of the input
 *   2. extract bedrooms (digits, "3BHK", small word numerals up to 5)
 *   3. extract listing_type via verb hints (rent / sale / buy)
 *   4. extract price ceiling + optional currency (under / below / less than / max)
 *   5. extract "<amenity> in <room>" tuples first, then any remaining bare
 *      amenities from a small known vocabulary
 *
 * The vocabularies are intentionally short — Phase 11 swaps this for an LLM
 * that does not need handwritten word lists.
 */

import { z } from 'zod';

// ── Vocabularies ──────────────────────────────────────────────────────────────

const AMENITY_VOCAB: readonly string[] = [
  'pool',
  'wifi',
  'gym',
  'parking',
  'balcony',
  'fireplace',
  'garden',
  'terrace',
  'kitchen',
  'washer',
  'dryer',
  'dishwasher',
  'oven',
  'refrigerator',
  'microwave',
  'heating',
  'air conditioning',
  'elevator',
  'doorman',
];

const ROOM_VOCAB: readonly string[] = [
  'living room',
  'bedroom',
  'kitchen',
  'bathroom',
  'dining room',
  'office',
  'balcony',
  'garage',
  'basement',
];

const WORD_NUMERALS: Record<string, number> = {
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
};

const CURRENCY_SYMBOLS: Record<string, string> = {
  '€': 'EUR',
  $: 'USD',
  '£': 'GBP',
};

// ── Zod schema — also the contract Phase 11 must keep stable ─────────────────

export const RoomAmenitySchema = z.object({
  room: z.string(),
  amenity: z.string(),
});

export const ParsedFiltersSchema = z.object({
  listing_type: z.enum(['rent', 'sale']).optional(),
  price_max: z.number().optional(),
  currency: z.string().optional(),
  num_bedrooms: z.number().int().optional(),
  room_amenities: z.array(RoomAmenitySchema).optional(),
});

export const ParsedQuerySchema = z.object({
  query: z.string(),
  amenities: z.array(z.string()),
  filters: ParsedFiltersSchema,
});

export type RoomAmenity = z.infer<typeof RoomAmenitySchema>;
export type ParsedFilters = z.infer<typeof ParsedFiltersSchema>;
export type ParsedQuery = z.infer<typeof ParsedQuerySchema>;

// ── Parser ───────────────────────────────────────────────────────────────────

export function parseQuery(input: string): ParsedQuery {
  const normalised = input.toLowerCase();

  const filters: ParsedFilters = {};
  const amenities = new Set<string>();

  // 1. Bedrooms — digit / NBHK / spelled-out numerals 1..5.
  const bedroomMatch =
    normalised.match(/(\d+)\s*bhk/) ??
    normalised.match(/(\d+)[-\s]?(?:bedroom|bedrooms|bed|beds)\b/) ??
    matchWordNumeralBedrooms(normalised);
  if (bedroomMatch != null) {
    const n = Number(bedroomMatch[1]);
    if (Number.isFinite(n) && n > 0) filters.num_bedrooms = n;
  }

  // 2. Listing type — rent vs sale (buy → sale).
  if (/\brent(?:al|als|ing)?\b/.test(normalised)) {
    filters.listing_type = 'rent';
  } else if (/\b(?:sale|buy|purchase|to\s+buy)\b/.test(normalised)) {
    filters.listing_type = 'sale';
  }

  // 3. Price ceiling — accepts "under/below/less than/max <amount> <currency?>"
  //    or "<symbol><amount>" anywhere in the query.
  const priceCeil = extractPriceCeiling(normalised);
  if (priceCeil != null) {
    filters.price_max = priceCeil.amount;
    if (priceCeil.currency != null) filters.currency = priceCeil.currency;
  }

  // 4. "<amenity> in <room>" tuples — must run before the bare amenity pass
  //    so the room word is not consumed by the bare scan.
  const roomTuples: RoomAmenity[] = [];
  for (const amenity of AMENITY_VOCAB) {
    for (const room of ROOM_VOCAB) {
      const pattern = new RegExp(
        `\\b${escapeRegex(amenity)}\\s+in\\s+(?:the\\s+)?${escapeRegex(room)}\\b`,
      );
      if (pattern.test(normalised)) {
        roomTuples.push({ room, amenity });
        amenities.add(amenity);
      }
    }
  }
  if (roomTuples.length > 0) filters.room_amenities = roomTuples;

  // 5. Bare amenity scan from the known vocabulary.
  for (const amenity of AMENITY_VOCAB) {
    const pattern = new RegExp(`\\b${escapeRegex(amenity)}\\b`);
    if (pattern.test(normalised)) amenities.add(amenity);
  }

  return {
    query: input,
    amenities: Array.from(amenities),
    filters,
  };
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function matchWordNumeralBedrooms(input: string): RegExpMatchArray | null {
  const re = /\b(one|two|three|four|five)\s+(?:bedroom|bedrooms|bed|beds)\b/;
  const m = input.match(re);
  if (m == null) return null;
  const numeral = m[1];
  if (numeral == null) return null;
  const n = WORD_NUMERALS[numeral];
  if (n == null) return null;
  // Reshape so the caller can read m[1] as a number string.
  return Object.assign([m[0], String(n)] as unknown as RegExpMatchArray, {
    index: m.index,
    input: m.input,
    groups: undefined,
  });
}

function extractPriceCeiling(
  input: string,
): { amount: number; currency: string | undefined } | null {
  // "<keyword> <symbol?><amount> <currency?>"
  const keywordPattern =
    /\b(?:under|below|less\s+than|max(?:imum)?|up\s+to)\s+([€$£])?\s*([\d,]+)\s*([a-z]{3})?/;
  const m = input.match(keywordPattern);
  if (m != null) {
    const amount = parseAmount(m[2]);
    if (amount == null) return null;
    const currency = resolveCurrency(m[1], m[3]);
    return { amount, currency };
  }

  // Bare "<symbol><amount>" fallback so "$2000" still parses.
  const bareSymbol = input.match(/([€$£])\s*([\d,]+)/);
  if (bareSymbol != null) {
    const amount = parseAmount(bareSymbol[2]);
    if (amount == null) return null;
    return { amount, currency: resolveCurrency(bareSymbol[1], undefined) };
  }

  return null;
}

function parseAmount(raw: string | undefined): number | null {
  if (raw == null) return null;
  const n = Number(raw.replace(/,/g, ''));
  return Number.isFinite(n) ? n : null;
}

function resolveCurrency(
  symbol: string | undefined,
  iso: string | undefined,
): string | undefined {
  if (iso != null && iso.length === 3) return iso.toUpperCase();
  if (symbol != null && symbol in CURRENCY_SYMBOLS) return CURRENCY_SYMBOLS[symbol];
  return undefined;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
