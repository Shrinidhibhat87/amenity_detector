import { describe, expect, it } from 'vitest';
import { parseQuery } from '../../lib/nl-parser';

describe('parseQuery — empty / trivial input', () => {
  it('returns the original query with empty filters when the input is empty', () => {
    expect(parseQuery('')).toEqual({
      query: '',
      amenities: [],
      filters: {},
    });
  });

  it('trims and lowercases the stored query while preserving structure', () => {
    const out = parseQuery('  POOL  ');
    expect(out.query).toBe('  POOL  ');
    expect(out.amenities).toEqual(['pool']);
    expect(out.filters).toEqual({});
  });
});

describe('parseQuery — bedrooms', () => {
  it.each([
    ['3 bedroom apartment', 3],
    ['2-bedroom flat', 2],
    ['four bedroom house', 4],
    ['3BHK', 3],
    ['1 bed', 1],
    ['studio', undefined],
  ])('extracts bedrooms from %s', (input, expected) => {
    expect(parseQuery(input).filters.num_bedrooms).toBe(expected);
  });
});

describe('parseQuery — listing type', () => {
  it('detects rent', () => {
    expect(parseQuery('apartment for rent').filters.listing_type).toBe('rent');
  });

  it('detects sale', () => {
    expect(parseQuery('house for sale').filters.listing_type).toBe('sale');
  });

  it('accepts "buy" as a synonym for sale', () => {
    expect(parseQuery('villas to buy').filters.listing_type).toBe('sale');
  });

  it('leaves it undefined when neither verb is present', () => {
    expect(parseQuery('3 bedroom apartment').filters.listing_type).toBeUndefined();
  });
});

describe('parseQuery — price ceiling', () => {
  it.each([
    ['under 1500 EUR', 1500, 'EUR'],
    ['below 200,000 USD', 200000, 'USD'],
    ['less than €1500', 1500, 'EUR'],
    ['under $2000', 2000, 'USD'],
    ['max 1500', 1500, undefined],
  ])('extracts price ceiling from %s', (input, price, currency) => {
    const out = parseQuery(input).filters;
    expect(out.price_max).toBe(price);
    expect(out.currency).toBe(currency);
  });

  it('returns no price when no ceiling phrasing is present', () => {
    expect(parseQuery('3 bedroom rent').filters.price_max).toBeUndefined();
  });
});

describe('parseQuery — amenities and room tuples', () => {
  it('extracts a single amenity from the known vocabulary', () => {
    expect(parseQuery('with a pool').amenities).toEqual(['pool']);
  });

  it('extracts multiple amenities joined by "and" / commas', () => {
    expect(parseQuery('pool, wifi and gym').amenities.sort()).toEqual(
      ['gym', 'pool', 'wifi'].sort(),
    );
  });

  it('detects "<amenity> in <room>" tuples', () => {
    const out = parseQuery('fireplace in living room');
    expect(out.amenities).toContain('fireplace');
    expect(out.filters.room_amenities).toEqual([
      { room: 'living room', amenity: 'fireplace' },
    ]);
  });

  it('does not double-count an amenity that also appears in a tuple', () => {
    const out = parseQuery('fireplace in living room');
    expect(out.amenities.filter((a) => a === 'fireplace')).toHaveLength(1);
  });

  it('ignores unknown words that look like nouns', () => {
    expect(parseQuery('purple elephant').amenities).toEqual([]);
  });
});

describe('parseQuery — combined Phase 11 example', () => {
  it('parses "3BHK rent under 1500 EUR with fireplace in living room"', () => {
    const out = parseQuery('3BHK rent under 1500 EUR with fireplace in living room');
    expect(out.filters.num_bedrooms).toBe(3);
    expect(out.filters.listing_type).toBe('rent');
    expect(out.filters.price_max).toBe(1500);
    expect(out.filters.currency).toBe('EUR');
    expect(out.amenities).toContain('fireplace');
    expect(out.filters.room_amenities).toEqual([
      { room: 'living room', amenity: 'fireplace' },
    ]);
  });
});
