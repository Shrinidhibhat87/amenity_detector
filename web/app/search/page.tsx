import type { Metadata } from 'next';
import Link from 'next/link';
import { ApiError, searchProperties, type ClientFilters } from '@/lib/api';
import { parseQuery } from '@/lib/nl-parser';
import { PropertyCard } from '@/components/property-card';
import { SearchFilterChips } from '@/components/search-filter-chips';

// Next 15: searchParams is a Promise in async Server Components.
type Props = { searchParams: Promise<Record<string, string | string[] | undefined>> };

// Skip build-time prerender — the API isn't reachable during `docker build`,
// and search results depend on user input anyway.
export const dynamic = 'force-dynamic';

export const metadata: Metadata = {
  title: 'Search — Amenity Detector',
  description: 'Natural-language property search.',
};

function pickString(value: string | string[] | undefined): string {
  if (Array.isArray(value)) return value[0] ?? '';
  return value ?? '';
}

/**
 * URL overrides take precedence over the parser. An empty string override
 * (`listing_type=`) means "user explicitly cleared this filter", so the
 * parser's value is discarded for that dimension and we leave it unset.
 * A missing param means "fall back to whatever the parser detected".
 */
function applyOverrides(
  parsed: ClientFilters,
  params: Record<string, string | string[] | undefined>,
): ClientFilters {
  const out: ClientFilters = {};

  const lt = params.listing_type;
  if (lt === undefined) {
    if (parsed.listing_type != null) out.listing_type = parsed.listing_type;
  } else if (lt === 'rent' || lt === 'sale') {
    out.listing_type = lt;
  }

  const bd = pickString(params.num_bedrooms);
  if (bd === '' && params.num_bedrooms === undefined) {
    if (parsed.num_bedrooms != null) out.num_bedrooms = parsed.num_bedrooms;
  } else if (bd !== '') {
    const n = Number(bd);
    if (Number.isFinite(n) && n > 0) out.num_bedrooms = n;
  }

  const pm = pickString(params.price_max);
  if (pm === '' && params.price_max === undefined) {
    if (parsed.price_max != null) out.price_max = parsed.price_max;
    if (parsed.currency != null) out.currency = parsed.currency;
  } else if (pm !== '') {
    const n = Number(pm);
    if (Number.isFinite(n) && n > 0) out.price_max = n;
    const curr = pickString(params.currency).toUpperCase();
    if (curr.length === 3) out.currency = curr;
  }

  return out;
}

export default async function SearchPage({ searchParams }: Props) {
  const params = await searchParams;
  const q = pickString(params.q).trim();

  // No query → render the empty landing state, no API call.
  if (q === '') {
    return (
      <main className="flex-1 px-6 py-12 max-w-7xl mx-auto w-full">
        <header className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <p className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted">
              Search
            </p>
            <Link
              href="/"
              className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted hover:text-ink transition-colors"
            >
              ← Home
            </Link>
          </div>
          <h1 className="font-display text-3xl text-ink">Find a place in plain English</h1>
          <p className="mt-2 text-ink-soft max-w-xl">
            Try something like{' '}
            <em className="text-ink">
              &ldquo;3 bedroom rent under 1500 EUR with fireplace in living room&rdquo;
            </em>
            . Use the search bar on{' '}
            <Link href="/browse" className="underline hover:text-ink">
              Browse
            </Link>{' '}
            to begin.
          </p>
        </header>
      </main>
    );
  }

  // Import the parser directly instead of fetching /api/parse: this is a
  // Server Component, calling our own route handler would be a needless
  // round-trip. Client-side surfaces (typeahead, etc.) still use /api/parse.
  const parsed = parseQuery(q);
  const effective = applyOverrides(parsed.filters, params);

  let results: Awaited<ReturnType<typeof searchProperties>> = [];
  let error: string | null = null;
  try {
    results = await searchProperties({
      amenities: parsed.amenities,
      filters: effective,
    });
  } catch (err) {
    error =
      err instanceof ApiError
        ? `Search failed: ${err.status} ${err.message}`
        : err instanceof Error
          ? err.message
          : 'Unknown error.';
  }

  return (
    <main className="flex-1 px-6 py-12 max-w-7xl mx-auto w-full">
      <header className="mb-8">
        <div className="flex items-center justify-between mb-2">
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted">
            Search
          </p>
          <Link
            href="/"
            className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted hover:text-ink transition-colors"
          >
            ← Home
          </Link>
        </div>
        <h1 className="font-display text-3xl text-ink">Results for &ldquo;{q}&rdquo;</h1>

        <SearchFilterChips
          filters={{
            listing_type: effective.listing_type,
            num_bedrooms: effective.num_bedrooms,
            price_max: effective.price_max,
            currency: effective.currency,
            amenities: parsed.amenities,
            room_amenities: parsed.filters.room_amenities ?? [],
          }}
        />
      </header>

      {error != null && (
        <p className="text-warn text-sm font-mono mb-6">{error}</p>
      )}

      {error == null && results.length === 0 && (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <p className="font-mono text-xs uppercase tracking-widest text-ink-muted mb-4">
            No matches
          </p>
          <p className="text-ink-soft max-w-sm">
            Try fewer terms or browse all listings on{' '}
            <Link href="/browse" className="underline hover:text-ink">
              Browse
            </Link>
            .
          </p>
        </div>
      )}

      {results.length > 0 && (
        <>
          <p className="mb-4 text-sm text-ink-soft">
            {results.length} match{results.length !== 1 ? 'es' : ''}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {results.map((p) => (
              <PropertyCard key={p.id} property={p} />
            ))}
          </div>
        </>
      )}
    </main>
  );
}
