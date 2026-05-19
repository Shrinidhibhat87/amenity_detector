import type { Metadata } from 'next';
import Link from 'next/link';
import { ApiError, searchProperties } from '@/lib/api';
import { parseQuery } from '@/lib/nl-parser';
import { PropertyCard } from '@/components/property-card';
import { Chip } from '@/components/ui';

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

  let results: Awaited<ReturnType<typeof searchProperties>> = [];
  let error: string | null = null;
  try {
    results = await searchProperties({
      amenities: parsed.amenities,
      filters: parsed.filters,
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

        {/* Read-only parsed-filter strip. Editable controls land in commit 5. */}
        <div className="mt-4 flex flex-wrap gap-2">
          {parsed.filters.listing_type != null && (
            <Chip tone="accent" active>
              {parsed.filters.listing_type === 'rent' ? 'For rent' : 'For sale'}
            </Chip>
          )}
          {parsed.filters.num_bedrooms != null && (
            <Chip tone="accent" active>
              {parsed.filters.num_bedrooms} bed
            </Chip>
          )}
          {parsed.filters.price_max != null && (
            <Chip tone="accent" active>
              ≤ {parsed.filters.price_max.toLocaleString()}{' '}
              {parsed.filters.currency ?? ''}
            </Chip>
          )}
          {parsed.amenities.map((a) => (
            <Chip key={a} active>
              {a}
            </Chip>
          ))}
          {(parsed.filters.room_amenities ?? []).map((ra) => (
            <Chip key={`${ra.room}-${ra.amenity}`} tone="ok" active>
              {ra.amenity} in {ra.room}
            </Chip>
          ))}
        </div>
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
