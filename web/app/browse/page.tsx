import type { Metadata } from 'next';
import Link from 'next/link';
import { listProperties } from '@/lib/api';
import { PropertyCard } from '@/components/property-card';

// Skip build-time prerender — the API isn't reachable during `docker build`.
// This page renders on every request (SSR). Acceptable: the listing index
// must reflect current state, and the API call is fast.
export const dynamic = 'force-dynamic';

export const metadata: Metadata = {
  title: 'Browse Properties — Amenity Detector',
  description: 'Browse all available property listings with detected amenities.',
};

export default async function BrowsePage() {
  const properties = await listProperties({ limit: 48 });

  return (
    <main className="flex-1 px-6 py-12 max-w-7xl mx-auto w-full">
      <header className="mb-8">
        <div className="flex items-center justify-between mb-2">
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted">
            Browse
          </p>
          <Link
            href="/"
            className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted hover:text-ink transition-colors"
          >
            ← Home
          </Link>
        </div>
        <h1 className="font-display text-3xl text-ink">Properties</h1>
        {properties.length > 0 && (
          <p className="mt-1 text-sm text-ink-soft">
            {properties.length} listing{properties.length !== 1 ? 's' : ''}
          </p>
        )}
      </header>

      {properties.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <p className="font-mono text-xs uppercase tracking-widest text-ink-muted mb-4">
            No listings yet
          </p>
          <p className="text-ink-soft max-w-sm">
            Upload some property photos to get started. The detector will analyse them and list
            amenities automatically.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {properties.map((p) => (
            <PropertyCard key={p.id} property={p} />
          ))}
        </div>
      )}
    </main>
  );
}
