import Link from 'next/link';
import { listProperties } from '@/lib/api';
import { PropertyCard } from '@/components/property-card';
import type { PropertySummary } from '@/lib/schemas';

// Skip build-time prerender — the API isn't reachable during `docker build`.
// Even though the try/catch below handles fetch failures gracefully, marking
// this dynamic also keeps the "featured" section fresh on every visit.
export const dynamic = 'force-dynamic';

export default async function Home() {
  // Fetch the three most recent listings for the featured section.
  // If the API is down during dev/build, degrade gracefully to empty list.
  let featured: PropertySummary[] = [];
  try {
    featured = await listProperties({ limit: 3 });
  } catch {
    // API unreachable — show the page without featured listings.
  }

  return (
    <>
      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <section className="relative overflow-hidden border-b border-border">
        {/* Decorative warm grid */}
        <div
          aria-hidden
          className="absolute inset-0 opacity-[0.035] pointer-events-none"
          style={{
            backgroundImage:
              'repeating-linear-gradient(0deg, var(--color-ink) 0, var(--color-ink) 1px, transparent 1px, transparent 60px), repeating-linear-gradient(90deg, var(--color-ink) 0, var(--color-ink) 1px, transparent 1px, transparent 60px)',
          }}
        />

        <div className="relative max-w-7xl mx-auto px-6 py-20 md:py-32">
          <div className="max-w-3xl">
            <p className="font-mono text-xs uppercase tracking-[0.2em] text-ink-muted mb-6">
              Amenity Detector · AI-powered listings
            </p>
            <h1 className="font-display text-5xl sm:text-6xl md:text-7xl text-ink leading-[1.05]">
              A warmer way to list, detect, and discover homes.
            </h1>
            <p className="mt-6 text-ink-soft text-lg leading-relaxed max-w-xl">
              Upload photos. The vision model detects amenities room by room. You review and
              publish. Buyers find the listing through natural-language search.
            </p>
            <div className="mt-10 flex flex-col sm:flex-row gap-3">
              <Link
                href="/detect/config"
                className="inline-flex items-center justify-center rounded-full bg-ink text-bg px-7 py-3.5 text-sm font-medium hover:-translate-y-0.5 transition-transform"
              >
                Upload &amp; detect
              </Link>
              <Link
                href="/browse"
                className="inline-flex items-center justify-center rounded-full border border-border text-ink px-7 py-3.5 text-sm font-medium hover:border-border-strong transition-colors"
              >
                Browse properties
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ── How it works ─────────────────────────────────────────────────── */}
      <section className="border-b border-border bg-surface">
        <div className="max-w-7xl mx-auto px-6 py-16 md:py-20">
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted mb-10">
            How it works
          </p>
          <div className="grid sm:grid-cols-3 gap-8">
            {[
              {
                step: '01',
                title: 'Upload photos',
                body: 'Add photos for each room. No rigid category selection — the model figures out what it sees.',
              },
              {
                step: '02',
                title: 'AI detects amenities',
                body: 'A vision-language model annotates each photo: WiFi, dishwasher, balcony, parking, and dozens more.',
              },
              {
                step: '03',
                title: 'Publish & get found',
                body: 'Review the results, generate a listing description, publish when it is ready, and let buyers search by the amenities they need.',
              },
            ].map(({ step, title, body }) => (
              <div key={step} className="space-y-3">
                <p className="font-mono text-[11px] uppercase tracking-widest text-accent">
                  {step}
                </p>
                <h3 className="text-lg font-medium text-ink">{title}</h3>
                <p className="text-sm text-ink-soft leading-relaxed">{body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Featured listings ─────────────────────────────────────────────── */}
      {featured.length > 0 && (
        <section className="max-w-7xl mx-auto px-6 py-16 md:py-20">
          <div className="flex items-end justify-between mb-8">
            <div>
              <p className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted mb-2">
                Latest
              </p>
              <h2 className="font-display text-3xl text-ink">Featured listings</h2>
            </div>
            <Link
              href="/browse"
              className="font-mono text-xs uppercase tracking-widest text-ink-muted hover:text-ink transition-colors hidden sm:block"
            >
              View all →
            </Link>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {featured.map((p) => (
              <PropertyCard key={p.id} property={p} />
            ))}
          </div>
          <div className="mt-8 sm:hidden">
            <Link
              href="/browse"
              className="font-mono text-xs uppercase tracking-widest text-ink-muted hover:text-ink transition-colors"
            >
              View all →
            </Link>
          </div>
        </section>
      )}

      {/* ── Footer strip ─────────────────────────────────────────────────── */}
      <footer className="mt-auto border-t border-border bg-surface">
        <div className="max-w-7xl mx-auto px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="font-mono text-xs text-ink-muted">
            Amenity Detector · Browse + SEO
          </p>
          <div className="flex gap-6 font-mono text-xs text-ink-muted">
            <a
              href="/sitemap.xml"
              className="hover:text-ink transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              Sitemap
            </a>
            <a
              href="/llms.txt"
              className="hover:text-ink transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              llms.txt
            </a>
            <a
              href="/api/feed.jsonl"
              className="hover:text-ink transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              JSONL feed
            </a>
          </div>
        </div>
      </footer>
    </>
  );
}
