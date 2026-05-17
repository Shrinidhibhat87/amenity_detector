import Link from "next/link";

export default function Home() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center px-6 py-24">
      <div className="max-w-2xl text-center">
        <p className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted">
          Amenity Detector
        </p>
        <h1 className="mt-4 font-display text-5xl sm:text-6xl text-ink">
          A warmer way to list, detect, and discover homes.
        </h1>
        <p className="mt-6 text-ink-soft text-lg leading-relaxed">
          Upload photos. We detect the amenities, room by room. You review.
          We write the listing. Buyers find you through natural-language search.
        </p>
        <div className="mt-10 flex flex-col sm:flex-row gap-3 justify-center">
          <Link
            href="/detect/config"
            className="inline-flex items-center justify-center rounded-full bg-ink text-bg px-6 py-3 text-sm font-medium hover:-translate-y-0.5 transition-transform"
          >
            Upload &amp; detect
          </Link>
          <Link
            href="/browse"
            className="inline-flex items-center justify-center rounded-full border border-border text-ink px-6 py-3 text-sm font-medium hover:border-border-strong transition-colors"
          >
            Browse properties
          </Link>
        </div>
        <p className="mt-12 font-mono text-[11px] uppercase tracking-widest text-ink-muted">
          Phase A · Foundation · TypeScript revamp in progress
        </p>
      </div>
    </main>
  );
}
