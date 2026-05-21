import Link from 'next/link';

export default function PropertyNotFound() {
  return (
    <main className="flex-1 flex flex-col items-center justify-center px-6 py-24 text-center">
      <p className="font-mono text-xs uppercase tracking-widest text-ink-muted mb-4">
        Property not found
      </p>
      <p className="text-ink-soft max-w-sm mb-8">
        This listing may have been removed or the link is incorrect.
      </p>
      <Link
        href="/browse"
        className="font-mono text-xs uppercase tracking-widest px-4 py-2 rounded-full border border-border text-ink hover:border-border-strong transition-colors"
      >
        ← Back to browse
      </Link>
    </main>
  );
}
