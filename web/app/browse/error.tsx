'use client';

export default function BrowseError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <main className="flex-1 flex flex-col items-center justify-center px-6 py-24 text-center">
      <p className="font-mono text-xs uppercase tracking-widest text-ink-muted mb-4">
        Could not load listings
      </p>
      <p className="text-ink-soft max-w-sm mb-8">
        {error.message.includes('fetch') || error.message.includes('ECONNREFUSED')
          ? 'The API server is not reachable. Make sure the backend is running.'
          : 'Something went wrong while fetching properties.'}
      </p>
      <button
        onClick={reset}
        className="font-mono text-xs uppercase tracking-widest px-4 py-2 rounded-full border border-border text-ink hover:border-border-strong transition-colors"
      >
        Try again
      </button>
    </main>
  );
}
