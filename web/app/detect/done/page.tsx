'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { ApiError, publishProperty } from '@/lib/api';
import { useWizardStore } from '@/lib/wizard-store';
import { Button } from '@/components/ui';

export default function DoneStepPage() {
  const router = useRouter();
  const state = useWizardStore((s) => s.state);
  const hasHydrated = useWizardStore((s) => s.hasHydrated);
  const markPublished = useWizardStore((s) => s.markPublished);
  const reset = useWizardStore((s) => s.reset);

  const [publishing, setPublishing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!hasHydrated) return;
    if (state.propertyId == null) router.replace('/detect/config');
  }, [hasHydrated, state.propertyId, router]);

  if (!hasHydrated || state.propertyId == null) return null;
  const propertyId = state.propertyId;

  async function publish() {
    setPublishing(true);
    setError(null);
    try {
      await publishProperty(propertyId);
      markPublished();
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.status === 409
            ? 'This listing is not complete enough to publish yet.'
            : `API ${err.status}: ${err.message}`
          : err instanceof Error
            ? err.message
            : 'Failed to publish the listing.',
      );
    } finally {
      setPublishing(false);
    }
  }

  if (state.published) {
    return (
      <div className="text-center space-y-6 py-10">
        <p className="font-mono text-xs uppercase tracking-widest text-accent">Published</p>
        <h1 className="font-display text-4xl text-ink">Your listing is live.</h1>
        <p className="text-ink-soft max-w-md mx-auto">
          Buyers and AI agents can now find it through the browse page, search, the sitemap
          and the JSONL feed.
        </p>

        <div className="flex items-center justify-center gap-3 pt-4">
          <Link
            href={`/properties/${propertyId}`}
            className="inline-flex items-center justify-center rounded-full bg-ink text-bg px-6 py-3 text-sm font-medium hover:-translate-y-0.5 transition-transform"
          >
            View the live listing →
          </Link>
          <Button
            variant="ghost"
            onClick={() => {
              reset();
              router.push('/detect/config');
            }}
          >
            Start another
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="text-center space-y-6 py-10">
      <p className="font-mono text-xs uppercase tracking-widest text-ink-muted">Draft saved</p>
      <h1 className="font-display text-4xl text-ink">Ready when you are.</h1>
      <p className="text-ink-soft max-w-md mx-auto">
        The listing and its detected amenities are saved, and nobody else can see them yet.
        Publishing puts it on the browse page, in search, the sitemap and the JSONL feed.
      </p>

      {error != null && <p className="text-warn text-sm font-mono">{error}</p>}

      <div className="flex items-center justify-center gap-3 pt-4">
        <Link
          href={`/properties/${propertyId}`}
          className="inline-flex items-center justify-center rounded-full border border-border text-ink px-6 py-3 text-sm font-medium hover:border-border-strong transition-colors"
        >
          Preview listing
        </Link>
        <Button onClick={() => void publish()} disabled={publishing}>
          {publishing ? 'Publishing…' : 'Publish →'}
        </Button>
      </div>
    </div>
  );
}
