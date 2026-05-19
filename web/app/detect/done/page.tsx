'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useWizardStore } from '@/lib/wizard-store';
import { Button } from '@/components/ui';

export default function DoneStepPage() {
  const router = useRouter();
  const state = useWizardStore((s) => s.state);
  const hasHydrated = useWizardStore((s) => s.hasHydrated);
  const reset = useWizardStore((s) => s.reset);

  useEffect(() => {
    if (!hasHydrated) return;
    if (state.step !== 'done') router.replace(`/detect/${state.step}`);
  }, [hasHydrated, state.step, router]);

  if (!hasHydrated || state.step !== 'done') return null;

  return (
    <div className="text-center space-y-6 py-10">
      <p className="font-mono text-xs uppercase tracking-widest text-accent">All set</p>
      <h1 className="font-display text-4xl text-ink">Your listing is live.</h1>
      <p className="text-ink-soft max-w-md mx-auto">
        The property and its detected amenities are saved. Buyers and AI agents can find it
        via the browse page, sitemap, and JSONL feed.
      </p>

      <div className="flex items-center justify-center gap-3 pt-4">
        <Link
          href={`/properties/${state.propertyId}`}
          className="inline-flex items-center justify-center rounded-full bg-ink text-bg px-6 py-3 text-sm font-medium hover:-translate-y-0.5 transition-transform"
        >
          View the listing →
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
