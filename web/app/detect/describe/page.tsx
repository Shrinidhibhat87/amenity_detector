'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { ApiError, patchProperty } from '@/lib/api';
import { useWizardStore } from '@/lib/wizard-store';
import { Button } from '@/components/ui';

export default function DescribeStepPage() {
  const router = useRouter();
  const state = useWizardStore((s) => s.state);
  const hasHydrated = useWizardStore((s) => s.hasHydrated);
  const setDescription = useWizardStore((s) => s.setDescription);
  const goBackToReview = useWizardStore((s) => s.goBackToReview);
  const finish = useWizardStore((s) => s.finish);

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Wrong-step guard, deferred until persist has rehydrated so the default
  // state does not bounce the user before the real step is known.
  useEffect(() => {
    if (!hasHydrated) return;
    if (state.step !== 'describe') {
      router.replace(state.step === 'done' ? '/detect/done' : `/detect/${state.step}`);
    }
  }, [hasHydrated, state.step, router]);

  if (!hasHydrated || state.step !== 'describe') return null;

  async function save() {
    if (state.step !== 'describe') return;
    setSaving(true);
    setError(null);
    try {
      // Persist description + any optional listing metadata the user filled
      // in during the config step. Description goes through the same PATCH
      // endpoint thanks to the schema change in api/schemas.py.
      const { config } = state;
      await patchProperty(state.propertyId, {
        description: state.description,
        ...(config.listing_type != null && { listing_type: config.listing_type }),
        ...(config.price != null && { price: config.price }),
        ...(config.currency != null && { currency: config.currency }),
        ...(config.price_period != null && { price_period: config.price_period }),
        ...(config.num_bedrooms != null && { num_bedrooms: config.num_bedrooms }),
        ...(config.num_bathrooms != null && { num_bathrooms: config.num_bathrooms }),
        ...(config.area_sqm != null && { area_sqm: config.area_sqm }),
        ...(config.property_type != null && { property_type: config.property_type }),
        ...(config.furnishing != null && { furnishing: config.furnishing }),
        ...(config.locality != null && { locality: config.locality }),
        ...(config.postal_code != null && { postal_code: config.postal_code }),
        ...(config.country_code != null && { country_code: config.country_code }),
      });
      finish();
      router.push('/detect/done');
    } catch (err) {
      setError(
        err instanceof ApiError
          ? `API ${err.status}: ${err.message}`
          : err instanceof Error
            ? err.message
            : 'Failed to save listing.',
      );
      setSaving(false);
    }
  }

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="font-display text-3xl text-ink">Listing description</h1>
        <p className="text-ink-soft">
          The model wrote this from the amenities you confirmed. Edit freely — this is what
          buyers and search engines will read.
        </p>
      </header>

      <textarea
        value={state.description}
        onChange={(e) => setDescription(e.target.value)}
        rows={14}
        className="w-full rounded-2xl border border-border bg-surface p-4 text-ink leading-relaxed font-sans focus:outline-none focus:border-border-strong"
        placeholder="Description will appear here…"
      />

      {error != null && <p className="text-warn text-sm font-mono">{error}</p>}

      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => {
            // Walk the step backwards before navigating so the review page's
            // wrong-step guard does not immediately redirect us back here.
            goBackToReview();
            router.push('/detect/review');
          }}
          className="font-mono text-xs uppercase tracking-widest text-ink-muted hover:text-ink transition-colors"
        >
          ← Back to review
        </button>
        <Button onClick={() => void save()} disabled={saving}>
          {saving ? 'Saving…' : 'Save & publish'}
        </Button>
      </div>
    </div>
  );
}
