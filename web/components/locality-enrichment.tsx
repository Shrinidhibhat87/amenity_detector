'use client';

/**
 * Locality (Lage) panel for the upload wizard.
 *
 * The address is entered once, in the config step, and lives on the property
 * from creation. This panel therefore collects nothing: it runs the enrichment
 * for that address as soon as the upload step opens, so the neighbourhood
 * blurb is usually ready by the time the user reaches review — the enrichment
 * and the image detections overlap instead of adding up.
 *
 * It re-runs only when the user asks, which is also the recovery path when
 * Nominatim or Overpass is briefly unavailable.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, streamPersistLocality, type LocalityProgress } from '@/lib/api';
import type { LocalityInsight } from '@/lib/schemas';
import { Button } from '@/components/ui';
import { LocalityPanel } from '@/components/locality-panel';

type Status = 'idle' | 'running' | 'done' | 'error';

// Friendly labels for the in-progress category step (mirrors core/locality).
const STEP_LABELS: Record<string, string> = {
  supermarket: 'supermarkets',
  school: 'schools',
  gym: 'gyms',
  park: 'parks',
  transit: 'public transport',
  pharmacy: 'pharmacies',
  airport: 'airports',
};

export interface LocalityEnrichmentProps {
  propertyId: string;
  postalCode: string | undefined;
  street: string | undefined;
  countryCode: string | undefined;
  radiusM: number | undefined;
}

export function LocalityEnrichment({
  propertyId,
  postalCode,
  street,
  countryCode,
  radiusM,
}: LocalityEnrichmentProps) {
  const [status, setStatus] = useState<Status>('idle');
  const [insight, setInsight] = useState<LocalityInsight | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<LocalityProgress | null>(null);

  // Guards the auto-run so React's development double-invoke (and any
  // re-render) does not fire a second enrichment for the same property.
  const startedFor = useRef<string | null>(null);

  const run = useCallback(async () => {
    const pin = postalCode?.trim() ?? '';
    if (pin.length === 0) return;
    setStatus('running');
    setError(null);
    setProgress(null);
    try {
      const result = await streamPersistLocality(
        propertyId,
        {
          postalCode: pin,
          street: street?.trim() !== '' ? street?.trim() : undefined,
          countryCode: countryCode ?? 'DE',
          radiusM: radiusM ?? 3000,
        },
        (p) => setProgress(p),
      );
      setInsight(result);
      setStatus('done');
    } catch (err) {
      const msg =
        err instanceof ApiError
          ? `API ${err.status}`
          : err instanceof Error
            ? err.message
            : 'Unknown error';
      setError(msg);
      setStatus('error');
    }
  }, [propertyId, postalCode, street, countryCode, radiusM]);

  useEffect(() => {
    if (startedFor.current === propertyId) return;
    startedFor.current = propertyId;
    void run();
  }, [propertyId, run]);

  const hasAddress = (postalCode?.trim() ?? '') !== '';

  return (
    <div className="rounded-2xl border border-border bg-surface p-5 space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h2 className="font-display text-lg text-ink">Location (Lage)</h2>
          <p className="text-sm text-ink-soft">
            {hasAddress
              ? `Finding what is nearby ${[street, postalCode].filter(Boolean).join(', ')} while your photos upload.`
              : 'No address was entered for this listing, so the neighbourhood section stays empty.'}
          </p>
        </div>
        {hasAddress && status !== 'running' && (
          <Button variant="ghost" onClick={() => void run()}>
            {status === 'error' ? 'Retry' : 'Refresh'}
          </Button>
        )}
      </div>

      {status === 'running' && (
        <div className="space-y-1.5" role="status" aria-live="polite">
          <div className="flex items-baseline justify-between font-mono text-xs text-ink-muted">
            <span>
              {progress == null
                ? 'Locating…'
                : `Scanning ${STEP_LABELS[progress.category] ?? progress.category}…`}
            </span>
            {progress != null && (
              <span className="text-ink-faint tabular-nums">
                {progress.done}/{progress.total}
              </span>
            )}
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-muted">
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-300 ease-out"
              style={{
                width: progress == null ? '8%' : `${(progress.done / progress.total) * 100}%`,
              }}
            />
          </div>
        </div>
      )}
      {status === 'error' && error != null && (
        <p className="font-mono text-xs text-err">Could not enrich location: {error}</p>
      )}
      {status === 'done' && insight != null && <LocalityPanel insight={insight} />}
    </div>
  );
}
