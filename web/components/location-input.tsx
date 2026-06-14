'use client';

/**
 * Location (Lage) side panel for the upload wizard.
 *
 * The user types a PIN code, street, or Stadtteil and we kick off the locality
 * agent in the background — persisted straight onto the property shell, which
 * already exists by the upload step. Because it runs while images upload and
 * process, the neighbourhood blurb is usually ready by the time the user
 * reaches review: the two timelines overlap instead of adding up.
 */

import { useState } from 'react';
import { ApiError, persistLocality } from '@/lib/api';
import type { LocalityInsight } from '@/lib/schemas';
import { Button } from '@/components/ui';
import { LocalityPanel } from '@/components/locality-panel';

type Status = 'idle' | 'running' | 'done' | 'error';

export function LocationInput({ propertyId }: { propertyId: string }) {
  const [location, setLocation] = useState('');
  const [status, setStatus] = useState<Status>('idle');
  const [insight, setInsight] = useState<LocalityInsight | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    const trimmed = location.trim();
    if (trimmed.length === 0 || status === 'running') return;
    setStatus('running');
    setError(null);
    try {
      const result = await persistLocality(propertyId, trimmed);
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
  }

  return (
    <div className="rounded-2xl border border-border bg-surface p-5 space-y-3">
      <div className="space-y-1">
        <h2 className="font-display text-lg text-ink">Location (Lage)</h2>
        <p className="text-sm text-ink-soft">
          PIN code, street, or Stadtteil. We find what is nearby while your photos upload.
        </p>
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') void run();
          }}
          placeholder="e.g. 60311 or Bockenheim"
          aria-label="Location"
          className="flex-1 rounded-xl border border-border bg-bg px-3 py-2 text-sm text-ink placeholder:text-ink-muted focus:border-border-strong focus:outline-none"
        />
        <Button onClick={() => void run()} disabled={location.trim().length === 0 || status === 'running'}>
          {status === 'running' ? 'Finding…' : 'Find'}
        </Button>
      </div>

      {status === 'running' && (
        <p className="font-mono text-xs text-ink-muted">Looking up the neighbourhood…</p>
      )}
      {status === 'error' && error != null && (
        <p className="font-mono text-xs text-warn">Could not enrich location: {error}</p>
      )}
      {status === 'done' && insight != null && <LocalityPanel insight={insight} />}
    </div>
  );
}
