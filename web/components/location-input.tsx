'use client';

/**
 * Location (Lage) side panel for the upload wizard.
 *
 * The user picks a country, types a PIN (required) and optional street, and sets
 * a search radius; we kick off the locality enrichment in the background —
 * persisted straight onto the property shell, which already exists by the upload
 * step. Because it runs while images upload and process, the neighbourhood blurb
 * is usually ready by the time the user reaches review: the two timelines overlap
 * instead of adding up.
 */

import { useState } from 'react';
import { ApiError, streamPersistLocality, type LocalityProgress } from '@/lib/api';
import type { LocalityInsight } from '@/lib/schemas';
import { Button, Input, Select } from '@/components/ui';
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

// Countries we bias the PIN lookup to. DE leads — the target portals are German.
const COUNTRIES = [
  { value: 'DE', label: '🇩🇪 DE' },
  { value: 'AT', label: '🇦🇹 AT' },
  { value: 'CH', label: '🇨🇭 CH' },
  { value: 'NL', label: '🇳🇱 NL' },
  { value: 'FR', label: '🇫🇷 FR' },
  { value: 'GB', label: '🇬🇧 GB' },
  { value: 'US', label: '🇺🇸 US' },
] as const;

const MIN_KM = 1;
const MAX_KM = 10;
const DEFAULT_M = 3000;

export function LocationInput({ propertyId }: { propertyId: string }) {
  const [country, setCountry] = useState('DE');
  const [postalCode, setPostalCode] = useState('');
  const [street, setStreet] = useState('');
  const [radiusM, setRadiusM] = useState(DEFAULT_M);
  const [status, setStatus] = useState<Status>('idle');
  const [insight, setInsight] = useState<LocalityInsight | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<LocalityProgress | null>(null);

  const radiusKm = radiusM / 1000;
  const fillPct = ((radiusKm - MIN_KM) / (MAX_KM - MIN_KM)) * 100;

  async function run() {
    const pin = postalCode.trim();
    if (pin.length === 0 || status === 'running') return;
    setStatus('running');
    setError(null);
    setProgress(null);
    try {
      const result = await streamPersistLocality(
        propertyId,
        {
          postalCode: pin,
          street: street.trim() || undefined,
          countryCode: country,
          radiusM,
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
  }

  return (
    <div className="rounded-2xl border border-border bg-surface p-5 space-y-4">
      <div className="space-y-1">
        <h2 className="font-display text-lg text-ink">Location (Lage)</h2>
        <p className="text-sm text-ink-soft">
          PIN code and country. We find what is nearby while your photos upload.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-2">
        <div className="w-[88px] shrink-0">
          <Select
            label="Country"
            options={COUNTRIES}
            value={country}
            onChange={(e) => setCountry(e.target.value)}
          />
        </div>
        <div className="w-[110px] shrink-0">
          <Input
            label="PIN"
            value={postalCode}
            onChange={(e) => setPostalCode(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void run();
            }}
            placeholder="52062"
            inputMode="numeric"
          />
        </div>
        <div className="min-w-[120px] flex-1">
          <Input
            label="Street (optional)"
            value={street}
            onChange={(e) => setStreet(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void run();
            }}
            placeholder="Bendelstraße"
          />
        </div>
        <Button
          variant="accent"
          onClick={() => void run()}
          disabled={postalCode.trim().length === 0 || status === 'running'}
        >
          {status === 'running' ? 'Finding…' : 'Find'}
        </Button>
      </div>

      <div className="space-y-1.5">
        <div className="flex items-baseline justify-between">
          <label htmlFor="locality-radius" className="text-xs font-medium text-ink-soft">
            Search radius
          </label>
          <span className="font-mono text-xs text-accent tabular-nums">{radiusKm} km</span>
        </div>
        <input
          id="locality-radius"
          type="range"
          className="ad-range"
          style={{ ['--ad-range-fill' as string]: `${fillPct}%` }}
          min={MIN_KM}
          max={MAX_KM}
          step={1}
          value={radiusKm}
          aria-label="Search radius in kilometres"
          onChange={(e) => setRadiusM(Number(e.target.value) * 1000)}
        />
        <div className="flex justify-between font-mono text-[10px] text-ink-faint">
          <span>{MIN_KM} km</span>
          <span>{MAX_KM} km</span>
        </div>
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
              style={{ width: progress == null ? '8%' : `${(progress.done / progress.total) * 100}%` }}
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
