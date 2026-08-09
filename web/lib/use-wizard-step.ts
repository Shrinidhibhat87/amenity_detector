'use client';

import { useEffect, useRef } from 'react';
import { useWizardStore, type WizardStep } from '@/lib/wizard-store';

/**
 * Point the store's `step` at the route that is currently open.
 *
 * Runs once per mount rather than on every store change. During a client-side
 * navigation the outgoing and incoming step pages are both mounted for a tick,
 * so a guard that re-ran on every change let the outgoing page claim `step`
 * straight back — the reason the stepper could not walk back to Configure,
 * whose page redirects to `state.step` instead of correcting it.
 *
 * `goToStep` ignores steps the work has not reached, so this only ever syncs
 * to a legitimately visitable step.
 */
export function useWizardStepSync(step: WizardStep): void {
  const hasHydrated = useWizardStore((s) => s.hasHydrated);
  const goToStep = useWizardStore((s) => s.goToStep);
  const synced = useRef(false);

  useEffect(() => {
    if (!hasHydrated || synced.current) return;
    synced.current = true;
    goToStep(step);
  }, [hasHydrated, step, goToStep]);
}
