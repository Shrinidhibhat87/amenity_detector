'use client';

import { useRouter } from 'next/navigation';
import { useWizardStore, WIZARD_STEPS, type WizardStep } from '@/lib/wizard-store';

const LABELS: Record<WizardStep, string> = {
  config: 'Configure',
  upload: 'Upload',
  review: 'Review',
  describe: 'Describe',
  done: 'Done',
};

export function WizardStepper() {
  const router = useRouter();
  const current = useWizardStore((s) => s.state.step);
  const furthest = useWizardStore((s) => s.state.furthest);
  const canVisit = useWizardStore((s) => s.canVisit);
  const goToStep = useWizardStore((s) => s.goToStep);

  const furthestIndex = WIZARD_STEPS.indexOf(furthest);

  function visit(step: WizardStep) {
    goToStep(step);
    router.push(`/detect/${step}`);
  }

  return (
    <ol className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest">
      {WIZARD_STEPS.map((step, i) => {
        const isCurrent = step === current;
        const isReached = i <= furthestIndex;
        const clickable = !isCurrent && canVisit(step);

        const marker = (
          <>
            <span
              className={[
                'inline-flex items-center justify-center w-6 h-6 rounded-full border',
                isCurrent
                  ? 'bg-ink text-bg border-ink'
                  : isReached
                    ? 'bg-accent-soft text-accent border-accent-soft'
                    : 'bg-transparent text-ink-muted border-border',
              ].join(' ')}
            >
              {i + 1}
            </span>
            <span
              className={isCurrent ? 'text-ink' : isReached ? 'text-accent' : 'text-ink-muted'}
            >
              {LABELS[step]}
            </span>
          </>
        );

        return (
          <li key={step} className="flex items-center gap-2">
            {clickable ? (
              <button
                type="button"
                onClick={() => visit(step)}
                aria-label={`Go back to ${LABELS[step]}`}
                className="flex items-center gap-2 cursor-pointer hover:opacity-70 transition-opacity"
              >
                {marker}
              </button>
            ) : (
              <span
                className="flex items-center gap-2"
                aria-current={isCurrent ? 'step' : undefined}
              >
                {marker}
              </span>
            )}
            {i < WIZARD_STEPS.length - 1 && (
              <span className="text-ink-muted/40 mx-1" aria-hidden>
                ─
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}
