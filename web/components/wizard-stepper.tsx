'use client';

import { useWizardStore, type WizardState } from '@/lib/wizard-store';

const STEPS: ReadonlyArray<{ key: WizardState['step']; label: string }> = [
  { key: 'config', label: 'Configure' },
  { key: 'upload', label: 'Upload' },
  { key: 'review', label: 'Review' },
  { key: 'describe', label: 'Describe' },
  { key: 'done', label: 'Done' },
];

export function WizardStepper() {
  const current = useWizardStore((s) => s.state.step);
  const currentIndex = STEPS.findIndex((s) => s.key === current);

  return (
    <ol className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest">
      {STEPS.map((step, i) => {
        const isCurrent = step.key === current;
        const isPast = i < currentIndex;
        return (
          <li key={step.key} className="flex items-center gap-2">
            <span
              className={[
                'inline-flex items-center justify-center w-6 h-6 rounded-full border',
                isCurrent
                  ? 'bg-ink text-bg border-ink'
                  : isPast
                    ? 'bg-accent-soft text-accent border-accent-soft'
                    : 'bg-transparent text-ink-muted border-border',
              ].join(' ')}
            >
              {i + 1}
            </span>
            <span
              className={
                isCurrent ? 'text-ink' : isPast ? 'text-accent' : 'text-ink-muted'
              }
            >
              {step.label}
            </span>
            {i < STEPS.length - 1 && (
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
