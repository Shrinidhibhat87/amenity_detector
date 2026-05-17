import type { ReactNode } from 'react';
import { WizardStepper } from '@/components/wizard-stepper';

export const metadata = {
  title: 'New listing — Amenity Detector',
  robots: { index: false, follow: false },
};

export default function DetectLayout({ children }: { children: ReactNode }) {
  return (
    <main className="flex-1 px-6 py-8 max-w-4xl mx-auto w-full">
      <header className="mb-10">
        <p className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted mb-3">
          New listing
        </p>
        <WizardStepper />
      </header>
      {children}
    </main>
  );
}
