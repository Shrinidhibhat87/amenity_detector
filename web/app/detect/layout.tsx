import type { ReactNode } from 'react';
import Link from 'next/link';
import { WizardStepper } from '@/components/wizard-stepper';

export const metadata = {
  title: 'New listing — Amenity Detector',
  robots: { index: false, follow: false },
};

export default function DetectLayout({ children }: { children: ReactNode }) {
  return (
    <main className="flex-1 px-6 py-8 max-w-4xl mx-auto w-full">
      <header className="mb-10">
        <div className="flex items-center justify-between mb-3">
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted">
            New listing
          </p>
          <Link
            href="/"
            className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted hover:text-ink transition-colors"
          >
            ← Home
          </Link>
        </div>
        <WizardStepper />
      </header>
      {children}
    </main>
  );
}
