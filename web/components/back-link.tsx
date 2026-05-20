'use client';

import { useRouter } from 'next/navigation';

/**
 * Browser-history back affordance.
 *
 * Calling ``router.back()`` walks the history stack — useful when the user
 * arrived from a different list/detail page and "Home" is too coarse an
 * exit. Renders as a button (not an <a>) so it doesn't navigate to a stable
 * URL; the destination depends on history.
 *
 * Falls back to ``/`` when there is no previous entry (deep link, fresh tab).
 */
export function BackLink({ className }: { className?: string }) {
  const router = useRouter();

  function onClick(): void {
    // Next's typed router does not expose a "can go back" check, so we lean
    // on the window history length as a proxy. On a fresh tab it is 1.
    if (typeof window !== 'undefined' && window.history.length > 1) {
      router.back();
    } else {
      router.push('/');
    }
  }

  return (
    <button
      type="button"
      onClick={onClick}
      className={
        className ??
        'font-mono text-xs uppercase tracking-[0.18em] text-ink-muted hover:text-ink transition-colors'
      }
    >
      ← Back
    </button>
  );
}
