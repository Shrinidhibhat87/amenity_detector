'use client';

import { useState, type FormEvent } from 'react';
import { useRouter } from 'next/navigation';

/**
 * Slim natural-language search input rendered in the /browse header.
 *
 * Lives in its own Client Component so the surrounding /browse page can stay
 * an async Server Component (no `'use client'` boundary on the page itself).
 * Submitting navigates to /search?q=… so results are SSR-rendered, shareable
 * via URL, and the back button restores the input.
 */
export function BrowseSearchBar() {
  const router = useRouter();
  const [value, setValue] = useState('');

  function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const q = value.trim();
    if (q === '') return;
    router.push(`/search?q=${encodeURIComponent(q)}`);
  }

  return (
    <form onSubmit={onSubmit} className="flex items-center gap-2 w-full sm:w-96">
      <input
        type="search"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="e.g. 3 bedroom rent under 1500 EUR with fireplace"
        aria-label="Search properties in natural language"
        className="flex-1 rounded-full border border-border bg-surface px-4 py-2 text-sm text-ink placeholder:text-ink-muted focus:outline-none focus:border-border-strong"
      />
      <button
        type="submit"
        disabled={value.trim() === ''}
        className="rounded-full bg-ink text-bg px-4 py-2 text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:-translate-y-0.5 transition-transform"
      >
        Search →
      </button>
    </form>
  );
}
