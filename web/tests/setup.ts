import '@testing-library/jest-dom/vitest';
import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import { createElement } from 'react';
import type { ReactNode } from 'react';

afterEach(() => {
  cleanup();
});

// Next.js modules that don't work in jsdom — stub them with plain HTML equivalents.

vi.mock('next/image', () => ({
  default: ({
    src,
    alt,
    className,
    'data-testid': testId,
  }: {
    src: string;
    alt: string;
    fill?: boolean;
    priority?: boolean;
    sizes?: string;
    className?: string;
    'data-testid'?: string;
  }) => createElement('img', { src, alt, className, 'data-testid': testId }),
}));

vi.mock('next/link', () => ({
  default: ({
    href,
    children,
    className,
  }: {
    href: string;
    children: ReactNode;
    className?: string;
  }) => createElement('a', { href, className }, children),
}));
