import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Compose Tailwind class lists safely.
 *
 * `clsx` flattens conditional inputs into a single string. `twMerge` then
 * resolves Tailwind conflicts (e.g. `p-2 p-4` → `p-4`) so callers can override
 * a base class by passing a later class with the same property family.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
