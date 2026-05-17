import { forwardRef, useId, type InputHTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  /** Optional helper text shown under the input when no error is present. */
  hint?: string;
  /** Validation error; presence flips aria-invalid and replaces hint. */
  error?: string;
}

/**
 * Why `useId` instead of a hand-rolled prop?
 *   SSR + concurrent React both demand deterministic IDs. `useId` gives one
 *   that survives hydration. We pair it with `aria-describedby` so screen
 *   readers announce the error alongside the field.
 */
export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { label, hint, error, id, type = 'text', className, ...rest },
  ref,
) {
  const reactId = useId();
  const inputId = id ?? `in-${reactId}`;
  const messageId = `${inputId}-msg`;
  const hasError = Boolean(error);

  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={inputId} className="text-xs font-medium text-ink-soft">
        {label}
      </label>
      <input
        ref={ref}
        id={inputId}
        type={type}
        aria-invalid={hasError || undefined}
        aria-describedby={hint || error ? messageId : undefined}
        className={cn(
          'w-full rounded-[10px] border bg-surface px-3 py-2 text-sm text-ink',
          'placeholder:text-ink-faint focus:outline-none focus:ring-2 focus:ring-accent/30',
          hasError ? 'border-err' : 'border-border focus:border-border-strong',
          className,
        )}
        {...rest}
      />
      {(error || hint) && (
        <p
          id={messageId}
          className={cn('text-[11px]', hasError ? 'text-err' : 'text-ink-muted')}
        >
          {error ?? hint}
        </p>
      )}
    </div>
  );
});
