import { forwardRef, useId, type SelectHTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

export interface SelectOption {
  value: string;
  label: string;
}

export interface SelectProps
  extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'children'> {
  label: string;
  options: ReadonlyArray<SelectOption>;
  hint?: string;
  error?: string;
}

/**
 * Use `Omit<…, 'children'>` to drop the native `children` prop and force
 * callers to pass `options` instead. That way each option carries a stable
 * `value` and a typed `label` rather than a free-form ReactNode children list.
 */
export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { label, options, hint, error, id, className, ...rest },
  ref,
) {
  const reactId = useId();
  const selectId = id ?? `sel-${reactId}`;
  const messageId = `${selectId}-msg`;
  const hasError = Boolean(error);

  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={selectId} className="text-xs font-medium text-ink-soft">
        {label}
      </label>
      <select
        ref={ref}
        id={selectId}
        aria-invalid={hasError || undefined}
        aria-describedby={hint || error ? messageId : undefined}
        className={cn(
          'w-full rounded-[10px] border bg-surface px-3 py-2 text-sm text-ink',
          'focus:outline-none focus:ring-2 focus:ring-accent/30',
          hasError ? 'border-err' : 'border-border focus:border-border-strong',
          className,
        )}
        {...rest}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
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
