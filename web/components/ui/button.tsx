import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

/**
 * Variant + size are string-literal unions instead of bare strings so the
 * compiler catches typos at the call site. They map 1:1 to the variants in
 * `ui/docs/design-system.jsx` (primary/accent/ghost/soft/quiet).
 */
export type ButtonVariant = 'primary' | 'accent' | 'ghost' | 'soft' | 'quiet';
export type ButtonSize = 'sm' | 'md' | 'lg';

// We extend the native button attributes so callers can pass anything a
// <button> accepts (aria-*, data-*, form, etc) without us re-declaring each.
export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

const VARIANTS: Record<ButtonVariant, string> = {
  primary: 'bg-ink text-bg border-ink hover:bg-ink-soft',
  accent: 'bg-accent text-bg border-accent hover:bg-accent-hover',
  ghost: 'bg-transparent text-ink border-border hover:border-border-strong',
  soft: 'bg-surface-alt text-ink border-transparent hover:bg-surface-muted',
  quiet: 'bg-transparent text-ink-soft border-transparent hover:text-ink',
};

const SIZES: Record<ButtonSize, string> = {
  sm: 'px-3.5 py-1.5 text-[13px]',
  md: 'px-5 py-2.5 text-sm',
  lg: 'px-6 py-3.5 text-[15px]',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'primary', size = 'md', type = 'button', className, disabled, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      disabled={disabled}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-full border font-medium',
        'transition-[transform,background-color,border-color,color] duration-150 ease-out',
        'hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40',
        'disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0',
        SIZES[size],
        VARIANTS[variant],
        className,
      )}
      {...rest}
    />
  );
});
