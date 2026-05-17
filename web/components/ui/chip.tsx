import { type ReactNode, type MouseEventHandler } from 'react';
import { cn } from '@/lib/cn';

export type ChipTone = 'default' | 'accent' | 'ok' | 'warn';

interface BaseChipProps {
  children: ReactNode;
  tone?: ChipTone;
  active?: boolean;
  className?: string;
}

interface ClickableChipProps extends BaseChipProps {
  onClick: MouseEventHandler<HTMLButtonElement>;
}

interface StaticChipProps extends BaseChipProps {
  onClick?: undefined;
}

/**
 * Discriminated by the presence of `onClick`: clickable chips render as a
 * <button> with `aria-pressed` reflecting `active`; static chips render as a
 * <span>. This keeps the DOM accessible without sprinkling role="button"
 * onto <div>s.
 *
 * The two prop variants are unioned so the compiler picks the right shape at
 * each call site.
 */
export type ChipProps = ClickableChipProps | StaticChipProps;

const TONES: Record<ChipTone, { idle: string; active: string }> = {
  default: {
    idle: 'bg-surface text-ink-soft border-border hover:border-border-strong',
    active: 'bg-ink text-bg border-ink',
  },
  accent: {
    idle: 'bg-accent-soft text-accent-ink border-transparent',
    active: 'bg-accent text-bg border-accent',
  },
  ok: {
    idle: 'bg-ok-soft text-ok border-transparent',
    active: 'bg-ok text-bg border-ok',
  },
  warn: {
    idle: 'bg-warn-soft text-warn border-transparent',
    active: 'bg-warn text-bg border-warn',
  },
};

const BASE =
  'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[12.5px] font-medium leading-none transition-colors';

export function Chip(props: ChipProps) {
  const { children, tone = 'default', active = false, className } = props;
  const palette = TONES[tone];
  const classes = cn(BASE, active ? palette.active : palette.idle, className);

  if (props.onClick) {
    return (
      <button
        type="button"
        onClick={props.onClick}
        aria-pressed={active}
        className={cn(classes, 'cursor-pointer')}
      >
        {children}
      </button>
    );
  }
  return <span className={classes}>{children}</span>;
}
