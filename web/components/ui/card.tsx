import { type ReactNode, type MouseEventHandler } from 'react';
import { cn } from '@/lib/cn';

interface CardBaseProps {
  children: ReactNode;
  padded?: boolean;
  hover?: boolean;
  className?: string;
}

interface ClickableCardProps extends CardBaseProps {
  onClick: MouseEventHandler<HTMLButtonElement>;
}

interface StaticCardProps extends CardBaseProps {
  onClick?: undefined;
}

/**
 * Cards default to a non-interactive <div>. Passing `onClick` upgrades the
 * element to a <button> so it is reachable by keyboard and surfaces through
 * the accessibility tree — without forcing every consumer to remember
 * role/tabIndex/keydown plumbing.
 */
export type CardProps = ClickableCardProps | StaticCardProps;

const BASE = 'bg-surface border border-border rounded-[14px] text-left block w-full';

export function Card(props: CardProps) {
  const { children, padded = true, hover = false, className } = props;
  const classes = cn(
    BASE,
    padded && 'p-6',
    hover &&
      'transition-[border-color,transform,box-shadow] duration-150 hover:border-border-strong hover:-translate-y-0.5 hover:shadow-[0_8px_24px_rgba(60,40,20,0.06)]',
    className,
  );

  if (props.onClick) {
    return (
      <button type="button" onClick={props.onClick} className={cn(classes, 'cursor-pointer')}>
        {children}
      </button>
    );
  }
  return <div className={classes}>{children}</div>;
}
