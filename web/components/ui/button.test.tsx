import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Button } from './button';

describe('Button', () => {
  it('renders its children', () => {
    render(<Button>Detect amenities</Button>);
    expect(screen.getByRole('button', { name: /detect amenities/i })).toBeInTheDocument();
  });

  it('fires onClick when clicked', async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Go</Button>);
    await user.click(screen.getByRole('button'));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it('does not fire onClick when disabled', async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(
      <Button onClick={onClick} disabled>
        Go
      </Button>,
    );
    await user.click(screen.getByRole('button'));
    expect(onClick).not.toHaveBeenCalled();
  });

  it('applies variant + size styles via classnames', () => {
    render(
      <Button variant="accent" size="lg">
        Big
      </Button>,
    );
    const btn = screen.getByRole('button');
    expect(btn.className).toMatch(/bg-accent/);
    expect(btn.className).toMatch(/text-base|text-\[15px\]/);
  });

  it('defaults type to "button" to avoid accidental form submit', () => {
    render(<Button>Safe</Button>);
    expect(screen.getByRole('button')).toHaveAttribute('type', 'button');
  });

  it('forwards refs', () => {
    const ref = { current: null as HTMLButtonElement | null };
    render(<Button ref={ref}>ref</Button>);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });
});
