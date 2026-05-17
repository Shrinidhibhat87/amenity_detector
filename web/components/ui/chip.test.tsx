import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Chip } from './chip';

describe('Chip', () => {
  it('renders children', () => {
    render(<Chip>Balcony</Chip>);
    expect(screen.getByText('Balcony')).toBeInTheDocument();
  });

  it('reflects active state via aria-pressed when clickable', () => {
    render(<Chip active onClick={() => {}}>Furnished</Chip>);
    expect(screen.getByRole('button', { name: /furnished/i })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
  });

  it('fires onClick when clicked', async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<Chip onClick={onClick}>Pet friendly</Chip>);
    await user.click(screen.getByRole('button'));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it('renders as a non-interactive span when no onClick is supplied', () => {
    const { container } = render(<Chip>Static</Chip>);
    expect(container.querySelector('span')).not.toBeNull();
    expect(screen.queryByRole('button')).toBeNull();
  });
});
