import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Card } from './card';

describe('Card', () => {
  it('renders children', () => {
    render(<Card>hello</Card>);
    expect(screen.getByText('hello')).toBeInTheDocument();
  });

  it('renders as a button when onClick is provided so it is keyboard reachable', async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<Card onClick={onClick}>click me</Card>);
    const node = screen.getByRole('button', { name: /click me/i });
    await user.click(node);
    expect(onClick).toHaveBeenCalledOnce();
  });

  it('renders as a plain div when no onClick is provided', () => {
    render(<Card>just content</Card>);
    expect(screen.queryByRole('button')).toBeNull();
  });
});
