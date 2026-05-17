import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Input } from './input';

describe('Input', () => {
  it('renders a labelled input with type="text" by default', () => {
    render(<Input label="Property name" name="name" />);
    const field = screen.getByLabelText('Property name');
    expect(field).toBeInTheDocument();
    expect(field).toHaveAttribute('type', 'text');
  });

  it('accepts typing', async () => {
    const user = userEvent.setup();
    render(<Input label="Locality" name="locality" />);
    const field = screen.getByLabelText('Locality');
    await user.type(field, 'Frankenberger Viertel');
    expect(field).toHaveValue('Frankenberger Viertel');
  });

  it('surfaces an error message via aria-describedby', () => {
    render(<Input label="Email" name="email" error="Not a valid address" />);
    const field = screen.getByLabelText('Email');
    expect(field).toHaveAttribute('aria-invalid', 'true');
    const describedBy = field.getAttribute('aria-describedby');
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy as string)?.textContent).toBe(
      'Not a valid address',
    );
  });
});
