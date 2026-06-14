import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { LocalityInsight } from '@/lib/schemas';

// Mock the API module so the component never hits the network.
const persistLocality = vi.fn();
vi.mock('@/lib/api', () => ({
  persistLocality: (...args: unknown[]) => persistLocality(...args),
  ApiError: class ApiError extends Error {
    constructor(
      public status: number,
      message: string,
    ) {
      super(message);
    }
  },
}));

import { LocationInput } from './location-input';

const result: LocalityInsight = {
  display_name: 'Frankfurt',
  latitude: 50.11,
  longitude: 8.68,
  radius_m: 1000,
  blurb: 'Lively central area near a school.',
  category_counts: { school: 1 },
  pois: [],
  attribution: '© OpenStreetMap contributors',
};

describe('LocationInput', () => {
  beforeEach(() => {
    persistLocality.mockReset();
  });

  it('persists the PIN with country and radius, then renders the blurb', async () => {
    persistLocality.mockResolvedValue(result);
    const user = userEvent.setup();
    render(<LocationInput propertyId="prop-1" />);

    await user.type(screen.getByLabelText('PIN'), '60311');
    await user.click(screen.getByRole('button', { name: 'Find' }));

    await waitFor(() => {
      expect(screen.getByText('Lively central area near a school.')).toBeInTheDocument();
    });
    expect(persistLocality).toHaveBeenCalledWith('prop-1', {
      postalCode: '60311',
      street: undefined,
      countryCode: 'DE',
      radiusM: 3000,
    });
  });

  it('forwards the chosen radius', async () => {
    persistLocality.mockResolvedValue(result);
    const user = userEvent.setup();
    render(<LocationInput propertyId="prop-1" />);

    await user.type(screen.getByLabelText('PIN'), '60311');
    const slider = screen.getByLabelText('Search radius in kilometres');
    fireEvent.change(slider, { target: { value: '7' } });
    await user.click(screen.getByRole('button', { name: 'Find' }));

    await waitFor(() => expect(persistLocality).toHaveBeenCalled());
    expect(persistLocality).toHaveBeenCalledWith(
      'prop-1',
      expect.objectContaining({ radiusM: 7000 }),
    );
  });

  it('does not call the API when the input is blank', async () => {
    const user = userEvent.setup();
    render(<LocationInput propertyId="prop-1" />);

    // Button is disabled while empty; clicking does nothing.
    const button = screen.getByRole('button', { name: 'Find' });
    expect(button).toBeDisabled();
    await user.click(button);
    expect(persistLocality).not.toHaveBeenCalled();
  });

  it('shows an error when enrichment fails', async () => {
    persistLocality.mockRejectedValue(new Error('boom'));
    const user = userEvent.setup();
    render(<LocationInput propertyId="prop-1" />);

    await user.type(screen.getByLabelText('PIN'), 'nowhere');
    await user.click(screen.getByRole('button', { name: 'Find' }));

    await waitFor(() => {
      expect(screen.getByText(/Could not enrich location/)).toBeInTheDocument();
    });
  });
});
