import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { LocalityInsight } from '@/lib/schemas';

// Mock the API module so the component never hits the network.
const streamPersistLocality = vi.fn();
vi.mock('@/lib/api', () => ({
  streamPersistLocality: (...args: unknown[]) => streamPersistLocality(...args),
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
    streamPersistLocality.mockReset();
  });

  it('persists the PIN with country and radius, then renders the blurb', async () => {
    streamPersistLocality.mockResolvedValue(result);
    const user = userEvent.setup();
    render(<LocationInput propertyId="prop-1" />);

    await user.type(screen.getByLabelText('PIN'), '60311');
    await user.click(screen.getByRole('button', { name: 'Find' }));

    await waitFor(() => {
      expect(screen.getByText('Lively central area near a school.')).toBeInTheDocument();
    });
    expect(streamPersistLocality).toHaveBeenCalledWith(
      'prop-1',
      {
        postalCode: '60311',
        street: undefined,
        countryCode: 'DE',
        radiusM: 3000,
      },
      expect.any(Function),
    );
  });

  it('forwards the chosen radius', async () => {
    streamPersistLocality.mockResolvedValue(result);
    const user = userEvent.setup();
    render(<LocationInput propertyId="prop-1" />);

    await user.type(screen.getByLabelText('PIN'), '60311');
    const slider = screen.getByLabelText('Search radius in kilometres');
    fireEvent.change(slider, { target: { value: '7' } });
    await user.click(screen.getByRole('button', { name: 'Find' }));

    await waitFor(() => expect(streamPersistLocality).toHaveBeenCalled());
    expect(streamPersistLocality).toHaveBeenCalledWith(
      'prop-1',
      expect.objectContaining({ radiusM: 7000 }),
      expect.any(Function),
    );
  });

  it('does not call the API when the input is blank', async () => {
    const user = userEvent.setup();
    render(<LocationInput propertyId="prop-1" />);

    // Button is disabled while empty; clicking does nothing.
    const button = screen.getByRole('button', { name: 'Find' });
    expect(button).toBeDisabled();
    await user.click(button);
    expect(streamPersistLocality).not.toHaveBeenCalled();
  });

  it('shows an error when enrichment fails', async () => {
    streamPersistLocality.mockRejectedValue(new Error('boom'));
    const user = userEvent.setup();
    render(<LocationInput propertyId="prop-1" />);

    await user.type(screen.getByLabelText('PIN'), 'nowhere');
    await user.click(screen.getByRole('button', { name: 'Find' }));

    await waitFor(() => {
      expect(screen.getByText(/Could not enrich location/)).toBeInTheDocument();
    });
  });
});
