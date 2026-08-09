import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
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

import { LocalityEnrichment } from './locality-enrichment';

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

const address = {
  postalCode: '60311',
  street: 'Bendelstraße',
  countryCode: 'DE',
  radiusM: 3000,
};

describe('LocalityEnrichment', () => {
  beforeEach(() => {
    streamPersistLocality.mockReset();
  });

  it('runs the enrichment for the address entered in the config step', async () => {
    streamPersistLocality.mockResolvedValue(result);
    render(<LocalityEnrichment propertyId="prop-1" {...address} />);

    await waitFor(() => {
      expect(screen.getByText('Lively central area near a school.')).toBeInTheDocument();
    });
    expect(streamPersistLocality).toHaveBeenCalledWith(
      'prop-1',
      { postalCode: '60311', street: 'Bendelstraße', countryCode: 'DE', radiusM: 3000 },
      expect.any(Function),
    );
  });

  it('runs once, not once per render', async () => {
    streamPersistLocality.mockResolvedValue(result);
    const { rerender } = render(<LocalityEnrichment propertyId="prop-1" {...address} />);
    rerender(<LocalityEnrichment propertyId="prop-1" {...address} />);

    await waitFor(() => expect(streamPersistLocality).toHaveBeenCalled());
    expect(streamPersistLocality).toHaveBeenCalledTimes(1);
  });

  it('does nothing when the listing has no address', async () => {
    render(
      <LocalityEnrichment
        propertyId="prop-1"
        postalCode={undefined}
        street={undefined}
        countryCode="DE"
        radiusM={3000}
      />,
    );

    expect(screen.getByText(/No address was entered/)).toBeInTheDocument();
    expect(streamPersistLocality).not.toHaveBeenCalled();
  });

  it('offers a retry after a failure', async () => {
    streamPersistLocality.mockRejectedValueOnce(new Error('boom'));
    render(<LocalityEnrichment propertyId="prop-1" {...address} />);

    await waitFor(() => {
      expect(screen.getByText(/Could not enrich location/)).toBeInTheDocument();
    });

    streamPersistLocality.mockResolvedValue(result);
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));

    await waitFor(() => {
      expect(screen.getByText('Lively central area near a school.')).toBeInTheDocument();
    });
  });
});
