import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LocalityPanel } from './locality-panel';
import type { LocalityInsight } from '@/lib/schemas';

const insight: LocalityInsight = {
  display_name: '60311 Frankfurt am Main, Germany',
  latitude: 50.1109,
  longitude: 8.6821,
  radius_m: 1000,
  blurb: 'Central spot with a school and a park within walking distance.',
  category_counts: { school: 1, park: 2, transit: 0 },
  pois: [
    {
      category: 'school',
      name: 'Goethe-Schule',
      latitude: 50.111,
      longitude: 8.682,
      distance_m: 120,
      osm_type: 'node',
      osm_id: 1,
    },
    {
      category: 'park',
      name: 'Grüneburgpark',
      latitude: 50.12,
      longitude: 8.67,
      distance_m: 1400,
      osm_type: 'way',
      osm_id: 2,
    },
  ],
  attribution: '© OpenStreetMap contributors',
};

describe('LocalityPanel', () => {
  it('shows the radius, category rows with counts, blurb and attribution', () => {
    render(<LocalityPanel insight={insight} />);

    expect(screen.getByText('within 1 km')).toBeInTheDocument();
    expect(screen.getByText('Schools')).toBeInTheDocument();
    expect(screen.getByText('Parks')).toBeInTheDocument();
    expect(screen.getByText(/within walking distance/)).toBeInTheDocument();
    expect(screen.getByText('© OpenStreetMap contributors')).toBeInTheDocument();
    // Zero-count category is hidden.
    expect(screen.queryByText('Public transport')).not.toBeInTheDocument();
  });

  it('expands the densest category by default and shows distances in km', () => {
    render(<LocalityPanel insight={insight} />);

    // Parks (count 2) is the densest → open by default.
    expect(screen.getByText('Grüneburgpark')).toBeInTheDocument();
    expect(screen.getByText('1.4 km')).toBeInTheDocument();
    // It has more total than sampled → "+1 more nearby".
    expect(screen.getByText('+1 more nearby')).toBeInTheDocument();
    // The other category starts collapsed.
    expect(screen.queryByText('Goethe-Schule')).not.toBeInTheDocument();
  });

  it('toggles a category open on click', async () => {
    const user = userEvent.setup();
    render(<LocalityPanel insight={insight} />);

    await user.click(screen.getByRole('button', { name: /Schools/ }));

    expect(screen.getByText('Goethe-Schule')).toBeInTheDocument();
    // Metres rendered as km: 120 m → 0.1 km.
    expect(screen.getByText('0.1 km')).toBeInTheDocument();
  });
});
