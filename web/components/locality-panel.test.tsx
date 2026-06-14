import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
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
  it('renders the blurb, non-zero category counts, nearest POIs and attribution', () => {
    render(<LocalityPanel insight={insight} />);

    expect(screen.getByText(/within walking distance/)).toBeInTheDocument();
    expect(screen.getByText('Schools: 1')).toBeInTheDocument();
    expect(screen.getByText('Parks: 2')).toBeInTheDocument();
    // Zero-count category is hidden.
    expect(screen.queryByText(/Public transport:/)).not.toBeInTheDocument();
    expect(screen.getByText('Goethe-Schule')).toBeInTheDocument();
    // Distance formatting: metres under 1 km, km above.
    expect(screen.getByText(/120 m/)).toBeInTheDocument();
    expect(screen.getByText(/1\.4 km/)).toBeInTheDocument();
    expect(screen.getByText('© OpenStreetMap contributors')).toBeInTheDocument();
  });
});
