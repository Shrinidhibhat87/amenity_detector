import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PropertyCard } from './property-card';
import type { PropertySummary } from '@/lib/schemas';

// getImageUrl depends on env; stub before import in tests.
vi.stubEnv('NEXT_PUBLIC_API_BASE_URL', 'http://test:8000');

const base: PropertySummary = {
  id: 'prop-1',
  name: 'Sunny Loft',
  created_at: '2024-06-01T10:00:00',
  description: 'A nice place',
  model_used: null,
  extra_info: null,
  image_count: 1,
  first_image_id: 'img-abc',
  slug: null,
  listing_type: 'rent',
  price: 1200,
  currency: 'EUR',
  price_period: 'monthly',
  num_bedrooms: 2,
  num_bathrooms: 1,
  area_sqm: 65,
  property_type: 'apartment',
  furnishing: 'furnished',
  available_from: null,
  locality: 'Berlin',
  country_code: 'DE',
  latitude: null,
  longitude: null,
  postal_code: null,
  owner_email: null,
};

describe('PropertyCard', () => {
  it('renders the property name', () => {
    render(<PropertyCard property={base} />);
    expect(screen.getByText('Sunny Loft')).toBeInTheDocument();
  });

  it('links to the property detail page', () => {
    render(<PropertyCard property={base} />);
    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', '/properties/prop-1');
  });

  it('renders locality and country code', () => {
    render(<PropertyCard property={base} />);
    expect(screen.getByText('Berlin, DE')).toBeInTheDocument();
  });

  it('does not render location when both locality and country_code are null', () => {
    render(<PropertyCard property={{ ...base, locality: null, country_code: null }} />);
    // No "," separator text should appear
    expect(screen.queryByText(/, /)).toBeNull();
  });

  it('renders bedroom and bathroom counts', () => {
    render(<PropertyCard property={base} />);
    expect(screen.getByText('2 beds')).toBeInTheDocument();
    expect(screen.getByText('1 bath')).toBeInTheDocument();
  });

  it('uses singular "bed" for 1 bedroom', () => {
    render(<PropertyCard property={{ ...base, num_bedrooms: 1 }} />);
    expect(screen.getByText('1 bed')).toBeInTheDocument();
  });

  it('renders area in m²', () => {
    render(<PropertyCard property={base} />);
    expect(screen.getByText('65 m²')).toBeInTheDocument();
  });

  it('shows the listing type badge', () => {
    render(<PropertyCard property={base} />);
    expect(screen.getByText('For rent')).toBeInTheDocument();
  });

  it('shows "For sale" badge for sale listings', () => {
    render(<PropertyCard property={{ ...base, listing_type: 'sale' }} />);
    expect(screen.getByText('For sale')).toBeInTheDocument();
  });

  it('renders a formatted price', () => {
    render(<PropertyCard property={base} />);
    // Intl.NumberFormat produces something like "€1,200/mo"
    expect(screen.getByText(/1[,.]?200/)).toBeInTheDocument();
    expect(screen.getByText(/\/mo/)).toBeInTheDocument();
  });

  it('shows an image element when first_image_id is set', () => {
    render(<PropertyCard property={base} />);
    expect(screen.getByRole('img', { name: 'Sunny Loft' })).toBeInTheDocument();
  });

  it('shows "No image" placeholder when first_image_id is null', () => {
    render(<PropertyCard property={{ ...base, first_image_id: null }} />);
    expect(screen.queryByRole('img')).toBeNull();
    expect(screen.getByText(/no image/i)).toBeInTheDocument();
  });

  it('hides stats that are null', () => {
    render(<PropertyCard property={{ ...base, num_bedrooms: null, num_bathrooms: null, area_sqm: null }} />);
    expect(screen.queryByText(/bed/)).toBeNull();
    expect(screen.queryByText(/bath/)).toBeNull();
    expect(screen.queryByText(/m²/)).toBeNull();
  });
});
