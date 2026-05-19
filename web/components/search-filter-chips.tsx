'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { Chip } from '@/components/ui';

/**
 * Editable filter strip rendered on /search.
 *
 * The parser turns the user's NL query into a baseline set of filters. The
 * user can refine them by clicking a chip to remove that dimension — the
 * remove gesture sets an explicit override in the URL (`<param>=`) so a
 * subsequent reload remembers the choice rather than re-parsing the query.
 *
 * (room, amenity) chips are read-only this phase. The backend `/search`
 * endpoint only AND-filters by amenity name and has no room dimension, so
 * editing them would have no effect until Phase 11's hybrid endpoint lands.
 */
export interface EffectiveFilters {
  listing_type?: 'rent' | 'sale' | undefined;
  num_bedrooms?: number | undefined;
  price_max?: number | undefined;
  currency?: string | undefined;
  amenities: string[];
  room_amenities: Array<{ room: string; amenity: string }>;
}

interface Props {
  filters: EffectiveFilters;
}

export function SearchFilterChips({ filters }: Props) {
  const router = useRouter();
  const searchParams = useSearchParams();

  function clearParam(name: string): void {
    // Set the param to empty so the server reads it as "explicitly cleared"
    // and skips the parser fallback for that dimension.
    const next = new URLSearchParams(searchParams.toString());
    next.set(name, '');
    router.push(`/search?${next.toString()}`);
  }

  const hasAnyChip =
    filters.listing_type != null ||
    filters.num_bedrooms != null ||
    filters.price_max != null ||
    filters.amenities.length > 0 ||
    filters.room_amenities.length > 0;

  if (!hasAnyChip) return null;

  return (
    <div className="mt-4 flex flex-wrap gap-2">
      {filters.listing_type != null && (
        <Chip tone="accent" active onClick={() => clearParam('listing_type')}>
          {filters.listing_type === 'rent' ? 'For rent' : 'For sale'} ✕
        </Chip>
      )}
      {filters.num_bedrooms != null && (
        <Chip tone="accent" active onClick={() => clearParam('num_bedrooms')}>
          {filters.num_bedrooms} bed ✕
        </Chip>
      )}
      {filters.price_max != null && (
        <Chip tone="accent" active onClick={() => clearParam('price_max')}>
          ≤ {filters.price_max.toLocaleString()} {filters.currency ?? ''} ✕
        </Chip>
      )}
      {filters.amenities.map((a) => (
        <Chip key={a} active>
          {a}
        </Chip>
      ))}
      {filters.room_amenities.map((ra) => (
        <Chip key={`${ra.room}-${ra.amenity}`} tone="ok" active>
          {ra.amenity} in {ra.room}
        </Chip>
      ))}
    </div>
  );
}
