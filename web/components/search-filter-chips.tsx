import { Chip } from '@/components/ui';

/**
 * Display strip rendered on /search.
 *
 * The chips visualise which signals the regex parser pulled from the user's
 * NL query. They are display-only: the backend now owns parsing + filtering,
 * so editing individual chips would require sending structured overrides on
 * top of the raw query — not yet wired. To refine results, edit the query
 * text itself.
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
        <Chip tone="accent" active>
          {filters.listing_type === 'rent' ? 'For rent' : 'For sale'}
        </Chip>
      )}
      {filters.num_bedrooms != null && (
        <Chip tone="accent" active>
          {filters.num_bedrooms} bed
        </Chip>
      )}
      {filters.price_max != null && (
        <Chip tone="accent" active>
          ≤ {filters.price_max.toLocaleString()} {filters.currency ?? ''}
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
