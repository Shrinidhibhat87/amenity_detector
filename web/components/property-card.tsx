import Image from 'next/image';
import Link from 'next/link';
import { getImageUrl, shouldUnoptimizeApiImages } from '@/lib/api';
import type { PropertySummary } from '@/lib/schemas';

function formatPrice(
  price: number | null,
  currency: string | null,
  period: string | null,
): string | null {
  if (price == null) return null;
  const curr = currency ?? 'EUR';
  const formatted = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: curr,
    maximumFractionDigits: 0,
  }).format(price);
  if (period === 'monthly') return `${formatted}/mo`;
  if (period === 'weekly') return `${formatted}/wk`;
  if (period === 'nightly') return `${formatted}/night`;
  return formatted;
}

interface PropertyCardProps {
  property: PropertySummary;
}

export function PropertyCard({ property }: PropertyCardProps) {
  const {
    id,
    name,
    listing_type,
    price,
    currency,
    price_period,
    locality,
    country_code,
    num_bedrooms,
    num_bathrooms,
    area_sqm,
    first_image_id,
    image_count,
  } = property;

  const imageUrl = first_image_id != null ? getImageUrl(first_image_id) : null;
  const priceStr = formatPrice(price, currency, price_period);
  const locationParts = [locality, country_code].filter((s): s is string => s != null);
  const location = locationParts.join(', ');

  return (
    <Link
      href={`/properties/${id}`}
      className="group block rounded-2xl overflow-hidden border border-border hover:border-border-strong transition-colors bg-surface"
    >
      {/* Thumbnail */}
      <div className="relative aspect-[4/3] bg-surface-alt overflow-hidden">
        {imageUrl != null ? (
          <Image
            src={imageUrl}
            alt={name}
            fill
            sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
            className="object-cover group-hover:scale-[1.02] transition-transform duration-300"
            unoptimized={shouldUnoptimizeApiImages()}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="font-mono text-xs text-ink-muted uppercase tracking-widest">
              No image
            </span>
          </div>
        )}
        {listing_type != null && (
          <span className="absolute top-3 left-3 font-mono text-[10px] uppercase tracking-widest px-2 py-0.5 rounded-full bg-bg/90 text-ink">
            {listing_type === 'rent' ? 'For rent' : 'For sale'}
          </span>
        )}
        {image_count > 1 && (
          <span className="absolute top-3 right-3 font-mono text-[10px] text-ink-muted bg-bg/80 px-2 py-0.5 rounded-full">
            +{image_count - 1} photos
          </span>
        )}
      </div>

      {/* Body */}
      <div className="p-4 space-y-1.5">
        <h3 className="font-medium text-ink truncate">{name}</h3>
        {location.length > 0 && (
          <p className="font-mono text-xs text-ink-muted">{location}</p>
        )}

        {/* Stats row */}
        <div className="flex items-center gap-3 font-mono text-xs text-ink-soft pt-0.5">
          {num_bedrooms != null && (
            <span>
              {num_bedrooms} {num_bedrooms === 1 ? 'bed' : 'beds'}
            </span>
          )}
          {num_bathrooms != null && (
            <span>
              {num_bathrooms} {num_bathrooms === 1 ? 'bath' : 'baths'}
            </span>
          )}
          {area_sqm != null && <span>{Math.round(area_sqm)} m²</span>}
        </div>

        {priceStr != null && (
          <p className="text-base font-semibold text-accent pt-1">{priceStr}</p>
        )}
      </div>
    </Link>
  );
}
