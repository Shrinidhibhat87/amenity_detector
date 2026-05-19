import { notFound } from 'next/navigation';
import Image from 'next/image';
import Link from 'next/link';
import type { Metadata } from 'next';
import { ApiError, getImageUrl, getProperty, shouldUnoptimizeApiImages } from '@/lib/api';
import type { PropertyDetail } from '@/lib/schemas';
import { Chip } from '@/components/ui';

// Next.js 15: params is a Promise in async Server Components.
type Props = { params: Promise<{ id: string }> };

// ── generateMetadata ──────────────────────────────────────────────────────────
// Runs before the page renders. Produces <title>, <meta description>, and
// Open Graph tags specific to this listing. Falls back gracefully if the API
// is unreachable (avoids blocking the build).

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  try {
    const p = await getProperty(id);
    const location = [p.locality, p.country_code].filter(Boolean).join(', ');
    const typeLabel = p.listing_type === 'rent' ? 'For rent' : p.listing_type === 'sale' ? 'For sale' : '';
    const descFallback = [typeLabel, location ? `in ${location}` : ''].filter(Boolean).join(' ');
    const ogImage = p.images[0]?.id != null ? getImageUrl(p.images[0].id) : undefined;

    return {
      title: `${p.name} — Amenity Detector`,
      description: p.description ?? descFallback,
      openGraph: {
        title: p.name,
        description: p.description ?? descFallback,
        ...(ogImage != null && { images: [ogImage] }),
      },
    };
  } catch {
    return { title: 'Property — Amenity Detector' };
  }
}

// ── JSON-LD builder ───────────────────────────────────────────────────────────
// Schema.org RealEstateListing so search engines and LLM agents can understand
// the listing structure without parsing HTML.

function buildJsonLd(p: PropertyDetail): Record<string, unknown> {
  const images = p.images.map((img) => getImageUrl(img.id));
  const presentAmenities = p.images
    .flatMap((img) => img.amenities.filter((a) => a.is_present).map((a) => a.amenity_name))
    .filter((name, i, arr) => arr.indexOf(name) === i); // deduplicate

  return {
    '@context': 'https://schema.org',
    '@type': 'RealEstateListing',
    name: p.name,
    ...(p.description != null && { description: p.description }),
    ...(images.length > 0 && { image: images }),
    ...(p.price != null && {
      offers: {
        '@type': 'Offer',
        price: p.price,
        priceCurrency: p.currency ?? 'EUR',
      },
    }),
    ...(p.latitude != null &&
      p.longitude != null && {
        geo: {
          '@type': 'GeoCoordinates',
          latitude: p.latitude,
          longitude: p.longitude,
        },
      }),
    ...(presentAmenities.length > 0 && {
      amenityFeature: presentAmenities.map((name) => ({
        '@type': 'LocationFeatureSpecification',
        name,
        value: true,
      })),
    }),
    ...(p.num_bedrooms != null && { numberOfRooms: p.num_bedrooms }),
    ...(p.area_sqm != null && {
      floorSize: {
        '@type': 'QuantitativeValue',
        value: p.area_sqm,
        unitCode: 'MTK',
      },
    }),
  };
}

// ── Page component ────────────────────────────────────────────────────────────

export default async function PropertyDetailPage({ params }: Props) {
  const { id } = await params;

  let property: PropertyDetail;
  try {
    property = await getProperty(id);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) notFound();
    throw err;
  }

  const jsonLd = buildJsonLd(property);
  const location = [property.locality, property.country_code].filter(Boolean).join(', ');

  // Group amenities by room — only present ones, deduplicated.
  const amenitiesByRoom = new Map<string, string[]>();
  for (const img of property.images) {
    for (const amenity of img.amenities) {
      if (!amenity.is_present) continue;
      const room = amenity.room_type ?? 'General';
      const existing = amenitiesByRoom.get(room) ?? [];
      if (!existing.includes(amenity.amenity_name)) {
        amenitiesByRoom.set(room, [...existing, amenity.amenity_name]);
      }
    }
  }

  const formattedPrice =
    property.price != null
      ? new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: property.currency ?? 'EUR',
          maximumFractionDigits: 0,
        }).format(property.price)
      : null;

  return (
    <>
      {/* Inject JSON-LD structured data for SEO / LLM agents */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <main className="flex-1 px-6 py-12 max-w-5xl mx-auto w-full">
        {/* Breadcrumb */}
        <nav className="mb-8">
          <Link
            href="/browse"
            className="font-mono text-xs uppercase tracking-widest text-ink-muted hover:text-ink transition-colors"
          >
            ← Browse
          </Link>
        </nav>

        {/* Header */}
        <header className="mb-8">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              {property.listing_type != null && (
                <p className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted mb-2">
                  {property.listing_type === 'rent' ? 'For rent' : 'For sale'}
                </p>
              )}
              <h1 className="font-display text-4xl text-ink">{property.name}</h1>
              {location.length > 0 && (
                <p className="mt-1 font-mono text-sm text-ink-muted">{location}</p>
              )}
            </div>
            {formattedPrice != null && (
              <div className="text-right shrink-0">
                <p className="text-2xl font-semibold text-accent">
                  {formattedPrice}
                  {property.price_period === 'monthly' && (
                    <span className="text-base font-normal text-ink-soft">/mo</span>
                  )}
                  {property.price_period === 'weekly' && (
                    <span className="text-base font-normal text-ink-soft">/wk</span>
                  )}
                </p>
              </div>
            )}
          </div>
        </header>

        {/* Image gallery — first image is wide, rest are square tiles */}
        {property.images.length > 0 && (
          <section className="mb-10">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {property.images.slice(0, 6).map((img, i) => (
                <div
                  key={img.id}
                  className={`relative overflow-hidden rounded-xl bg-surface-alt ${
                    i === 0 ? 'col-span-2 aspect-video' : 'aspect-square'
                  }`}
                >
                  <Image
                    src={getImageUrl(img.id)}
                    alt={img.alt_text ?? `Photo ${i + 1} of ${property.name}`}
                    fill
                    sizes="(max-width: 768px) 50vw, 33vw"
                    className="object-cover"
                    priority={i === 0}
                    unoptimized={shouldUnoptimizeApiImages()}
                  />
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Two-column layout: description + amenities left, metadata right */}
        <div className="grid md:grid-cols-3 gap-10">
          <div className="md:col-span-2 space-y-8">
            {property.description != null && (
              <section>
                <h2 className="font-mono text-xs uppercase tracking-widest text-ink-muted mb-3">
                  Description
                </h2>
                <p className="text-ink-soft leading-relaxed whitespace-pre-line">
                  {property.description}
                </p>
              </section>
            )}

            {amenitiesByRoom.size > 0 && (
              <section>
                <h2 className="font-mono text-xs uppercase tracking-widest text-ink-muted mb-4">
                  Detected amenities
                </h2>
                <div className="space-y-4">
                  {[...amenitiesByRoom.entries()].map(([room, amenities]) => (
                    <div key={room}>
                      <p className="text-sm font-medium text-ink mb-2">{room}</p>
                      <div className="flex flex-wrap gap-2">
                        {amenities.map((a) => (
                          <Chip key={a}>{a}</Chip>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>

          {/* Metadata sidebar */}
          <aside>
            <div className="rounded-2xl border border-border p-5 space-y-3 sticky top-6">
              <h2 className="font-mono text-xs uppercase tracking-widest text-ink-muted">
                Details
              </h2>
              {property.property_type != null && (
                <MetaRow label="Type" value={property.property_type} />
              )}
              {property.furnishing != null && (
                <MetaRow label="Furnished" value={property.furnishing.replace('_', ' ')} />
              )}
              {property.num_bedrooms != null && (
                <MetaRow label="Bedrooms" value={String(property.num_bedrooms)} />
              )}
              {property.num_bathrooms != null && (
                <MetaRow label="Bathrooms" value={String(property.num_bathrooms)} />
              )}
              {property.area_sqm != null && (
                <MetaRow label="Area" value={`${Math.round(property.area_sqm)} m²`} />
              )}
              {property.available_from != null && (
                <MetaRow label="Available" value={property.available_from} />
              )}
            </div>
          </aside>
        </div>
      </main>
    </>
  );
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-sm gap-2">
      <span className="text-ink-muted shrink-0">{label}</span>
      <span className="text-ink text-right capitalize">{value}</span>
    </div>
  );
}
