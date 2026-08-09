import { notFound, permanentRedirect } from 'next/navigation';
import Image from 'next/image';
import Link from 'next/link';
import type { Metadata } from 'next';
import {
  ApiError,
  getImageUrl,
  getProperty,
  getPropertyBySlug,
  looksLikeUuid,
  shouldUnoptimizeApiImages,
} from '@/lib/api';
import type { PropertyDetail } from '@/lib/schemas';
import { buildPropertyJsonLd } from '@/lib/json-ld';
import { buildPropertyMetadata } from '@/lib/seo-metadata';
import { Chip } from '@/components/ui';
import { LocalityPanel } from '@/components/locality-panel';

// Next.js 15: params is a Promise in async Server Components.
// `handle` is either a database UUID (legacy bookmarks, pre-slug rows) or
// a slug (the canonical public URL). The page disambiguates at request time.
type Props = { params: Promise<{ handle: string }> };

const SITE_URL = process.env['NEXT_PUBLIC_SITE_URL'] ?? 'http://localhost:3000';

/**
 * Resolve the URL segment to a Property, treating UUIDs and slugs alike.
 *
 * When the segment is a UUID and the resolved property has a slug, the
 * caller is redirected to the slug URL (308 permanent) so search engines
 * collapse the two URLs onto the canonical one and the share-link looks
 * human-readable. Legacy rows without a slug stay reachable at their UUID.
 */
async function resolveProperty(handle: string): Promise<PropertyDetail> {
  if (looksLikeUuid(handle)) {
    const p = await getProperty(handle);
    if (p.slug != null && p.slug.length > 0) {
      // 308 — preserves method and tells the crawler this is the new home.
      permanentRedirect(`/properties/${p.slug}`);
    }
    return p;
  }
  return getPropertyBySlug(handle);
}

// ── generateMetadata ──────────────────────────────────────────────────────────
// Runs before the page renders. Produces <title>, <meta description>,
// canonical URL, OpenGraph card, and Twitter card. The full metadata shape
// is built in lib/seo-metadata.ts so it is unit-testable. Falls back to a
// generic title if the API is unreachable so `next build` is not blocked.

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { handle } = await params;
  try {
    const p = looksLikeUuid(handle)
      ? await getProperty(handle)
      : await getPropertyBySlug(handle);
    const metadata = buildPropertyMetadata(p, SITE_URL);
    // An unpublished draft is only reachable by its id, as a private preview.
    // Keep crawlers out of it even if the link leaks.
    if (p.status !== 'published') {
      return { ...metadata, robots: { index: false, follow: false } };
    }
    return metadata;
  } catch {
    return { title: 'Property — Amenity Detector' };
  }
}

// ── Page component ────────────────────────────────────────────────────────────

export default async function PropertyDetailPage({ params }: Props) {
  const { handle } = await params;

  let property: PropertyDetail;
  try {
    property = await resolveProperty(handle);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) notFound();
    throw err;
  }

  // buildPropertyJsonLd returns two blocks: the listing entity (Accommodation
  // for rent, RealEstateListing for sale) and a BreadcrumbList. Both are
  // rendered as separate <script> tags so a future per-type Google rich
  // result does not collide with the breadcrumb trail in the SERP.
  const jsonLdEntries = buildPropertyJsonLd(property, SITE_URL);
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
      {/* Inject JSON-LD structured data for SEO / LLM agents.
          One <script> per schema.org entity so Google's rich-result
          parser handles them independently. */}
      {jsonLdEntries.map((entry, i) => (
        <script
          key={i}
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(entry) }}
        />
      ))}

      <main className="flex-1 px-6 py-12 max-w-5xl mx-auto w-full">
        {property.status !== 'published' && (
          <p className="mb-6 rounded-xl border border-border bg-surface-alt px-4 py-3 font-mono text-xs text-ink-muted">
            Draft preview — this listing is not public yet.
          </p>
        )}

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

            {property.locality_insight != null && (
              <LocalityPanel insight={property.locality_insight} />
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
