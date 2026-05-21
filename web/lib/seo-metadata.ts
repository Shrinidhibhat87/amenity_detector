/**
 * Next.js Metadata builder for property listing pages.
 *
 * Centralised so the metadata block is unit-testable: tests can pass a
 * fixture PropertyDetail without spinning up a Server Component.
 *
 * Emits canonical URL, OpenGraph card, Twitter card, and a sensible
 * synthesised description when the property has none of its own.
 */

import type { Metadata } from 'next';
import type { PropertyDetail } from './schemas';
import { propertyUrl } from './property-url';

function listingTypeLabel(p: PropertyDetail): string {
  if (p.listing_type === 'rent') return 'For rent';
  if (p.listing_type === 'sale') return 'For sale';
  return '';
}

function locationLabel(p: PropertyDetail): string {
  return [p.locality, p.country_code].filter((s): s is string => s != null).join(', ');
}

function synthesisedDescription(p: PropertyDetail): string {
  const type = listingTypeLabel(p);
  const loc = locationLabel(p);
  return [type, loc ? `in ${loc}` : ''].filter(Boolean).join(' ').trim();
}

function imageUrl(siteUrl: string, imageId: string): string {
  return `${siteUrl}/api/images/${imageId}`;
}

export function buildPropertyMetadata(
  p: PropertyDetail,
  siteUrl: string,
): Metadata {
  const canonical = propertyUrl(p, siteUrl);
  const description = p.description ?? (synthesisedDescription(p) || 'Property listing.');
  const primary = p.images[0];

  // OG / Twitter image descriptors. Twitter accepts a plain URL string
  // (the SDK normalises it to <meta name="twitter:image">), while OG
  // wants the OGImage descriptor for alt-text support.
  const ogImage =
    primary != null
      ? { url: imageUrl(siteUrl, primary.id), alt: primary.alt_text ?? p.name }
      : null;
  const twitterImage = primary != null ? imageUrl(siteUrl, primary.id) : null;

  const metadata: Metadata = {
    title: `${p.name} — Amenity Detector`,
    description,
    alternates: { canonical },
    openGraph: {
      title: p.name,
      description,
      url: canonical,
      type: 'website',
      ...(ogImage != null && { images: [ogImage] }),
    },
    twitter: {
      card: twitterImage != null ? 'summary_large_image' : 'summary',
      title: p.name,
      description,
      ...(twitterImage != null && { images: [twitterImage] }),
    },
  };
  return metadata;
}
