/**
 * schema.org JSON-LD builders for property listings.
 *
 * Emits one of:
 *   - Accommodation (or Apartment / House / etc. when property_type is set)
 *     for rent listings — the type used by Booking.com and Airbnb.
 *   - RealEstateListing for sale listings — the type used by Zillow / Realtor.
 *
 * Plus a BreadcrumbList for SERP breadcrumb display.
 *
 * Builders never emit `null` or `undefined`; missing optional fields are
 * dropped from the output so the rendered JSON is the smallest valid
 * document Google's rich-result validator will accept.
 */

import type { PropertyDetail, PropertyImage } from './schemas';
import { propertyUrl } from './property-url';

// ── Public types ─────────────────────────────────────────────────────────────

interface PostalAddress {
  '@type': 'PostalAddress';
  addressLocality?: string;
  postalCode?: string;
  addressCountry?: string;
}

interface GeoCoordinates {
  '@type': 'GeoCoordinates';
  latitude: number;
  longitude: number;
}

interface QuantitativeValue {
  '@type': 'QuantitativeValue';
  value: number;
  unitCode: string;
}

interface Offer {
  '@type': 'Offer';
  price: number;
  priceCurrency: string;
  availability: string;
  url: string;
}

interface LocationFeatureSpecification {
  '@type': 'LocationFeatureSpecification';
  name: string;
  value: boolean;
}

export interface Accommodation {
  '@context': 'https://schema.org';
  '@type': 'Accommodation' | 'Apartment' | 'House' | 'SingleFamilyResidence';
  name: string;
  url: string;
  description?: string;
  image?: string[];
  address?: PostalAddress;
  geo?: GeoCoordinates;
  numberOfRooms?: number;
  numberOfBathroomsTotal?: number;
  floorSize?: QuantitativeValue;
  amenityFeature?: LocationFeatureSpecification[];
}

export interface RealEstateListing {
  '@context': 'https://schema.org';
  '@type': 'RealEstateListing';
  name: string;
  url: string;
  description?: string;
  image?: string[];
  address?: PostalAddress;
  geo?: GeoCoordinates;
  numberOfRooms?: number;
  numberOfBathroomsTotal?: number;
  floorSize?: QuantitativeValue;
  amenityFeature?: LocationFeatureSpecification[];
  offers?: Offer;
}

export interface BreadcrumbList {
  '@context': 'https://schema.org';
  '@type': 'BreadcrumbList';
  itemListElement: Array<{
    '@type': 'ListItem';
    position: number;
    name: string;
    item: string;
  }>;
}

export type PropertyJsonLd = Accommodation | RealEstateListing | BreadcrumbList;

// ── Shared helpers ───────────────────────────────────────────────────────────

function toFiniteNumber(value: string | number | null | undefined): number | null {
  if (value == null) return null;
  const n = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

function imageUrls(images: PropertyImage[], siteUrl: string): string[] {
  // Sort by display_order so the primary / first image is listed first.
  const sorted = [...images].sort((a, b) => a.display_order - b.display_order);
  return sorted.map((img) => `${siteUrl}/api/images/${img.id}`);
}

function buildAddress(p: PropertyDetail): PostalAddress | undefined {
  if (p.locality == null && p.postal_code == null && p.country_code == null) {
    return undefined;
  }
  const addr: PostalAddress = { '@type': 'PostalAddress' };
  if (p.locality != null) addr.addressLocality = p.locality;
  if (p.postal_code != null) addr.postalCode = p.postal_code;
  if (p.country_code != null) addr.addressCountry = p.country_code;
  return addr;
}

function buildGeo(p: PropertyDetail): GeoCoordinates | undefined {
  // Prefer the listing's own coordinate; fall back to the locality insight's
  // geocoded centre so a property that only gave a PIN/street still emits geo.
  const lat = toFiniteNumber(p.latitude) ?? toFiniteNumber(p.locality_insight?.latitude);
  const lng = toFiniteNumber(p.longitude) ?? toFiniteNumber(p.locality_insight?.longitude);
  if (lat == null || lng == null) return undefined;
  return { '@type': 'GeoCoordinates', latitude: lat, longitude: lng };
}

function buildFloorSize(p: PropertyDetail): QuantitativeValue | undefined {
  const area = toFiniteNumber(p.area_sqm);
  if (area == null) return undefined;
  // UN/CEFACT code MTK = square metre. Google's validator requires unitCode,
  // not the friendlier `unitText`.
  return { '@type': 'QuantitativeValue', value: area, unitCode: 'MTK' };
}

function buildAmenityFeatures(p: PropertyDetail): LocationFeatureSpecification[] | undefined {
  const seen = new Set<string>();
  const features: LocationFeatureSpecification[] = [];
  for (const img of p.images) {
    for (const amenity of img.amenities) {
      if (!amenity.is_present) continue;
      if (seen.has(amenity.amenity_name)) continue;
      seen.add(amenity.amenity_name);
      features.push({
        '@type': 'LocationFeatureSpecification',
        name: amenity.amenity_name,
        value: true,
      });
    }
  }
  return features.length > 0 ? features : undefined;
}

function accommodationType(p: PropertyDetail): Accommodation['@type'] {
  switch (p.property_type) {
    case 'apartment':
      return 'Apartment';
    case 'house':
      return 'House';
    case 'villa':
      return 'SingleFamilyResidence';
    default:
      return 'Accommodation';
  }
}

// ── buildAccommodation ───────────────────────────────────────────────────────

export function buildAccommodation(
  p: PropertyDetail,
  siteUrl: string,
): Accommodation {
  const ld: Accommodation = {
    '@context': 'https://schema.org',
    '@type': accommodationType(p),
    name: p.name,
    url: propertyUrl(p, siteUrl),
  };
  if (p.description != null) ld.description = p.description;
  if (p.images.length > 0) ld.image = imageUrls(p.images, siteUrl);
  const address = buildAddress(p);
  if (address != null) ld.address = address;
  const geo = buildGeo(p);
  if (geo != null) ld.geo = geo;
  if (p.num_bedrooms != null) ld.numberOfRooms = p.num_bedrooms;
  if (p.num_bathrooms != null) ld.numberOfBathroomsTotal = p.num_bathrooms;
  const floorSize = buildFloorSize(p);
  if (floorSize != null) ld.floorSize = floorSize;
  const features = buildAmenityFeatures(p);
  if (features != null) ld.amenityFeature = features;
  return ld;
}

// ── buildRealEstateListing ───────────────────────────────────────────────────

export function buildRealEstateListing(
  p: PropertyDetail,
  siteUrl: string,
): RealEstateListing {
  const url = propertyUrl(p, siteUrl);
  const ld: RealEstateListing = {
    '@context': 'https://schema.org',
    '@type': 'RealEstateListing',
    name: p.name,
    url,
  };
  if (p.description != null) ld.description = p.description;
  if (p.images.length > 0) ld.image = imageUrls(p.images, siteUrl);
  const address = buildAddress(p);
  if (address != null) ld.address = address;
  const geo = buildGeo(p);
  if (geo != null) ld.geo = geo;
  if (p.num_bedrooms != null) ld.numberOfRooms = p.num_bedrooms;
  if (p.num_bathrooms != null) ld.numberOfBathroomsTotal = p.num_bathrooms;
  const floorSize = buildFloorSize(p);
  if (floorSize != null) ld.floorSize = floorSize;
  const features = buildAmenityFeatures(p);
  if (features != null) ld.amenityFeature = features;
  const price = toFiniteNumber(p.price);
  if (price != null) {
    ld.offers = {
      '@type': 'Offer',
      price,
      priceCurrency: p.currency ?? 'EUR',
      availability: 'https://schema.org/InStock',
      url,
    };
  }
  return ld;
}

// ── buildBreadcrumbList ──────────────────────────────────────────────────────

export function buildBreadcrumbList(
  p: PropertyDetail,
  siteUrl: string,
): BreadcrumbList {
  return {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: [
      { '@type': 'ListItem', position: 1, name: 'Home', item: siteUrl },
      { '@type': 'ListItem', position: 2, name: 'Browse', item: `${siteUrl}/browse` },
      {
        '@type': 'ListItem',
        position: 3,
        name: p.name,
        item: propertyUrl(p, siteUrl),
      },
    ],
  };
}

// ── buildPropertyJsonLd (router) ─────────────────────────────────────────────

/**
 * Pick the right primary entity by listing_type and pair it with the
 * BreadcrumbList. Both blocks should be emitted on every property page so
 * Google can build a rich result AND a breadcrumb trail in the SERP.
 *
 * Defaults to Accommodation when listing_type is missing — that's the safer
 * choice because RealEstateListing without an Offer is not a valid rich
 * result, whereas Accommodation has no required price field.
 */
export function buildPropertyJsonLd(
  p: PropertyDetail,
  siteUrl: string,
): PropertyJsonLd[] {
  const primary =
    p.listing_type === 'sale'
      ? buildRealEstateListing(p, siteUrl)
      : buildAccommodation(p, siteUrl);
  return [primary, buildBreadcrumbList(p, siteUrl)];
}
