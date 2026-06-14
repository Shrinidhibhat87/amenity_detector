/**
 * Single source of truth for API shapes consumed by the web app.
 *
 * Why Zod instead of bare TS types?
 *   The FastAPI backend returns JSON we cannot blindly trust at runtime — a
 *   shape mismatch (missing field, renamed enum, decimal-as-string vs number)
 *   would otherwise crash a Server Component during render. Zod gives us:
 *     1. Runtime validation at the network boundary.
 *     2. TS types derived from the schema via `z.infer<typeof X>`.
 *   One definition, two guarantees. Keep this file in sync with
 *   `api/schemas.py` — that is the upstream contract.
 *
 * Decimal-as-string note:
 *   Pydantic serialises Decimal as a JSON string ("980" not 980). We coerce
 *   to number at the boundary so React components can format with toLocale*.
 */
import { z } from 'zod';

// ── Phase 9 listing-metadata enums (mirror api/schemas.py Literals) ──────────
export const ListingType = z.enum(['rent', 'sale']);
export type ListingType = z.infer<typeof ListingType>;

export const PricePeriod = z.enum(['monthly', 'weekly', 'nightly', 'total']);
export type PricePeriod = z.infer<typeof PricePeriod>;

export const PropertyType = z.enum(['apartment', 'house', 'villa', 'studio', 'other']);
export type PropertyType = z.infer<typeof PropertyType>;

export const Furnishing = z.enum(['furnished', 'semi_furnished', 'unfurnished']);
export type Furnishing = z.infer<typeof Furnishing>;

/** ISO-3166 alpha-2, normalised to upper-case to match FastAPI StringConstraints. */
const CountryCode = z
  .string()
  .length(2)
  .transform((s) => s.toUpperCase());

/** Pydantic Decimal arrives as a JSON string. Coerce to number on the client. */
const DecimalLike = z.union([z.number(), z.string()]).transform((v) => Number(v));

const Latitude = DecimalLike.refine((n) => n >= -90 && n <= 90, 'latitude out of range');
const Longitude = DecimalLike.refine((n) => n >= -180 && n <= 180, 'longitude out of range');

// ── Detected amenity ─────────────────────────────────────────────────────────
export const DetectedAmenity = z.object({
  id: z.string(),
  amenity_name: z.string().min(1),
  room_type: z.string().nullable(),
  is_present: z.boolean(),
  /** 0..1 confidence from the VLM, or null if the user added the amenity by hand. */
  confidence: z.number().nullable(),
});
export type DetectedAmenity = z.infer<typeof DetectedAmenity>;

// ── Property image ───────────────────────────────────────────────────────────
export const PropertyImage = z.object({
  id: z.string(),
  file_path: z.string(),
  room_type: z.string().nullable(),
  amenities: z.array(DetectedAmenity),
  alt_text: z.string().nullable().default(null),
  caption: z.string().nullable().default(null),
  is_primary: z.boolean().default(false),
  display_order: z.number().int().nonnegative().default(0),
});
export type PropertyImage = z.infer<typeof PropertyImage>;

// ── Shared Phase 9 listing metadata block ────────────────────────────────────
const ListingMetadata = z.object({
  slug: z.string().nullable().default(null),
  listing_type: ListingType.nullable().default(null),
  price: DecimalLike.nullable().default(null),
  currency: z.string().nullable().default(null),
  price_period: PricePeriod.nullable().default(null),
  num_bedrooms: z.number().int().nullable().default(null),
  num_bathrooms: z.number().int().nullable().default(null),
  area_sqm: DecimalLike.nullable().default(null),
  property_type: PropertyType.nullable().default(null),
  furnishing: Furnishing.nullable().default(null),
  available_from: z.string().nullable().default(null),
  locality: z.string().nullable().default(null),
  postal_code: z.string().nullable().default(null),
  country_code: z.string().nullable().default(null),
  latitude: Latitude.nullable().default(null),
  longitude: Longitude.nullable().default(null),
  owner_email: z.string().nullable().default(null),
});

// ── PropertySummary (list endpoints) ─────────────────────────────────────────
export const PropertySummary = ListingMetadata.extend({
  id: z.string(),
  name: z.string(),
  description: z.string().nullable(),
  model_used: z.string().nullable(),
  extra_info: z.string().nullable(),
  created_at: z.string(),
  image_count: z.number().int().nonnegative().default(0),
  first_image_id: z.string().nullable().default(null),
});
export type PropertySummary = z.infer<typeof PropertySummary>;

// ── Locality insight (nested on PropertyDetail) ──────────────────────────────
// Mirrors api/schemas.py LocalityInsightSummary. The neighbourhood enrichment
// produced by the locality agent: a written "Lage" blurb, per-category POI
// counts, the raw nearby POIs, and the mandatory OSM attribution.
export const LocalityPoi = z.object({
  category: z.string(),
  name: z.string().default(''),
  latitude: z.number(),
  longitude: z.number(),
  distance_m: z.number().default(0),
  osm_type: z.string().default(''),
  osm_id: z.number().default(0),
  // Transit subtype for the "transit" category: bus/tram/subway/light_rail/rail.
  // null/absent for every other category (and for stops we couldn't classify,
  // including legacy rows persisted before subtypes existed).
  transit_type: z.string().nullish(),
});
export type LocalityPoi = z.infer<typeof LocalityPoi>;

// Request body for the locality endpoints (mirrors api/schemas.py LocalityRequest).
// PIN required; street optional; country defaults to DE; everyday radius 1–10 km.
export const LocalityRequest = z.object({
  postal_code: z.string().min(1).max(16),
  street: z.string().max(255).optional(),
  country_code: CountryCode.default('DE'),
  radius_m: z.number().int().min(1000).max(10000).default(3000),
});
export type LocalityRequest = z.infer<typeof LocalityRequest>;

export const LocalityInsight = z.object({
  display_name: z.string().nullable().default(null),
  latitude: Latitude.nullable().default(null),
  longitude: Longitude.nullable().default(null),
  radius_m: z.number().int().nullable().default(null),
  blurb: z.string().nullable().default(null),
  category_counts: z.record(z.string(), z.number()).default({}),
  pois: z.array(LocalityPoi).default([]),
  attribution: z.string(),
});
export type LocalityInsight = z.infer<typeof LocalityInsight>;

// ── PropertyDetail (single property endpoint) ────────────────────────────────
export const PropertyDetail = ListingMetadata.extend({
  id: z.string(),
  name: z.string(),
  description: z.string().nullable(),
  model_used: z.string().nullable(),
  extra_info: z.string().nullable(),
  created_at: z.string(),
  images: z.array(PropertyImage),
  // Optional so existing fixtures/legacy rows without enrichment still parse.
  locality_insight: LocalityInsight.nullable().optional(),
});
export type PropertyDetail = z.infer<typeof PropertyDetail>;

// ── Create / update request bodies ───────────────────────────────────────────
// These mirror api/schemas.py PropertyCreateRequest / PropertyUpdateRequest.
// We re-validate on the client so we never POST a body that the server will
// 422 back at us.

const _listingFields = {
  listing_type: ListingType.optional(),
  price: z.number().nonnegative().optional(),
  currency: z.string().optional(),
  price_period: PricePeriod.optional(),
  num_bedrooms: z.number().int().min(0).max(50).optional(),
  num_bathrooms: z.number().int().min(0).max(50).optional(),
  area_sqm: z.number().nonnegative().optional(),
  property_type: PropertyType.optional(),
  furnishing: Furnishing.optional(),
  available_from: z.string().optional(),
  locality: z.string().optional(),
  postal_code: z.string().optional(),
  country_code: CountryCode.optional(),
  latitude: z.number().min(-90).max(90).optional(),
  longitude: z.number().min(-180).max(180).optional(),
  owner_email: z.string().optional(),
} as const;

export const PropertyCreateRequest = z.object({
  name: z.string().min(1),
  model_name: z.string().min(1),
  extra_info: z.string().optional(),
  ..._listingFields,
});
export type PropertyCreateRequest = z.infer<typeof PropertyCreateRequest>;

export const PropertyUpdateRequest = z
  .object({
    ..._listingFields,
    // The wizard PATCHes the generated description here after the describe
    // step. Backend mirrors this field on api/schemas.py PropertyUpdateRequest.
    description: z.string().optional(),
  })
  .strict();
export type PropertyUpdateRequest = z.infer<typeof PropertyUpdateRequest>;

// ── Image PATCH body ─────────────────────────────────────────────────────────
export const ImageUpdateRequest = z
  .object({
    alt_text: z.string().optional(),
    caption: z.string().optional(),
    is_primary: z.boolean().optional(),
    display_order: z.number().int().min(0).max(999).optional(),
  })
  .strict();
export type ImageUpdateRequest = z.infer<typeof ImageUpdateRequest>;

// ── Describe endpoint ────────────────────────────────────────────────────────
export const AmenityEditItem = z.object({
  amenity_name: z.string(),
  room_type: z.string(),
  is_present: z.boolean(),
});
export type AmenityEditItem = z.infer<typeof AmenityEditItem>;

export const DescribeRequest = z.object({
  amenities: z.array(AmenityEditItem),
  model_name: z.string(),
  num_rooms: z.number().int().optional(),
  has_kitchen: z.boolean().optional(),
  has_balcony: z.boolean().optional(),
  has_living_room: z.boolean().optional(),
  hints: z.record(z.string(), z.boolean()).optional(),
});
export type DescribeRequest = z.infer<typeof DescribeRequest>;

export const DescribeResponse = z.object({ description: z.string() });
export type DescribeResponse = z.infer<typeof DescribeResponse>;

// ── Mutation response wrappers ───────────────────────────────────────────────
// These mirror api/schemas.py PropertyCreateResponse and ImageDetectionResponse.

export const PropertyCreateResponse = z.object({
  property_id: z.string(),
  message: z.string(),
  property: PropertyDetail,
});
export type PropertyCreateResponse = z.infer<typeof PropertyCreateResponse>;

export const ImageDetectionResponse = z.object({
  property_id: z.string(),
  image: PropertyImage,
});
export type ImageDetectionResponse = z.infer<typeof ImageDetectionResponse>;

// ── Model registry info ──────────────────────────────────────────────────────
// Returned by GET /api/v1/models/ — feeds the wizard's model dropdown.

export const ModelInfo = z.object({
  name: z.string(),
  available: z.boolean(),
  description: z.string(),
});
export type ModelInfo = z.infer<typeof ModelInfo>;

// ── Health check ─────────────────────────────────────────────────────────────
export const Health = z.object({
  status: z.string(),
  database: z.string(),
  version: z.string().default('1.0.0'),
});
export type Health = z.infer<typeof Health>;

// ── Client-only: natural-language search filter ──────────────────────────────
// Lives here even though FastAPI does not own it, because the URL/search-bar
// is the source of truth for the browse page. Phase 11 backend will consume
// the same shape via /api/v1/properties/search.
export const SearchFilter = z.object({
  query: z.string().default(''),
  rooms: z.union([z.literal('any'), z.number().int().min(1).max(5)]).default('any'),
  amenities: z.array(z.string()).default([]),
  sort: z.enum(['recent', 'mostAmenities', 'name']).default('recent'),
});
export type SearchFilter = z.infer<typeof SearchFilter>;
