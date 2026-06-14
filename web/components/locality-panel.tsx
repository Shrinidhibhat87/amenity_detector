'use client';

/**
 * Neighbourhood ("Lage") panel — accordion of nearby categories.
 *
 * Each category is a collapsible row: icon, label, the true count at the search
 * radius, and a chevron. Expanding a row reveals the closest few places (name ·
 * distance in km). Public-transport rows show the stop's sub-type icon
 * (bus / tram / U-Bahn / S-Bahn / rail). The written blurb sits *below* the
 * structured rows, and the mandatory OpenStreetMap attribution closes the panel.
 *
 * Client component because the accordion is interactive; it still takes a parsed
 * LocalityInsight and fetches nothing itself.
 */

import { useState } from 'react';
import type { LocalityInsight, LocalityPoi } from '@/lib/schemas';
import { CategoryIcon, PoiIcon } from '@/components/locality-icons';

// Display order + labels for the OSM category keys (mirrors core/locality).
const CATEGORY_ORDER = [
  'supermarket',
  'school',
  'gym',
  'park',
  'transit',
  'pharmacy',
  'airport',
] as const;

const CATEGORY_LABELS: Record<string, string> = {
  supermarket: 'Supermarkets',
  school: 'Schools',
  gym: 'Gyms',
  park: 'Parks',
  transit: 'Public transport',
  pharmacy: 'Pharmacies',
  airport: 'Airports',
};

function label(category: string): string {
  return CATEGORY_LABELS[category] ?? category;
}

/** Distances are shown in km throughout: one decimal under 10 km, whole above. */
function formatKm(metres: number): string {
  const km = metres / 1000;
  return km >= 10 ? `${Math.round(km)} km` : `${km.toFixed(1)} km`;
}

function poisFor(insight: LocalityInsight, category: string): LocalityPoi[] {
  return insight.pois
    .filter((p) => p.category === category)
    .sort((a, b) => a.distance_m - b.distance_m);
}

function CategoryRow({
  insight,
  category,
  count,
  open,
  onToggle,
}: {
  insight: LocalityInsight;
  category: string;
  count: number;
  open: boolean;
  onToggle: () => void;
}) {
  const sample = poisFor(insight, category);
  const remaining = count - sample.length;

  return (
    <div className="border-b border-border last:border-b-0">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex w-full items-center gap-3 py-3 text-left transition-colors hover:bg-surface-alt/50"
      >
        <span className="text-accent text-[18px] shrink-0">
          <CategoryIcon category={category} />
        </span>
        <span className="flex-1 text-sm text-ink">{label(category)}</span>
        <span className="font-mono text-xs text-ink-muted tabular-nums">{count}</span>
        <svg
          viewBox="0 0 24 24"
          width="14"
          height="14"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
          className={`text-ink-faint transition-transform duration-200 ${open ? 'rotate-90' : ''}`}
        >
          <path d="m9 18 6-6-6-6" />
        </svg>
      </button>

      {open && (
        <ul className="ad-fade-up space-y-1.5 pb-3 pl-[30px] pr-1">
          {sample.length === 0 && (
            <li className="text-xs text-ink-muted">Names not available in OpenStreetMap.</li>
          )}
          {sample.map((poi) => (
            <li
              key={`${poi.osm_type}-${poi.osm_id}`}
              className="flex items-baseline justify-between gap-3 text-sm"
            >
              <span className="flex items-center gap-2 min-w-0">
                <span className="text-ink-soft text-[15px] shrink-0 self-center">
                  <PoiIcon category={poi.category} transitType={poi.transit_type} />
                </span>
                <span className="text-ink truncate">{poi.name || 'Unnamed'}</span>
              </span>
              <span className="font-mono text-xs text-ink-muted shrink-0 tabular-nums">
                {formatKm(poi.distance_m)}
              </span>
            </li>
          ))}
          {remaining > 0 && (
            <li className="font-mono text-[11px] text-ink-faint pt-0.5">+{remaining} more nearby</li>
          )}
        </ul>
      )}
    </div>
  );
}

export function LocalityPanel({ insight }: { insight: LocalityInsight }) {
  const present = CATEGORY_ORDER.filter((c) => (insight.category_counts[c] ?? 0) > 0);
  // Single-open accordion: start with the densest category expanded.
  const densest = present.reduce<string | null>(
    (best, c) =>
      best == null || (insight.category_counts[c] ?? 0) > (insight.category_counts[best] ?? 0)
        ? c
        : best,
    null,
  );
  const [openCategory, setOpenCategory] = useState<string | null>(densest);

  const radiusKm = insight.radius_m != null ? insight.radius_m / 1000 : null;

  return (
    <section aria-label="Neighbourhood">
      <div className="mb-1 flex items-baseline justify-between">
        <h2 className="font-mono text-xs uppercase tracking-widest text-ink-muted">Neighbourhood</h2>
        {radiusKm != null && (
          <span className="font-mono text-[11px] text-ink-faint">
            within {radiusKm % 1 === 0 ? radiusKm : radiusKm.toFixed(1)} km
          </span>
        )}
      </div>

      {present.length > 0 ? (
        <div className="mb-4 border-t border-border">
          {present.map((category) => (
            <CategoryRow
              key={category}
              insight={insight}
              category={category}
              count={insight.category_counts[category] ?? 0}
              open={openCategory === category}
              onToggle={() =>
                setOpenCategory((cur) => (cur === category ? null : category))
              }
            />
          ))}
        </div>
      ) : (
        <p className="mb-4 text-sm text-ink-muted">No notable amenities found nearby.</p>
      )}

      {insight.blurb != null && insight.blurb.length > 0 && (
        <p className="mb-4 leading-relaxed text-ink-soft whitespace-pre-line">{insight.blurb}</p>
      )}

      <p className="font-mono text-[11px] text-ink-muted">{insight.attribution}</p>
    </section>
  );
}
