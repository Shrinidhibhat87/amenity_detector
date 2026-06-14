/**
 * Presentational panel for the neighbourhood ("Lage") enrichment.
 *
 * Renders the synthesized blurb, per-category nearby-POI counts, a short list
 * of the closest places, and the mandatory OpenStreetMap attribution. Pure and
 * server-renderable — it takes a parsed LocalityInsight and emits markup, no
 * data fetching of its own.
 */

import type { LocalityInsight } from '@/lib/schemas';

// Human labels for the fixed OSM category keys (mirrors core/locality/overpass.py).
const CATEGORY_LABELS: Record<string, string> = {
  school: 'Schools',
  gym: 'Gyms',
  supermarket: 'Supermarkets',
  park: 'Parks',
  transit: 'Public transport',
  pharmacy: 'Pharmacies',
};

function label(category: string): string {
  return CATEGORY_LABELS[category] ?? category;
}

function formatDistance(metres: number): string {
  if (metres < 1000) return `${Math.round(metres)} m`;
  return `${(metres / 1000).toFixed(1)} km`;
}

export function LocalityPanel({ insight }: { insight: LocalityInsight }) {
  const counts = Object.entries(insight.category_counts).filter(([, n]) => n > 0);
  // Nearest few named POIs across categories, closest first.
  const nearest = [...insight.pois]
    .filter((p) => p.name.length > 0)
    .sort((a, b) => a.distance_m - b.distance_m)
    .slice(0, 6);

  return (
    <section aria-label="Neighbourhood">
      <h2 className="font-mono text-xs uppercase tracking-widest text-ink-muted mb-3">
        Neighbourhood
      </h2>

      {insight.blurb != null && insight.blurb.length > 0 && (
        <p className="text-ink-soft leading-relaxed whitespace-pre-line mb-4">{insight.blurb}</p>
      )}

      {counts.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {counts.map(([category, n]) => (
            <span
              key={category}
              className="font-mono text-[11px] px-2.5 py-1 rounded-full bg-surface-alt text-ink-soft"
            >
              {label(category)}: {n}
            </span>
          ))}
        </div>
      )}

      {nearest.length > 0 && (
        <ul className="space-y-1.5 mb-4">
          {nearest.map((poi) => (
            <li
              key={`${poi.osm_type}-${poi.osm_id}`}
              className="flex items-baseline justify-between gap-3 text-sm"
            >
              <span className="text-ink truncate">{poi.name}</span>
              <span className="font-mono text-xs text-ink-muted shrink-0">
                {label(poi.category)} · {formatDistance(poi.distance_m)}
              </span>
            </li>
          ))}
        </ul>
      )}

      <p className="font-mono text-[11px] text-ink-muted">{insight.attribution}</p>
    </section>
  );
}
