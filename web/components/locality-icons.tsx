/**
 * Line-style icons for the neighbourhood panel.
 *
 * One icon per OSM category, plus transit sub-type icons (bus / tram / rail) and
 * the German U-Bahn / S-Bahn badges riders actually recognise. Everything draws
 * in `currentColor` at 1em so the icons inherit the surrounding text colour and
 * size; the U/S badges are the one exception — they carry their own recognisable
 * tint (blue U-Bahn, green S-Bahn) because that colour *is* the signal.
 */

type IconProps = { className?: string | undefined };

const base = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.6,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  width: '1em',
  height: '1em',
};

function Supermarket({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M3 4h2l2.4 11.2a1 1 0 0 0 1 .8h7.8a1 1 0 0 0 1-.78L20 8H6" />
      <circle cx="9" cy="20" r="1.2" />
      <circle cx="17" cy="20" r="1.2" />
    </svg>
  );
}

function School({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M12 4 2 9l10 5 10-5-10-5Z" />
      <path d="M6 11v5c0 1.1 2.7 2.5 6 2.5s6-1.4 6-2.5v-5" />
    </svg>
  );
}

function Gym({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M4 9v6M7 7v10M17 7v10M20 9v6M7 12h10" />
    </svg>
  );
}

function Park({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M12 22v-5" />
      <path d="M12 17a5 5 0 0 0 4.5-7.2A4 4 0 0 0 12 3a4 4 0 0 0-4.5 6.8A5 5 0 0 0 12 17Z" />
    </svg>
  );
}

function Pharmacy({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <rect x="3.5" y="3.5" width="17" height="17" rx="3.5" />
      <path d="M12 8v8M8 12h8" />
    </svg>
  );
}

function Airport({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M21 16.5 14 13V6.5a1.5 1.5 0 0 0-3 0V13l-7 3.5V18l7-2v3l-2 1.2V21l3-.8 3 .8v-.8L13 19v-3l7 2v-1.5Z" />
    </svg>
  );
}

function Bus({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <rect x="4" y="4" width="16" height="12" rx="2" />
      <path d="M4 11h16M8 16v2M16 16v2" />
      <circle cx="8" cy="13" r="0.6" />
      <circle cx="16" cy="13" r="0.6" />
    </svg>
  );
}

function Tram({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <rect x="5" y="4" width="14" height="13" rx="2.5" />
      <path d="M5 11h14M12 4V2M8 17l-2 3M16 17l2 3" />
    </svg>
  );
}

function Rail({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <rect x="6" y="3" width="12" height="14" rx="2.5" />
      <path d="M6 11h12M9 20l2-3M15 20l-2-3" />
      <circle cx="9" cy="13.5" r="0.6" />
      <circle cx="15" cy="13.5" r="0.6" />
    </svg>
  );
}

/** U-Bahn / S-Bahn badge — the tinted square/letter riders scan for. */
function TransitBadge({ letter, tint }: { letter: string; tint: string }) {
  return (
    <span
      aria-hidden="true"
      className="inline-flex h-[1em] w-[1em] items-center justify-center rounded-[3px] text-[0.62em] font-bold leading-none text-white"
      style={{ backgroundColor: tint }}
    >
      {letter}
    </span>
  );
}

const CATEGORY_ICON: Record<string, (p: IconProps) => React.ReactElement> = {
  supermarket: Supermarket,
  school: School,
  gym: Gym,
  park: Park,
  pharmacy: Pharmacy,
  airport: Airport,
  transit: Bus, // header icon for the transit category; rows use the sub-type
};

/** Icon for an accordion category header. */
export function CategoryIcon({
  category,
  className,
}: {
  category: string;
  className?: string | undefined;
}) {
  const Icon = CATEGORY_ICON[category] ?? Bus;
  return <Icon className={className} />;
}

/** Icon for a single POI — transit POIs resolve to their sub-type, else the category icon. */
export function PoiIcon({
  category,
  transitType,
  className,
}: {
  category: string;
  transitType?: string | null | undefined;
  className?: string | undefined;
}) {
  if (category === 'transit') {
    switch (transitType) {
      case 'subway':
        return <TransitBadge letter="U" tint="#2B4C8C" />;
      case 'light_rail':
        return <TransitBadge letter="S" tint="#3F7A3F" />;
      case 'tram':
        return <Tram className={className} />;
      case 'rail':
        return <Rail className={className} />;
      default:
        return <Bus className={className} />;
    }
  }
  return <CategoryIcon category={category} className={className} />;
}
