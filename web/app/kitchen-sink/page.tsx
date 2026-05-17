'use client';

import {
  Button,
  Chip,
  Card,
  Input,
  Select,
  type ButtonVariant,
  type ButtonSize,
  type ChipTone,
} from '@/components/ui';

// Why `'use client'` here?
//   This page wires `onClick` handlers into Chip/Card for the active-state
//   showcase. Event handlers cannot cross the Server→Client boundary, so the
//   showcase opts in to client rendering. The `noindex` robots directive
//   lives in app/kitchen-sink/layout.tsx where `metadata` is legal.

const BUTTON_VARIANTS: ButtonVariant[] = ['primary', 'accent', 'ghost', 'soft', 'quiet'];
const BUTTON_SIZES: ButtonSize[] = ['sm', 'md', 'lg'];
const CHIP_TONES: ChipTone[] = ['default', 'accent', 'ok', 'warn'];

const LISTING_OPTIONS = [
  { value: 'rent', label: 'For rent' },
  { value: 'sale', label: 'For sale' },
];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-4">
      <h2 className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-muted">
        {title}
      </h2>
      <div className="flex flex-wrap items-end gap-3">{children}</div>
    </section>
  );
}

export default function KitchenSink() {
  return (
    <main className="mx-auto flex max-w-4xl flex-1 flex-col gap-12 px-6 py-16">
      <header>
        <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-muted">
          Phase A · Foundation
        </p>
        <h1 className="mt-2 font-display text-4xl text-ink">UI primitives</h1>
        <p className="mt-3 max-w-xl text-ink-soft">
          Every Button, Chip, Card, Input, and Select variant. Use this page to
          eyeball spacing, contrast, and motion against the design tokens.
        </p>
      </header>

      <Section title="Button — variants">
        {BUTTON_VARIANTS.map((v) => (
          <Button key={v} variant={v}>
            {v}
          </Button>
        ))}
        <Button disabled>disabled</Button>
      </Section>

      <Section title="Button — sizes">
        {BUTTON_SIZES.map((s) => (
          <Button key={s} size={s}>
            size {s}
          </Button>
        ))}
      </Section>

      <Section title="Chip — tones (idle / active)">
        {CHIP_TONES.flatMap((tone) => [
          <Chip key={`${tone}-idle`} tone={tone}>
            {tone}
          </Chip>,
          <Chip key={`${tone}-active`} tone={tone} active onClick={() => {}}>
            {tone} active
          </Chip>,
        ])}
      </Section>

      <Section title="Card">
        <Card className="w-72">
          <p className="font-display text-lg text-ink">Bendel 26</p>
          <p className="mt-1 text-xs text-ink-muted">Aachen · Frankenberger Viertel</p>
          <p className="mt-3 text-sm text-ink-soft">
            3 rooms · 78 m² · €980 / month
          </p>
        </Card>
        <Card hover onClick={() => {}} className="w-72">
          <p className="font-display text-lg text-ink">Loft Hafenkante</p>
          <p className="mt-1 text-xs text-ink-muted">Hamburg · HafenCity</p>
          <p className="mt-3 text-sm text-ink-soft">
            2 rooms · 64 m² · €1290 / month
          </p>
        </Card>
      </Section>

      <Section title="Input + Select">
        <div className="grid w-full max-w-md grid-cols-1 gap-4 sm:grid-cols-2">
          <Input label="Property name" name="name" placeholder="e.g. Bendel 26" />
          <Input
            label="Email"
            name="email"
            type="email"
            error="Not a valid address"
            defaultValue="not-an-email"
          />
          <Select
            label="Listing"
            name="listing"
            options={LISTING_OPTIONS}
            defaultValue="rent"
          />
          <Input label="Price" name="price" type="number" hint="EUR per month" />
        </div>
      </Section>
    </main>
  );
}
