'use client';

import { useEffect, useState, type FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import { ApiError, createProperty, listModels } from '@/lib/api';
import { useWizardStore, type ConfigInput } from '@/lib/wizard-store';
import type { ModelInfo, PropertyCreateRequest } from '@/lib/schemas';
import { Button, Input, Select } from '@/components/ui';

export default function ConfigStepPage() {
  const router = useRouter();
  const state = useWizardStore((s) => s.state);
  const setConfig = useWizardStore((s) => s.setConfig);
  const startUpload = useWizardStore((s) => s.startUpload);

  const config = state.step === 'config' ? state.config : null;

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [showMore, setShowMore] = useState(false);

  // Load available VLMs on mount. If the API is down, show a soft error so
  // the user knows why the dropdown is empty.
  useEffect(() => {
    let cancelled = false;
    void listModels()
      .then((res) => {
        if (!cancelled) setModels(res);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        setModelsError(`Could not load models: ${msg}`);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // If we're past the config step, redirect to wherever we left off.
  // Lets the user pick up a wizard run across reloads (persist middleware).
  useEffect(() => {
    if (state.step !== 'config') {
      router.replace(`/detect/${state.step}`);
    }
  }, [state.step, router]);

  if (config == null) return null;

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (config == null || config.name.trim() === '' || config.model_name === '') {
      setSubmitError('Name and model are required.');
      return;
    }
    setSubmitting(true);
    setSubmitError(null);

    try {
      const body: PropertyCreateRequest = {
        name: config.name.trim(),
        model_name: config.model_name,
        // Forward any optional listing metadata the user filled in.
        ...stripEmpty({
          listing_type: config.listing_type,
          price: config.price,
          currency: config.currency,
          price_period: config.price_period,
          num_bedrooms: config.num_bedrooms,
          num_bathrooms: config.num_bathrooms,
          area_sqm: config.area_sqm,
          property_type: config.property_type,
          furnishing: config.furnishing,
          available_from: config.available_from,
          locality: config.locality,
          postal_code: config.postal_code,
          country_code: config.country_code,
        }),
      };
      const res = await createProperty(body);
      startUpload(res.property_id);
      router.push('/detect/upload');
    } catch (err) {
      const msg =
        err instanceof ApiError
          ? `API error (${err.status}): ${err.message}`
          : err instanceof Error
            ? err.message
            : 'Failed to create property.';
      setSubmitError(msg);
      setSubmitting(false);
    }
  }

  const modelOptions = models.map((m) => ({
    value: m.name,
    label: m.available ? m.name : `${m.name} (unavailable)`,
    disabled: !m.available,
  }));

  return (
    <form onSubmit={onSubmit} className="space-y-8">
      <section className="space-y-4">
        <h1 className="font-display text-3xl text-ink">Tell us about the place</h1>
        <p className="text-ink-soft">
          Give your listing a name and pick the vision-language model that will analyse your
          photos.
        </p>

        <Input
          label="Listing name"
          name="name"
          value={config.name}
          onChange={(e) => setConfig({ name: e.target.value })}
          placeholder="Sunny one-bedroom in Kreuzberg"
          required
        />

        <Select
          label="Detection model"
          value={config.model_name}
          onChange={(e) => setConfig({ model_name: e.target.value })}
          options={[{ value: '', label: 'Select a model…' }, ...modelOptions]}
          required
        />
        {modelsError != null && (
          <p className="text-warn text-sm font-mono">{modelsError}</p>
        )}
      </section>

      <section>
        <button
          type="button"
          onClick={() => setShowMore((v) => !v)}
          className="font-mono text-xs uppercase tracking-widest text-ink-muted hover:text-ink transition-colors"
        >
          {showMore ? '− Hide listing details' : '+ Add listing details (optional)'}
        </button>

        {showMore && (
          <div className="mt-4 grid sm:grid-cols-2 gap-4">
            <Select
              label="Listing type"
              value={config.listing_type ?? ''}
              onChange={(e) =>
                setConfig({ listing_type: (e.target.value || undefined) as 'rent' | 'sale' | undefined })
              }
              options={[
                { value: '', label: '—' },
                { value: 'rent', label: 'For rent' },
                { value: 'sale', label: 'For sale' },
              ]}
            />
            <Input
              label="Price"
              type="number"
              value={config.price ?? ''}
              onChange={(e) =>
                setConfig({ price: e.target.value === '' ? undefined : Number(e.target.value) })
              }
              placeholder="1200"
            />
            <Input
              label="Currency"
              value={config.currency ?? ''}
              onChange={(e) => setConfig({ currency: e.target.value || undefined })}
              placeholder="EUR"
            />
            <Select
              label="Price period"
              value={config.price_period ?? ''}
              onChange={(e) =>
                setConfig({
                  price_period:
                    (e.target.value || undefined) as ConfigInput['price_period'],
                })
              }
              options={[
                { value: '', label: '—' },
                { value: 'monthly', label: 'Monthly' },
                { value: 'weekly', label: 'Weekly' },
                { value: 'nightly', label: 'Nightly' },
                { value: 'total', label: 'Total' },
              ]}
            />
            <Input
              label="Bedrooms"
              type="number"
              value={config.num_bedrooms ?? ''}
              onChange={(e) =>
                setConfig({
                  num_bedrooms: e.target.value === '' ? undefined : Number(e.target.value),
                })
              }
            />
            <Input
              label="Bathrooms"
              type="number"
              value={config.num_bathrooms ?? ''}
              onChange={(e) =>
                setConfig({
                  num_bathrooms: e.target.value === '' ? undefined : Number(e.target.value),
                })
              }
            />
            <Input
              label="Area (m²)"
              type="number"
              value={config.area_sqm ?? ''}
              onChange={(e) =>
                setConfig({
                  area_sqm: e.target.value === '' ? undefined : Number(e.target.value),
                })
              }
            />
            <Select
              label="Property type"
              value={config.property_type ?? ''}
              onChange={(e) =>
                setConfig({
                  property_type:
                    (e.target.value || undefined) as ConfigInput['property_type'],
                })
              }
              options={[
                { value: '', label: '—' },
                { value: 'apartment', label: 'Apartment' },
                { value: 'house', label: 'House' },
                { value: 'villa', label: 'Villa' },
                { value: 'studio', label: 'Studio' },
                { value: 'other', label: 'Other' },
              ]}
            />
            <Select
              label="Furnishing"
              value={config.furnishing ?? ''}
              onChange={(e) =>
                setConfig({
                  furnishing:
                    (e.target.value || undefined) as ConfigInput['furnishing'],
                })
              }
              options={[
                { value: '', label: '—' },
                { value: 'furnished', label: 'Furnished' },
                { value: 'semi_furnished', label: 'Semi-furnished' },
                { value: 'unfurnished', label: 'Unfurnished' },
              ]}
            />
            <Input
              label="Locality"
              value={config.locality ?? ''}
              onChange={(e) => setConfig({ locality: e.target.value || undefined })}
              placeholder="Berlin"
            />
            <Input
              label="Country (ISO-2)"
              value={config.country_code ?? ''}
              onChange={(e) =>
                setConfig({ country_code: e.target.value.toUpperCase() || undefined })
              }
              placeholder="DE"
              maxLength={2}
            />
          </div>
        )}
      </section>

      {submitError != null && (
        <p className="text-warn text-sm font-mono">{submitError}</p>
      )}

      <div className="flex items-center justify-end gap-3">
        <Button type="submit" disabled={submitting}>
          {submitting ? 'Creating…' : 'Continue → Upload'}
        </Button>
      </div>
    </form>
  );
}

/**
 * Drop undefined / empty-string entries so the JSON body sent to FastAPI
 * uses `exclude_unset=True` semantics on the server side.
 */
function stripEmpty<T extends Record<string, unknown>>(obj: T): Partial<T> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (v !== undefined && v !== '' && v !== null) out[k] = v;
  }
  return out as Partial<T>;
}
