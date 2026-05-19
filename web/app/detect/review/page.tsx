'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import {
  ApiError,
  describeProperty,
  getImageUrl,
  patchImage,
  shouldUnoptimizeApiImages,
} from '@/lib/api';
import { useWizardStore, type AmenityItem, type UploadedImage } from '@/lib/wizard-store';
import { Button } from '@/components/ui';

export default function ReviewStepPage() {
  const router = useRouter();
  const state = useWizardStore((s) => s.state);
  const hasHydrated = useWizardStore((s) => s.hasHydrated);
  const updateImage = useWizardStore((s) => s.updateImage);
  const setDescription = useWizardStore((s) => s.setDescription);

  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Wrong-step guard. Held until rehydration completes so the default
  // state does not trigger a redirect before the persisted step is known.
  useEffect(() => {
    if (!hasHydrated) return;
    if (state.step === 'config' || state.step === 'upload') {
      router.replace(`/detect/${state.step}`);
    }
    if (state.step === 'describe' || state.step === 'done') {
      router.replace(`/detect/${state.step}`);
    }
  }, [hasHydrated, state.step, router]);

  if (!hasHydrated || state.step !== 'review') return null;

  const doneImages = state.images.filter(
    (img): img is UploadedImage & { serverId: string; amenities: AmenityItem[] } =>
      img.status === 'done' && img.serverId != null && img.amenities != null,
  );

  function toggleAmenity(clientId: string, idx: number) {
    const img = doneImages.find((i) => i.clientId === clientId);
    if (img == null) return;
    const next = img.amenities.map((a, i) =>
      i === idx ? { ...a, present: !a.present } : a,
    );
    updateImage(clientId, { amenities: next });
  }

  async function setPrimary(image: UploadedImage & { serverId: string }) {
    if (state.step !== 'review') return;
    try {
      // Tell the server which image is primary so JSON-LD / OG tags pick it.
      await patchImage(state.propertyId, image.serverId, { is_primary: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to set primary.');
    }
  }

  async function generate() {
    if (state.step !== 'review') return;
    setGenerating(true);
    setError(null);

    // Consolidate the user's edited amenities from every image into one list.
    const flat = doneImages.flatMap((img) =>
      img.amenities.map((a) => ({
        amenity_name: a.name,
        room_type: a.room,
        is_present: a.present,
      })),
    );

    try {
      const res = await describeProperty(state.propertyId, {
        amenities: flat,
        model_name: state.config.model_name,
      });
      setDescription(res.description);
      router.push('/detect/describe');
    } catch (err) {
      setError(
        err instanceof ApiError
          ? `API ${err.status}: ${err.message}`
          : err instanceof Error
            ? err.message
            : 'Failed to generate description.',
      );
      setGenerating(false);
    }
  }

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <h1 className="font-display text-3xl text-ink">Review detected amenities</h1>
        <p className="text-ink-soft">
          Tick or untick to refine what the model found. Mark your favourite as the primary
          photo (it becomes the OpenGraph / JSON-LD hero).
        </p>
      </header>

      {doneImages.length === 0 ? (
        <p className="text-ink-muted">
          No successful uploads to review. Go back and upload at least one image.
        </p>
      ) : (
        <ul className="space-y-6">
          {doneImages.map((img) => (
            <li
              key={img.clientId}
              className="grid sm:grid-cols-[200px_1fr] gap-4 rounded-2xl border border-border bg-surface p-4"
            >
              <div className="relative aspect-square rounded-xl overflow-hidden bg-surface-alt">
                <Image
                  src={getImageUrl(img.serverId)}
                  alt={img.fileName}
                  fill
                  sizes="200px"
                  className="object-cover"
                  unoptimized={shouldUnoptimizeApiImages()}
                />
              </div>
              <div className="space-y-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-ink truncate">{img.fileName}</p>
                    <p className="font-mono text-[11px] text-ink-muted mt-0.5">
                      {img.roomType ?? 'unknown room'}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => void setPrimary(img)}
                    className="font-mono text-[11px] uppercase tracking-widest text-ink-muted hover:text-accent transition-colors"
                  >
                    Make primary
                  </button>
                </div>

                <div className="flex flex-wrap gap-1.5">
                  {img.amenities.length === 0 && (
                    <p className="text-sm text-ink-muted">No amenities detected.</p>
                  )}
                  {img.amenities.map((a, i) => (
                    <button
                      key={`${a.name}-${i}`}
                      type="button"
                      onClick={() => toggleAmenity(img.clientId, i)}
                      aria-pressed={a.present}
                      className={[
                        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[12.5px] font-medium leading-none transition-colors cursor-pointer',
                        a.present
                          ? 'bg-ink text-bg border-ink'
                          : 'bg-surface text-ink-muted border-border line-through opacity-60',
                      ].join(' ')}
                    >
                      {a.name}
                    </button>
                  ))}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}

      {error != null && <p className="text-warn text-sm font-mono">{error}</p>}

      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => router.push('/detect/upload')}
          className="font-mono text-xs uppercase tracking-widest text-ink-muted hover:text-ink transition-colors"
        >
          ← Add more photos
        </button>
        <Button onClick={() => void generate()} disabled={generating || doneImages.length === 0}>
          {generating ? 'Generating…' : 'Generate description →'}
        </Button>
      </div>
    </div>
  );
}
