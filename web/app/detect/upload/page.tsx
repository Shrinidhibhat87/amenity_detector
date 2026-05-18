'use client';

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from 'react';
import { useRouter } from 'next/navigation';
import { ApiError, uploadImage } from '@/lib/api';
import {
  useWizardStore,
  type AmenityItem,
  type UploadedImage,
} from '@/lib/wizard-store';
import { Button } from '@/components/ui';

// Per-image timeout — matches the Gradio UI's skip-on-failure pattern so a
// slow / stuck VLM call does not block the whole batch.
const PER_IMAGE_TIMEOUT_MS = 60_000;

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'] as const;

export default function UploadStepPage() {
  const router = useRouter();
  const state = useWizardStore((s) => s.state);
  const addImage = useWizardStore((s) => s.addImage);
  const updateImage = useWizardStore((s) => s.updateImage);
  const goToReview = useWizardStore((s) => s.goToReview);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  // Wrong-step guard — bounce the user to the correct page.
  useEffect(() => {
    if (state.step === 'config') router.replace('/detect/config');
    if (state.step === 'review' || state.step === 'describe' || state.step === 'done') {
      router.replace(`/detect/${state.step}`);
    }
  }, [state.step, router]);

  const startUpload = useCallback(
    async (files: File[]) => {
      if (state.step !== 'upload') return;
      setIsUploading(true);

      // Register every file in the store upfront so the total count is stable
      // throughout the batch — the user sees "1/N, 2/N, … N/N" instead of
      // "1/1, 1/2, …" caused by the denominator growing as we added one at a
      // time. Invalid types are recorded as 'failed' immediately and skipped
      // during the upload pass.
      type Pending = {
        clientId: string;
        file: File;
        valid: boolean;
      };
      const queue: Pending[] = files.map((file) => ({
        clientId: crypto.randomUUID(),
        file,
        valid: ALLOWED_TYPES.includes(file.type as (typeof ALLOWED_TYPES)[number]),
      }));

      for (const item of queue) {
        const entry: UploadedImage = item.valid
          ? {
              clientId: item.clientId,
              fileName: item.file.name,
              status: 'uploading',
            }
          : {
              clientId: item.clientId,
              fileName: item.file.name,
              status: 'failed',
              errorMessage: `Unsupported type: ${item.file.type || 'unknown'}`,
            };
        addImage(entry);
      }

      // Process valid files sequentially — one VLM call at a time keeps
      // backend load predictable.
      for (const item of queue) {
        if (!item.valid) continue;

        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), PER_IMAGE_TIMEOUT_MS);

        try {
          const res = await uploadImage({
            propertyId: state.propertyId,
            file: item.file,
            modelName: state.config.model_name,
            signal: controller.signal,
          });
          clearTimeout(timer);

          const amenities: AmenityItem[] = res.image.amenities.map((a) => ({
            name: a.amenity_name,
            room: a.room_type ?? '',
            present: a.is_present,
          }));

          updateImage(item.clientId, {
            status: 'done',
            serverId: res.image.id,
            amenities,
            roomType: res.image.room_type,
          });
        } catch (err) {
          clearTimeout(timer);
          const msg =
            err instanceof DOMException && err.name === 'AbortError'
              ? `Timed out after ${PER_IMAGE_TIMEOUT_MS / 1000}s`
              : err instanceof ApiError
                ? `API ${err.status}: ${err.message}`
                : err instanceof Error
                  ? err.message
                  : 'Unknown error';
          updateImage(item.clientId, { status: 'failed', errorMessage: msg });
        }
      }

      setIsUploading(false);
    },
    [state, addImage, updateImage],
  );

  function onFileInput(e: ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (files == null || files.length === 0) return;
    void startUpload(Array.from(files));
    // Reset so picking the same file twice re-fires onChange.
    e.target.value = '';
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;
    void startUpload(files);
  }

  if (state.step !== 'upload') return null;
  const doneCount = state.images.filter((i) => i.status === 'done').length;
  const finishedCount = state.images.filter(
    (i) => i.status === 'done' || i.status === 'failed',
  ).length;
  // While a file is mid-flight we show "(finished + 1)/total" so the user
  // sees the in-progress index (1/N, 2/N, …) rather than the lagging
  // finished-count.
  const total = state.images.length;
  const displayedCount = isUploading
    ? Math.min(finishedCount + 1, total)
    : finishedCount;
  const canContinue = doneCount > 0 && !isUploading;

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="font-display text-3xl text-ink">Upload property photos</h1>
        <p className="text-ink-soft">
          Drop or pick JPEG / PNG / WebP files. Each one is uploaded and analysed in turn —
          slow images are skipped after {PER_IMAGE_TIMEOUT_MS / 1000}s so the batch keeps
          moving.
        </p>
      </header>

      {/* Dropzone */}
      <div
        onDrop={onDrop}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        className={[
          'rounded-2xl border-2 border-dashed p-10 text-center transition-colors cursor-pointer',
          dragOver
            ? 'border-accent bg-accent-soft'
            : 'border-border hover:border-border-strong bg-surface',
        ].join(' ')}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click();
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={ALLOWED_TYPES.join(',')}
          onChange={onFileInput}
          className="sr-only"
        />
        <p className="font-mono text-xs uppercase tracking-widest text-ink-muted mb-2">
          {isUploading ? 'Uploading…' : 'Drop files or click to browse'}
        </p>
        <p className="text-sm text-ink-soft">
          Up to ~10 photos works best. Each VLM call takes ~3–15s.
        </p>
      </div>

      {/* Per-image progress list */}
      {state.images.length > 0 && (
        <ul className="space-y-2">
          {state.images.map((img) => (
            <li
              key={img.clientId}
              className="flex items-center justify-between gap-4 rounded-xl border border-border bg-surface px-4 py-3"
            >
              <div className="min-w-0">
                <p className="text-sm text-ink truncate">{img.fileName}</p>
                {img.status === 'failed' && img.errorMessage != null && (
                  <p className="font-mono text-[11px] text-warn truncate mt-0.5">
                    {img.errorMessage}
                  </p>
                )}
                {img.status === 'done' && img.amenities != null && (
                  <p className="font-mono text-[11px] text-ink-muted mt-0.5">
                    {img.amenities.filter((a) => a.present).length} amenities ·{' '}
                    {img.roomType ?? 'unknown room'}
                  </p>
                )}
              </div>
              <StatusBadge status={img.status} />
            </li>
          ))}
        </ul>
      )}

      <div className="flex items-center justify-between pt-2">
        <p className="font-mono text-xs text-ink-muted">
          {displayedCount}/{total} processed
        </p>
        <Button
          onClick={() => {
            goToReview();
            router.push('/detect/review');
          }}
          disabled={!canContinue}
        >
          Continue → Review
        </Button>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: 'uploading' | 'done' | 'failed' }) {
  if (status === 'uploading') {
    return (
      <span className="font-mono text-[10px] uppercase tracking-widest px-2 py-0.5 rounded-full bg-surface-alt text-ink-muted">
        ⋯ Processing
      </span>
    );
  }
  if (status === 'done') {
    return (
      <span className="font-mono text-[10px] uppercase tracking-widest px-2 py-0.5 rounded-full bg-ok-soft text-ok">
        ✓ Done
      </span>
    );
  }
  return (
    <span className="font-mono text-[10px] uppercase tracking-widest px-2 py-0.5 rounded-full bg-warn-soft text-warn">
      ✕ Failed
    </span>
  );
}
