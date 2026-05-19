/**
 * Detection wizard state — Zustand store with localStorage persistence.
 *
 * Why a discriminated union over a flat state?
 *   Each step needs different fields. In the config step there's no
 *   propertyId yet; in the upload step it must exist. A flat shape with
 *   propertyId?: string forces every consumer to handle `undefined` even
 *   when the step guarantees it. The discriminated union lets the compiler
 *   narrow by the `step` field, so `state.step === 'upload'` proves
 *   `state.propertyId: string`.
 *
 * Persistence:
 *   The store survives page reloads via Zustand's `persist` middleware,
 *   which serialises to localStorage. Only the state (not actions) is
 *   persisted; actions are recreated by the store factory on every load.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

// ── Domain types ──────────────────────────────────────────────────────────────

// `field?: T | undefined` (not `field?: T`) is required by exactOptionalPropertyTypes.
// Setters need to write `undefined` explicitly to clear a value; the bare `?`
// form would forbid that under the strict tsconfig.
export interface ConfigInput {
  name: string;
  model_name: string;
  // Optional Phase 9 listing metadata
  listing_type?: 'rent' | 'sale' | undefined;
  price?: number | undefined;
  currency?: string | undefined;
  price_period?: 'monthly' | 'weekly' | 'nightly' | 'total' | undefined;
  num_bedrooms?: number | undefined;
  num_bathrooms?: number | undefined;
  area_sqm?: number | undefined;
  property_type?: 'apartment' | 'house' | 'villa' | 'studio' | 'other' | undefined;
  furnishing?: 'furnished' | 'semi_furnished' | 'unfurnished' | undefined;
  available_from?: string | undefined;
  locality?: string | undefined;
  postal_code?: string | undefined;
  country_code?: string | undefined;
}

export interface AmenityItem {
  name: string;
  room: string;
  present: boolean;
}

export interface UploadedImage {
  /** Client-side stable id, used to update the entry while the server id is pending. */
  clientId: string;
  fileName: string;
  status: 'uploading' | 'done' | 'failed';
  errorMessage?: string;
  /** Server-assigned image id once upload + detection completes. */
  serverId?: string;
  /** Detected amenities returned by the server. */
  amenities?: AmenityItem[];
  /** Server-classified room type. */
  roomType?: string | null;
}

// ── Discriminated union for step state ────────────────────────────────────────

const EMPTY_CONFIG: ConfigInput = { name: '', model_name: '' };

export type WizardState =
  | { step: 'config'; config: ConfigInput }
  | {
      step: 'upload';
      config: ConfigInput;
      propertyId: string;
      images: UploadedImage[];
    }
  | {
      step: 'review';
      config: ConfigInput;
      propertyId: string;
      images: UploadedImage[];
    }
  | {
      step: 'describe';
      config: ConfigInput;
      propertyId: string;
      images: UploadedImage[];
      description: string;
    }
  | {
      step: 'done';
      config: ConfigInput;
      propertyId: string;
      images: UploadedImage[];
      description: string;
    };

// ── Store shape (state + actions in one) ──────────────────────────────────────

interface WizardStore {
  state: WizardState;

  // True once Zustand persist has restored from localStorage. Step pages must
  // gate their wrong-step guards on this so the initial default state
  // (`step: 'config'`) does not cause a redirect before rehydration finishes.
  hasHydrated: boolean;
  setHasHydrated: (value: boolean) => void;

  setConfig: (patch: Partial<ConfigInput>) => void;
  startUpload: (propertyId: string) => void;
  addImage: (image: UploadedImage) => void;
  updateImage: (clientId: string, patch: Partial<UploadedImage>) => void;
  goToReview: () => void;
  setDescription: (text: string) => void;
  finish: () => void;
  reset: () => void;
}

// ── Store factory ─────────────────────────────────────────────────────────────

export const useWizardStore = create<WizardStore>()(
  persist(
    (set) => ({
      state: { step: 'config', config: { ...EMPTY_CONFIG } },
      hasHydrated: false,
      setHasHydrated: (value) => set({ hasHydrated: value }),

      setConfig: (patch) =>
        set((s) => {
          // Only the config step accepts edits to config; other steps freeze it.
          if (s.state.step !== 'config') return s;
          return {
            state: {
              step: 'config',
              config: { ...s.state.config, ...patch },
            },
          };
        }),

      startUpload: (propertyId) =>
        set((s) => {
          if (s.state.step !== 'config') return s;
          return {
            state: {
              step: 'upload',
              config: s.state.config,
              propertyId,
              images: [],
            },
          };
        }),

      addImage: (image) =>
        set((s) => {
          if (s.state.step !== 'upload') return s;
          return {
            state: { ...s.state, images: [...s.state.images, image] },
          };
        }),

      updateImage: (clientId, patch) =>
        set((s) => {
          if (s.state.step !== 'upload' && s.state.step !== 'review') return s;
          const images = s.state.images.map((img) =>
            img.clientId === clientId ? { ...img, ...patch } : img,
          );
          return { state: { ...s.state, images } };
        }),

      goToReview: () =>
        set((s) => {
          if (s.state.step !== 'upload') return s;
          return {
            state: {
              step: 'review',
              config: s.state.config,
              propertyId: s.state.propertyId,
              images: s.state.images,
            },
          };
        }),

      setDescription: (text) =>
        set((s) => {
          if (s.state.step === 'review') {
            return {
              state: {
                step: 'describe',
                config: s.state.config,
                propertyId: s.state.propertyId,
                images: s.state.images,
                description: text,
              },
            };
          }
          if (s.state.step === 'describe') {
            return { state: { ...s.state, description: text } };
          }
          return s;
        }),

      finish: () =>
        set((s) => {
          if (s.state.step !== 'describe') return s;
          return {
            state: {
              step: 'done',
              config: s.state.config,
              propertyId: s.state.propertyId,
              images: s.state.images,
              description: s.state.description,
            },
          };
        }),

      reset: () =>
        set({
          state: { step: 'config', config: { ...EMPTY_CONFIG } },
        }),
    }),
    {
      name: 'wizard',
      storage: createJSONStorage(() => localStorage),
      // Persist only the state slice, not actions (which are reconstructed by
      // the factory anyway). hasHydrated is intentionally excluded so it
      // always starts false until the rehydrate callback fires.
      partialize: (s) => ({ state: s.state }),
      onRehydrateStorage: () => (rehydrated, error) => {
        if (error != null || rehydrated == null) return;
        // `done` is terminal — restoring it would trap every subsequent
        // /detect/* visit on the completion screen. Reset to a clean config
        // so reopening the app starts a fresh listing.
        if (rehydrated.state.step === 'done') {
          rehydrated.state = { step: 'config', config: { ...EMPTY_CONFIG } };
        }
        rehydrated.setHasHydrated(true);
      },
    },
  ),
);
