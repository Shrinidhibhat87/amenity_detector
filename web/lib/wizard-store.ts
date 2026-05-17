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

export interface ConfigInput {
  name: string;
  model_name: string;
  // Optional Phase 9 listing metadata
  listing_type?: 'rent' | 'sale';
  price?: number;
  currency?: string;
  price_period?: 'monthly' | 'weekly' | 'nightly' | 'total';
  num_bedrooms?: number;
  num_bathrooms?: number;
  area_sqm?: number;
  property_type?: 'apartment' | 'house' | 'villa' | 'studio' | 'other';
  furnishing?: 'furnished' | 'semi_furnished' | 'unfurnished';
  available_from?: string;
  locality?: string;
  postal_code?: string;
  country_code?: string;
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
      // the factory anyway).
      partialize: (s) => ({ state: s.state }),
    },
  ),
);
