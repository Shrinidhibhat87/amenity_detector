/**
 * Detection wizard state — Zustand store with localStorage persistence.
 *
 * Why a flat shape rather than a discriminated union per step?
 *   The union modelled the wizard as a one-way pipeline: each transition
 *   rebuilt the state for the next step, so walking backwards meant either
 *   losing fields or adding a bespoke "go back" action per hop. The wizard is
 *   not one-way — a user adds photos from review, re-reads the description,
 *   returns. A flat state with a `furthest` marker says the same thing in less
 *   code: every field that has been collected stays collected, `step` is where
 *   the user is looking, and `furthest` is how far the work has actually got.
 *
 * Invalidation:
 *   Editing amenities changes the facts the description was written from, so
 *   the description is marked stale and the user is told to regenerate.
 *   Editing listing metadata (price, tone later) does not touch detection or
 *   the description — no invalidation.
 *
 * Persistence:
 *   The store survives page reloads via Zustand's `persist` middleware, which
 *   serialises to localStorage. Only state is persisted; actions are recreated
 *   by the store factory on every load.
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
  // Address entered once, in the config step. `street` and `radius_m` are not
  // property columns — they parameterise the locality enrichment that runs on
  // the upload step, which is why they live on the wizard config rather than
  // in the listing metadata sent to POST /properties.
  street?: string | undefined;
  radius_m?: number | undefined;
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

// ── Steps ─────────────────────────────────────────────────────────────────────

export const WIZARD_STEPS = ['config', 'upload', 'review', 'describe', 'done'] as const;
export type WizardStep = (typeof WIZARD_STEPS)[number];

const stepIndex = (step: WizardStep): number => WIZARD_STEPS.indexOf(step);

export interface WizardState {
  /** Where the user is looking right now. */
  step: WizardStep;
  /** The furthest step the work has actually reached — the navigable range. */
  furthest: WizardStep;
  config: ConfigInput;
  /** Set once the server has created the property shell. */
  propertyId: string | null;
  images: UploadedImage[];
  description: string;
  /** The description no longer matches the reviewed amenities. */
  descriptionStale: boolean;
  /** True once the publish action has succeeded on the server. */
  published: boolean;
}

const EMPTY_CONFIG: ConfigInput = { name: '', model_name: '' };

const INITIAL_STATE: WizardState = {
  step: 'config',
  furthest: 'config',
  config: { ...EMPTY_CONFIG },
  propertyId: null,
  images: [],
  description: '',
  descriptionStale: false,
  published: false,
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
  /** Navigate to a step the work has already reached. No-op otherwise. */
  goToStep: (step: WizardStep) => void;
  /** Whether `goToStep` would move to that step — drives the clickable stepper. */
  canVisit: (step: WizardStep) => boolean;
  goToReview: () => void;
  setDescription: (text: string) => void;
  finish: () => void;
  markPublished: () => void;
  reset: () => void;
}

/** Extend the navigable range without ever shrinking it. */
function reach(state: WizardState, step: WizardStep): WizardStep {
  return stepIndex(step) > stepIndex(state.furthest) ? step : state.furthest;
}

// ── Store factory ─────────────────────────────────────────────────────────────

export const useWizardStore = create<WizardStore>()(
  persist(
    (set, get) => ({
      state: { ...INITIAL_STATE, config: { ...EMPTY_CONFIG } },
      hasHydrated: false,
      setHasHydrated: (value) => set({ hasHydrated: value }),

      setConfig: (patch) =>
        set((s) => ({ state: { ...s.state, config: { ...s.state.config, ...patch } } })),

      startUpload: (propertyId) =>
        set((s) => ({
          state: {
            ...s.state,
            propertyId,
            step: 'upload',
            furthest: reach(s.state, 'upload'),
          },
        })),

      addImage: (image) =>
        set((s) => ({ state: { ...s.state, images: [...s.state.images, image] } })),

      updateImage: (clientId, patch) =>
        set((s) => {
          const images = s.state.images.map((img) =>
            img.clientId === clientId ? { ...img, ...patch } : img,
          );
          // Amenities are the facts the description was generated from. Change
          // them and the existing description is out of date — but only if
          // there is one; a detection result arriving during upload is not an
          // edit.
          const editsAmenities = patch.amenities != null && s.state.description !== '';
          return {
            state: {
              ...s.state,
              images,
              descriptionStale: s.state.descriptionStale || editsAmenities,
            },
          };
        }),

      canVisit: (step) => {
        const { state } = get();
        if (stepIndex(step) > stepIndex(state.furthest)) return false;
        // Every step past config needs a property to act on.
        return step === 'config' || state.propertyId != null;
      },

      goToStep: (step) =>
        set((s) => (get().canVisit(step) ? { state: { ...s.state, step } } : s)),

      goToReview: () =>
        set((s) =>
          s.state.propertyId == null
            ? s
            : { state: { ...s.state, step: 'review', furthest: reach(s.state, 'review') } },
        ),

      setDescription: (text) =>
        set((s) => ({
          state: {
            ...s.state,
            description: text,
            // The text now reflects whatever the user last approved, whether
            // it was regenerated or hand-edited.
            descriptionStale: false,
            step: 'describe',
            furthest: reach(s.state, 'describe'),
          },
        })),

      finish: () =>
        set((s) => ({ state: { ...s.state, step: 'done', furthest: reach(s.state, 'done') } })),

      markPublished: () => set((s) => ({ state: { ...s.state, published: true } })),

      reset: () => set({ state: { ...INITIAL_STATE, config: { ...EMPTY_CONFIG } } }),
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
        // A published listing is finished work. Restoring it would trap every
        // subsequent /detect/* visit on the completion screen, so reopening
        // the app starts a fresh listing instead. An unpublished draft is
        // restored exactly as it was — that is the point of persisting it.
        if (rehydrated.state.published) {
          rehydrated.state = { ...INITIAL_STATE, config: { ...EMPTY_CONFIG } };
        }
        rehydrated.setHasHydrated(true);
      },
    },
  ),
);
