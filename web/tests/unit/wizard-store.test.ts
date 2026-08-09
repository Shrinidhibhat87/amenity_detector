import { beforeEach, describe, expect, it } from 'vitest';
import { useWizardStore } from '../../lib/wizard-store';

beforeEach(() => {
  useWizardStore.getState().reset();
  // Clear persisted localStorage between tests
  localStorage.clear();
});

/** Walk the wizard as far as `upload`, with one detected image. */
function uploadedOne() {
  const store = useWizardStore.getState();
  store.setConfig({ name: 'X', model_name: 'gpt' });
  store.startUpload('prop-1');
  store.addImage({ clientId: 'c1', fileName: 'p.jpg', status: 'uploading' });
  store.updateImage('c1', {
    status: 'done',
    serverId: 's1',
    amenities: [{ name: 'WiFi', room: 'living_room', present: true }],
  });
}

describe('wizard store — initial state', () => {
  it('starts on the config step with no property', () => {
    const { state } = useWizardStore.getState();
    expect(state.step).toBe('config');
    expect(state.furthest).toBe('config');
    expect(state.propertyId).toBeNull();
  });

  it('has empty config defaults', () => {
    const { state } = useWizardStore.getState();
    expect(state.config.name).toBe('');
    expect(state.config.model_name).toBe('');
  });
});

describe('wizard store — setConfig', () => {
  it('patches config fields', () => {
    useWizardStore.getState().setConfig({ name: 'Beach Loft' });
    expect(useWizardStore.getState().state.config.name).toBe('Beach Loft');
  });

  it('merges multiple patches without losing previous fields', () => {
    useWizardStore.getState().setConfig({ name: 'X' });
    useWizardStore.getState().setConfig({ model_name: 'gpt-4o-mini' });
    const { config } = useWizardStore.getState().state;
    expect(config.name).toBe('X');
    expect(config.model_name).toBe('gpt-4o-mini');
  });

  it('accepts the address and listing metadata', () => {
    useWizardStore.getState().setConfig({
      listing_type: 'rent',
      price: 1200,
      locality: 'Berlin',
      postal_code: '10999',
      street: 'Oranienstraße',
      radius_m: 5000,
    });
    const { config } = useWizardStore.getState().state;
    expect(config.listing_type).toBe('rent');
    expect(config.price).toBe(1200);
    expect(config.postal_code).toBe('10999');
    expect(config.street).toBe('Oranienstraße');
    expect(config.radius_m).toBe(5000);
  });
});

describe('wizard store — startUpload', () => {
  it('records the property and opens the upload step', () => {
    useWizardStore.getState().setConfig({ name: 'X', model_name: 'gpt' });
    useWizardStore.getState().startUpload('prop-123');

    const { state } = useWizardStore.getState();
    expect(state.step).toBe('upload');
    expect(state.furthest).toBe('upload');
    expect(state.propertyId).toBe('prop-123');
    expect(state.images).toEqual([]);
    expect(state.config.name).toBe('X');
  });
});

describe('wizard store — image lifecycle', () => {
  beforeEach(uploadedOne);

  it('addImage appends to the images list', () => {
    useWizardStore.getState().addImage({
      clientId: 'c2',
      fileName: 'second.jpg',
      status: 'uploading',
    });
    expect(useWizardStore.getState().state.images).toHaveLength(2);
  });

  it('updateImage patches by clientId', () => {
    expect(useWizardStore.getState().state.images[0]?.serverId).toBe('s1');
  });

  it('updateImage ignores an unknown clientId silently', () => {
    useWizardStore.getState().updateImage('nope', { status: 'failed' });
    expect(useWizardStore.getState().state.images[0]?.status).toBe('done');
  });

  it('accepts photos added while on the review step', () => {
    useWizardStore.getState().goToReview();
    useWizardStore.getState().addImage({
      clientId: 'c3',
      fileName: 'late.jpg',
      status: 'uploading',
    });

    const { state } = useWizardStore.getState();
    expect(state.images).toHaveLength(2);
    expect(state.step).toBe('review');
  });
});

describe('wizard store — navigation', () => {
  beforeEach(uploadedOne);

  it('allows revisiting a step the work already reached', () => {
    useWizardStore.getState().goToReview();
    useWizardStore.getState().goToStep('upload');

    const { state } = useWizardStore.getState();
    expect(state.step).toBe('upload');
    // Going back does not undo the progress already made.
    expect(state.furthest).toBe('review');
    expect(state.images).toHaveLength(1);
  });

  it('refuses to skip ahead to a step the work has not reached', () => {
    useWizardStore.getState().goToStep('describe');
    expect(useWizardStore.getState().state.step).toBe('upload');
  });

  it('reports which steps are reachable', () => {
    useWizardStore.getState().goToReview();
    const store = useWizardStore.getState();
    expect(store.canVisit('upload')).toBe(true);
    expect(store.canVisit('review')).toBe(true);
    expect(store.canVisit('describe')).toBe(false);
  });

  it('never navigates past config without a property', () => {
    useWizardStore.getState().reset();
    expect(useWizardStore.getState().canVisit('upload')).toBe(false);
  });
});

describe('wizard store — description flow', () => {
  beforeEach(() => {
    uploadedOne();
    useWizardStore.getState().goToReview();
  });

  it('setDescription opens the describe step with the text', () => {
    useWizardStore.getState().setDescription('Lovely seaside place.');
    const { state } = useWizardStore.getState();
    expect(state.step).toBe('describe');
    expect(state.description).toBe('Lovely seaside place.');
    expect(state.propertyId).toBe('prop-1');
  });

  it('setDescription on the describe step updates the text in place', () => {
    useWizardStore.getState().setDescription('Draft');
    useWizardStore.getState().setDescription('Final');
    const { state } = useWizardStore.getState();
    expect(state.step).toBe('describe');
    expect(state.description).toBe('Final');
  });

  it('finish opens the done step', () => {
    useWizardStore.getState().setDescription('Done text');
    useWizardStore.getState().finish();
    const { state } = useWizardStore.getState();
    expect(state.step).toBe('done');
    expect(state.propertyId).toBe('prop-1');
  });
});

describe('wizard store — dependency-aware invalidation', () => {
  beforeEach(() => {
    uploadedOne();
    useWizardStore.getState().goToReview();
    useWizardStore.getState().setDescription('Written from the confirmed amenities.');
  });

  it('marks the description stale when amenities change', () => {
    useWizardStore.getState().updateImage('c1', {
      amenities: [{ name: 'WiFi', room: 'living_room', present: false }],
    });
    expect(useWizardStore.getState().state.descriptionStale).toBe(true);
  });

  it('leaves the description alone when listing metadata changes', () => {
    useWizardStore.getState().setConfig({ price: 1500, furnishing: 'furnished' });
    expect(useWizardStore.getState().state.descriptionStale).toBe(false);
  });

  it('clears staleness once the description is regenerated', () => {
    useWizardStore.getState().updateImage('c1', {
      amenities: [{ name: 'WiFi', room: 'living_room', present: false }],
    });
    useWizardStore.getState().setDescription('Rewritten without WiFi.');
    expect(useWizardStore.getState().state.descriptionStale).toBe(false);
  });

  it('does not mark a fresh detection result as an edit', () => {
    useWizardStore.getState().reset();
    uploadedOne();
    expect(useWizardStore.getState().state.descriptionStale).toBe(false);
  });
});

describe('wizard store — reset', () => {
  it('returns the store to the initial config step', () => {
    uploadedOne();
    useWizardStore.getState().reset();
    const { state } = useWizardStore.getState();
    expect(state.step).toBe('config');
    expect(state.propertyId).toBeNull();
    expect(state.images).toEqual([]);
    expect(state.config.name).toBe('');
  });
});

describe('wizard store — persistence', () => {
  it('persists state to localStorage under the "wizard" key', () => {
    useWizardStore.getState().setConfig({ name: 'Persisted' });
    const raw = localStorage.getItem('wizard');
    expect(raw).not.toBeNull();
    expect(raw).toContain('Persisted');
  });

  it('keeps an unfinished draft, including its images, in the persisted state', () => {
    uploadedOne();
    const raw = localStorage.getItem('wizard');
    expect(raw).toContain('prop-1');
    expect(raw).toContain('p.jpg');
  });
});
