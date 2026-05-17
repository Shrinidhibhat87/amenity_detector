import { beforeEach, describe, expect, it } from 'vitest';
import { useWizardStore } from '../../lib/wizard-store';

beforeEach(() => {
  useWizardStore.getState().reset();
  // Clear persisted localStorage between tests
  localStorage.clear();
});

describe('wizard store — initial state', () => {
  it('starts on the config step', () => {
    expect(useWizardStore.getState().state.step).toBe('config');
  });

  it('has empty config defaults', () => {
    const { state } = useWizardStore.getState();
    if (state.step !== 'config') throw new Error('expected config step');
    expect(state.config.name).toBe('');
    expect(state.config.model_name).toBe('');
  });
});

describe('wizard store — setConfig', () => {
  it('patches config fields', () => {
    useWizardStore.getState().setConfig({ name: 'Beach Loft' });
    const { state } = useWizardStore.getState();
    if (state.step !== 'config') throw new Error('expected config step');
    expect(state.config.name).toBe('Beach Loft');
  });

  it('merges multiple patches without losing previous fields', () => {
    useWizardStore.getState().setConfig({ name: 'X' });
    useWizardStore.getState().setConfig({ model_name: 'gpt-4o-mini' });
    const { state } = useWizardStore.getState();
    if (state.step !== 'config') throw new Error('expected config step');
    expect(state.config.name).toBe('X');
    expect(state.config.model_name).toBe('gpt-4o-mini');
  });

  it('accepts optional listing metadata', () => {
    useWizardStore.getState().setConfig({
      listing_type: 'rent',
      price: 1200,
      locality: 'Berlin',
    });
    const { state } = useWizardStore.getState();
    if (state.step !== 'config') throw new Error('expected config step');
    expect(state.config.listing_type).toBe('rent');
    expect(state.config.price).toBe(1200);
    expect(state.config.locality).toBe('Berlin');
  });
});

describe('wizard store — startUpload', () => {
  it('transitions config → upload with propertyId', () => {
    useWizardStore.getState().setConfig({ name: 'X', model_name: 'gpt' });
    useWizardStore.getState().startUpload('prop-123');

    const { state } = useWizardStore.getState();
    expect(state.step).toBe('upload');
    if (state.step !== 'upload') throw new Error('expected upload step');
    expect(state.propertyId).toBe('prop-123');
    expect(state.images).toEqual([]);
    // Config data carries through the transition
    expect(state.config.name).toBe('X');
  });
});

describe('wizard store — image lifecycle', () => {
  beforeEach(() => {
    useWizardStore.getState().setConfig({ name: 'X', model_name: 'gpt' });
    useWizardStore.getState().startUpload('prop-1');
  });

  it('addImage appends to images list', () => {
    useWizardStore.getState().addImage({
      clientId: 'c1',
      fileName: 'photo.jpg',
      status: 'uploading',
    });
    const { state } = useWizardStore.getState();
    if (state.step !== 'upload') throw new Error('expected upload step');
    expect(state.images).toHaveLength(1);
    expect(state.images[0]?.fileName).toBe('photo.jpg');
  });

  it('updateImage patches by clientId', () => {
    useWizardStore.getState().addImage({
      clientId: 'c1',
      fileName: 'photo.jpg',
      status: 'uploading',
    });
    useWizardStore.getState().updateImage('c1', {
      status: 'done',
      serverId: 'srv-9',
    });
    const { state } = useWizardStore.getState();
    if (state.step !== 'upload') throw new Error('expected upload step');
    expect(state.images[0]?.status).toBe('done');
    expect(state.images[0]?.serverId).toBe('srv-9');
  });

  it('updateImage ignores unknown clientId silently', () => {
    useWizardStore.getState().addImage({
      clientId: 'c1',
      fileName: 'photo.jpg',
      status: 'done',
    });
    useWizardStore.getState().updateImage('nope', { status: 'failed' });
    const { state } = useWizardStore.getState();
    if (state.step !== 'upload') throw new Error('expected upload step');
    expect(state.images[0]?.status).toBe('done');
  });

  it('goToReview transitions upload → review', () => {
    useWizardStore.getState().addImage({
      clientId: 'c1',
      fileName: 'p.jpg',
      status: 'done',
      serverId: 's1',
      amenities: [{ name: 'WiFi', room: 'living_room', present: true }],
    });
    useWizardStore.getState().goToReview();
    const { state } = useWizardStore.getState();
    expect(state.step).toBe('review');
    if (state.step !== 'review') throw new Error('expected review step');
    expect(state.images).toHaveLength(1);
    expect(state.propertyId).toBe('prop-1');
  });
});

describe('wizard store — description flow', () => {
  beforeEach(() => {
    useWizardStore.getState().setConfig({ name: 'X', model_name: 'gpt' });
    useWizardStore.getState().startUpload('prop-1');
    useWizardStore.getState().addImage({
      clientId: 'c1',
      fileName: 'p.jpg',
      status: 'done',
      serverId: 's1',
    });
    useWizardStore.getState().goToReview();
  });

  it('setDescription transitions review → describe', () => {
    useWizardStore.getState().setDescription('Lovely seaside place.');
    const { state } = useWizardStore.getState();
    expect(state.step).toBe('describe');
    if (state.step !== 'describe') throw new Error('expected describe step');
    expect(state.description).toBe('Lovely seaside place.');
    expect(state.propertyId).toBe('prop-1');
  });

  it('setDescription on describe step updates text without losing step', () => {
    useWizardStore.getState().setDescription('Draft');
    useWizardStore.getState().setDescription('Final');
    const { state } = useWizardStore.getState();
    expect(state.step).toBe('describe');
    if (state.step !== 'describe') throw new Error('expected describe step');
    expect(state.description).toBe('Final');
  });

  it('finish transitions describe → done', () => {
    useWizardStore.getState().setDescription('Done text');
    useWizardStore.getState().finish();
    const { state } = useWizardStore.getState();
    expect(state.step).toBe('done');
    if (state.step !== 'done') throw new Error('expected done step');
    expect(state.propertyId).toBe('prop-1');
  });
});

describe('wizard store — reset', () => {
  it('returns the store to the initial config step', () => {
    useWizardStore.getState().setConfig({ name: 'X', model_name: 'gpt' });
    useWizardStore.getState().startUpload('prop-1');
    useWizardStore.getState().reset();
    const { state } = useWizardStore.getState();
    expect(state.step).toBe('config');
    if (state.step !== 'config') throw new Error('expected config step');
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
});
