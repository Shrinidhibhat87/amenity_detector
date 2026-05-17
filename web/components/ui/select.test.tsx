import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Select } from './select';

const options = [
  { value: 'rent', label: 'For rent' },
  { value: 'sale', label: 'For sale' },
];

describe('Select', () => {
  it('renders a labelled select with the given options', () => {
    render(<Select label="Listing" name="listing" options={options} />);
    const field = screen.getByLabelText('Listing') as HTMLSelectElement;
    expect(field.tagName).toBe('SELECT');
    expect(field.options).toHaveLength(2);
    expect(field.options[0]!.textContent).toBe('For rent');
  });

  it('calls onChange with the chosen value', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <Select
        label="Listing"
        name="listing"
        options={options}
        value="rent"
        onChange={onChange}
      />,
    );
    await user.selectOptions(screen.getByLabelText('Listing'), 'sale');
    expect(onChange).toHaveBeenCalled();
  });
});
