import { test, expect, type Page } from '@playwright/test';
import { mockApi, type MockApi } from './api-mock';

// A tiny valid JPEG, so the browser's file input has something real to carry.
const JPEG = Buffer.from(
  '/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a' +
    'HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAA' +
    'AAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q==',
  'base64',
);

async function fillConfig(page: Page) {
  await page.goto('/detect/config');
  await page.getByLabel('Listing name').fill('E2E Apartment');
  await page.getByLabel('Detection model').selectOption('openai/gpt-4o-mini');
  await page.getByLabel('PIN code').fill('60311');
  await page.getByRole('button', { name: 'Continue → Upload' }).click();
  await expect(page).toHaveURL(/\/detect\/upload/);
}

async function uploadPhoto(page: Page, name = 'kitchen.jpg') {
  await page.locator('input[type="file"]').setInputFiles({
    name,
    mimeType: 'image/jpeg',
    buffer: JPEG,
  });
  await expect(page.getByText(name)).toBeVisible();
  await expect(page.getByText('✓ Done').first()).toBeVisible();
}

test.describe('detect wizard', () => {
  let api: MockApi;

  test.beforeEach(async ({ page }) => {
    api = await mockApi(page);
  });

  test('happy path: a draft stays private until it is published', async ({ page }) => {
    await fillConfig(page);
    await uploadPhoto(page);

    await page.getByRole('button', { name: 'Continue → Review' }).click();
    await expect(page).toHaveURL(/\/detect\/review/);
    await expect(page.getByRole('button', { name: 'refrigerator' })).toBeVisible();

    await page.getByRole('button', { name: /Generate description/ }).click();
    await expect(page).toHaveURL(/\/detect\/describe/);
    await expect(page.locator('textarea')).toHaveValue(/bright kitchen/);

    await page.getByRole('button', { name: 'Save & continue' }).click();
    await expect(page).toHaveURL(/\/detect\/done/);

    // Saved is not published: the final step must not claim otherwise.
    await expect(page.getByText('Draft saved')).toBeVisible();
    expect(api.calls.publish).toBe(0);

    await page.getByRole('button', { name: 'Publish →' }).click();
    await expect(page.getByRole('heading', { name: 'Your listing is live.' })).toBeVisible();
    expect(api.calls.publish).toBe(1);
    expect(api.status).toBe('published');
  });

  test('backtracking: photos can be added from review and the work survives', async ({
    page,
  }) => {
    await fillConfig(page);
    await uploadPhoto(page, 'first.jpg');
    await page.getByRole('button', { name: 'Continue → Review' }).click();
    await expect(page).toHaveURL(/\/detect\/review/);

    await page.getByRole('button', { name: '← Add more photos' }).click();
    await expect(page).toHaveURL(/\/detect\/upload/);
    // The earlier upload is still there — going back did not discard it.
    await expect(page.getByText('first.jpg')).toBeVisible();

    await uploadPhoto(page, 'second.jpg');
    await page.getByRole('button', { name: 'Continue → Review' }).click();
    await expect(page).toHaveURL(/\/detect\/review/);
    await expect(page.getByText('first.jpg')).toBeVisible();
    await expect(page.getByText('second.jpg')).toBeVisible();
  });

  test('backtracking: the stepper walks back to a completed step', async ({ page }) => {
    await fillConfig(page);
    await uploadPhoto(page);
    await page.getByRole('button', { name: 'Continue → Review' }).click();
    await expect(page).toHaveURL(/\/detect\/review/);

    await page.getByRole('button', { name: 'Go back to Upload' }).click();
    await expect(page).toHaveURL(/\/detect\/upload/);
    await expect(page.getByText('kitchen.jpg')).toBeVisible();
  });

  test('refresh: a reload on review keeps the uploaded work', async ({ page }) => {
    await fillConfig(page);
    await uploadPhoto(page);
    await page.getByRole('button', { name: 'Continue → Review' }).click();
    await expect(page).toHaveURL(/\/detect\/review/);

    await page.reload();

    await expect(page).toHaveURL(/\/detect\/review/);
    await expect(page.getByText('kitchen.jpg')).toBeVisible();
    await expect(page.getByRole('button', { name: 'refrigerator' })).toBeVisible();
  });

  test('abandoned draft: leaving mid-wizard publishes nothing', async ({ page }) => {
    await fillConfig(page);
    await uploadPhoto(page);

    // The user wanders off to the home page and comes back later.
    await page.goto('/detect/config');

    // Resumed where they left off, still unpublished.
    await expect(page).toHaveURL(/\/detect\/upload/);
    await expect(page.getByText('kitchen.jpg')).toBeVisible();
    expect(api.calls.publish).toBe(0);
    expect(api.status).not.toBe('published');
  });
});
