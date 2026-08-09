import { defineConfig, devices } from '@playwright/test';

/**
 * End-to-end config for the detect wizard.
 *
 * The specs stub the FastAPI backend at the network boundary (see
 * tests/e2e/api-mock.ts) rather than running it. What these tests are for is
 * the wizard's state transitions — back, refresh, abandon, publish — and those
 * break in the browser, not in the model. Stubbing also keeps the suite
 * hermetic: no VLM calls, no database, no OpenRouter spend in CI.
 */
const PORT = 3100;

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env['CI'],
  retries: process.env['CI'] ? 2 : 0,
  reporter: process.env['CI'] ? 'github' : 'list',
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'on-first-retry',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: `pnpm exec next dev --port ${PORT}`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env['CI'],
    timeout: 120_000,
  },
});
