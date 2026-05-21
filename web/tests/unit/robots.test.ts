import { describe, expect, it } from 'vitest';
import robots from '../../app/robots';

describe('robots', () => {
  const result = robots();
  const rule = result.rules;
  if (!rule || Array.isArray(rule) === false) {
    throw new Error('expected robots.rules to be an array');
  }
  const first = (Array.isArray(rule) ? rule : [rule])[0];
  if (first == null) throw new Error('expected one rule entry');

  it('targets all user agents', () => {
    expect(first.userAgent).toBe('*');
  });

  it('allows the public surfaces — home, browse, properties, search, feeds', () => {
    const allow = Array.isArray(first.allow) ? first.allow : [first.allow];
    expect(allow).toContain('/');
    expect(allow).toContain('/browse');
    expect(allow).toContain('/properties/');
    expect(allow).toContain('/search');
    expect(allow).toContain('/llms.txt');
    expect(allow).toContain('/api/feed.jsonl');
  });

  it('disallows owner-only and internal endpoints', () => {
    const disallow = Array.isArray(first.disallow) ? first.disallow : [first.disallow];
    expect(disallow).toContain('/kitchen-sink');
    expect(disallow).toContain('/detect/');
    expect(disallow).toContain('/api/parse');
  });

  it('points to the canonical sitemap URL', () => {
    expect(result.sitemap).toContain('/sitemap.xml');
  });
});
