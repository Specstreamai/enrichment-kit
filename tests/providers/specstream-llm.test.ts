import { describe, it, expect, vi, afterEach } from 'vitest';
import { SpecStreamLLMProvider } from '../../src/providers/specstream-llm.js';
import { buildProviders, credentialsFromEnv } from '../../src/waterfall/builder.js';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

const searchReply = {
  data: {
    kind: 'list',
    sources: [
      {
        url: 'https://www.linkedin.com/in/bchesky',
        title: 'Brian Chesky - Co-Founder and CEO - Airbnb | LinkedIn',
        site: 'linkedin.com',
        snippet: 'Co-Founder and CEO at Airbnb. San Francisco Bay Area.',
        cited: false,
      },
      { url: 'https://www.linkedin.com/in/someone-else', title: 'Brian C. - Engineer', site: 'linkedin.com', snippet: '' },
    ],
    took_ms: 900,
    price_usd: 0.005,
  },
};

const groqReply = {
  choices: [{
    message: {
      content: JSON.stringify({
        linkedinUrl: 'https://www.linkedin.com/in/bchesky',
        title: 'Co-Founder and CEO',
        company: 'Airbnb',
        fullName: 'Brian Chesky',
        location: 'San Francisco Bay Area',
        confidence: 0.9,
      }),
    },
  }],
};

function mockFetch(search: Response, llm?: Response) {
  const fetchMock = vi.fn(async (url: string | URL | Request) => {
    const u = String(url);
    if (u.startsWith('https://api.specstream.ai/')) return search;
    if (u.startsWith('https://api.groq.com/') && llm) return llm;
    throw new Error(`unexpected fetch ${u}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('SpecStreamLLMProvider', () => {
  it('requires both keys', () => {
    expect(() => new SpecStreamLLMProvider('', 'groq', 'llm-key')).toThrow();
    expect(() => new SpecStreamLLMProvider('ss-key', 'groq')).toThrow();
  });

  it('searches with a Bearer key and extracts the profile', async () => {
    const fetchMock = mockFetch(jsonResponse(searchReply), jsonResponse(groqReply));
    const p = new SpecStreamLLMProvider('ss-key', 'groq', 'llm-key');
    const out = await p.enrich({ fullName: 'Brian Chesky', company: 'Airbnb' });

    const [searchUrl, searchInit] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    const params = new URL(searchUrl).searchParams;
    expect(searchUrl.startsWith('https://api.specstream.ai/v1/web-search?')).toBe(true);
    expect(params.get('q')).toBe('"Brian Chesky" "Airbnb" site:linkedin.com/in');
    expect((searchInit.headers as Record<string, string>).Authorization).toBe('Bearer ss-key');

    // The LLM sees the search results under the same keys as with SerpAPI.
    const llmBody = JSON.parse((fetchMock.mock.calls[1] as unknown as [string, RequestInit])[1].body as string);
    expect(llmBody.messages[1].content).toContain('"link":"https://www.linkedin.com/in/bchesky"');

    expect(out).not.toBeNull();
    expect(out!.linkedinUrl).toBe('https://www.linkedin.com/in/bchesky');
    expect(out!.title).toBe('Co-Founder and CEO');
    expect(out!.company).toBe('Airbnb');
    expect(out!.source).toBe('specstream-llm');
    expect(out!.cost).toBe(p.costPerCall);
  });

  it('returns null without calling the LLM when search has no results', async () => {
    const fetchMock = mockFetch(jsonResponse({ data: { kind: 'list', sources: [] } }));
    const p = new SpecStreamLLMProvider('ss-key', 'groq', 'llm-key');
    expect(await p.enrich({ fullName: 'Nobody Known', company: 'Nowhere' })).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('returns null when the LLM is not confident', async () => {
    const lowConfidence = { choices: [{ message: { content: JSON.stringify({ linkedinUrl: null, title: null, company: null, fullName: null, location: null, confidence: 0.2 }) } }] };
    mockFetch(jsonResponse(searchReply), jsonResponse(lowConfidence));
    const p = new SpecStreamLLMProvider('ss-key', 'groq', 'llm-key');
    expect(await p.enrich({ fullName: 'Brian Chesky', company: 'Airbnb' })).toBeNull();
  });

  it('throws ProviderError on HTTP errors so the waterfall records the attempt', async () => {
    mockFetch(jsonResponse({ error: 'insufficient_balance' }, 402));
    const p = new SpecStreamLLMProvider('ss-key', 'groq', 'llm-key');
    await expect(p.enrich({ fullName: 'Brian Chesky', company: 'Airbnb' })).rejects.toThrow('specstream-llm HTTP 402');
  });

  it('is built from SPECSTREAM_API_KEY plus an LLM key', () => {
    const creds = credentialsFromEnv({ SPECSTREAM_API_KEY: 'ss-key', GROQ_API_KEY: 'llm-key' });
    expect(creds.specstream).toEqual({ apiKey: 'ss-key', llmProvider: 'groq', llmApiKey: 'llm-key' });
    expect(buildProviders(creds).map(p => p.name)).toEqual(['specstream-llm']);
    expect(buildProviders(credentialsFromEnv({ SPECSTREAM_API_KEY: 'ss-key' }))).toEqual([]);
  });
});
