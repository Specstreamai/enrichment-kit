/**
 * SpecStream Search API + LLM provider.
 *
 * Same shape as serpapi-llm: one web search for the person's LinkedIn profile, then the
 * shared LLM extract stage (./llm-extract.ts) reads the top results. Only the search
 * call differs: GET https://api.specstream.ai/v1/web-search with a Bearer key.
 * Docs: https://specstream.ai/docs/search-api
 *
 * costPerCall: $0.005 per search (published price, $5 per 1,000 searches of up to
 * 10 results) plus about $0.001 for the LLM extract call.
 *
 * typicalHitRate: not measured on a labelled lead list yet. Set below serpapi-llm's
 * 0.80 so the waterfall does not favour this provider on an unproven number; replace
 * it with measured data when someone has it.
 */

import { BaseProvider } from './base.js';
import type { EnrichmentInput, EnrichmentOutput, ProviderCapability } from '../types.js';
import { llmExtract, type LLMProvider } from './llm-extract.js';

interface SpecStreamSearchResponse {
  data?: {
    sources?: Array<{ url?: string; title?: string; snippet?: string; site?: string; published?: string }>;
  };
}

export class SpecStreamLLMProvider extends BaseProvider {
  name = 'specstream-llm';
  costPerCall = 0.006;
  typicalHitRate = 0.75;
  capabilities: ProviderCapability[] = ['linkedin', 'title', 'company'];

  constructor(
    private specStreamApiKey: string,
    private llmProvider: LLMProvider = 'groq',
    private llmApiKey?: string
  ) {
    super();
    if (!specStreamApiKey) throw new Error('SpecStreamLLM: specStreamApiKey required');
    if (!llmApiKey) throw new Error('SpecStreamLLM: llmApiKey required');
  }

  async enrich(input: EnrichmentInput): Promise<EnrichmentOutput | null> {
    const fullName = input.fullName ?? [input.firstName, input.lastName].filter(Boolean).join(' ');
    if (!fullName && !input.email) return null;

    const queryParts = [
      fullName ? `"${fullName}"` : null,
      input.company ? `"${input.company}"` : null,
      'site:linkedin.com/in',
    ].filter(Boolean);
    const query = queryParts.join(' ');

    const params = new URLSearchParams({ q: query, num: '5' });
    const searchData = await this.httpRequest<SpecStreamSearchResponse>(
      `https://api.specstream.ai/v1/web-search?${params}`,
      {
        headers: { Authorization: `Bearer ${this.specStreamApiKey}` },
        timeoutMs: 15_000,
      }
    );

    // `sources` holds the search results read, in rank order.
    const results = (searchData.data?.sources ?? []).filter(r => r.url);
    if (results.length === 0) return null;

    const extracted = await llmExtract((url, options) => this.httpRequest(url, options), this.llmProvider, this.llmApiKey, {
      query: { fullName, company: input.company, domain: input.domain },
      serpResults: results.slice(0, 5).map(r => ({
        title: r.title ?? '',
        link: r.url ?? '',
        snippet: r.snippet ?? '',
      })),
    });

    if (!extracted || extracted.confidence < 0.5) return null;

    return this.makeOutput({
      fullName: extracted.fullName ?? fullName,
      firstName: input.firstName,
      lastName: input.lastName,
      title: extracted.title,
      company: extracted.company ?? input.company,
      linkedinUrl: extracted.linkedinUrl,
      location: extracted.location,
      confidence: extracted.confidence,
      rawProviderData: { searchResults: results.slice(0, 3), extracted },
    });
  }
}
