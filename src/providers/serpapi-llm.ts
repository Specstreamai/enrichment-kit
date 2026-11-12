import { BaseProvider } from './base.js';
import type { EnrichmentInput, EnrichmentOutput, ProviderCapability } from '../types.js';
import { llmExtract, type LLMProvider } from './llm-extract.js';

interface SerpApiResponse {
  organic_results?: Array<{ title?: string; link?: string; snippet?: string; displayed_link?: string }>;
  knowledge_graph?: Record<string, unknown>;
}

export class SerpApiLLMProvider extends BaseProvider {
  name = 'serpapi-llm';
  costPerCall = 0.02;
  typicalHitRate = 0.80;
  capabilities: ProviderCapability[] = ['linkedin', 'title', 'company'];

  constructor(
    private serpApiKey: string,
    private llmProvider: LLMProvider = 'groq',
    private llmApiKey?: string
  ) {
    super();
    if (!serpApiKey) throw new Error('SerpApiLLM: serpApiKey required');
    if (!llmApiKey) throw new Error('SerpApiLLM: llmApiKey required');
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

    const serpParams = new URLSearchParams({
      q: query,
      api_key: this.serpApiKey,
      engine: 'google',
      num: '5',
    });
    const serpData = await this.httpRequest<SerpApiResponse>(
      `https://serpapi.com/search.json?${serpParams}`,
      { timeoutMs: 15_000 }
    );

    const results = serpData.organic_results ?? [];
    if (results.length === 0) return null;

    const extracted = await llmExtract((url, options) => this.httpRequest(url, options), this.llmProvider, this.llmApiKey, {
      query: { fullName, company: input.company, domain: input.domain },
      serpResults: results.slice(0, 5).map(r => ({
        title: r.title ?? '',
        link: r.link ?? '',
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
      rawProviderData: { serpResults: results.slice(0, 3), extracted },
    });
  }
}
