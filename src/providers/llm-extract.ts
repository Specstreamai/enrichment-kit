/**
 * LLM extract stage shared by the search + LLM providers (serpapi-llm, specstream-llm).
 *
 * The search step differs per provider; the prompt, the models and the parsing stay
 * the same, so a given set of search results is read the same way whichever search
 * API found them.
 */

export type LLMProvider = 'groq' | 'openai' | 'anthropic';

export interface SearchResultForLLM {
  title: string;
  link: string;
  snippet: string;
}

export interface LLMExtractPayload {
  query: { fullName: string; company?: string; domain?: string };
  serpResults: SearchResultForLLM[];
}

export interface LLMExtracted {
  linkedinUrl: string | null;
  title: string | null;
  company: string | null;
  fullName: string | null;
  location: string | null;
  confidence: number;
}

/** The provider's own `httpRequest`, so timeouts and ProviderError stay the provider's. */
export type HttpRequest = <T = unknown>(
  url: string,
  options?: RequestInit & { timeoutMs?: number }
) => Promise<T>;

export async function llmExtract(
  httpRequest: HttpRequest,
  llmProvider: LLMProvider,
  llmApiKey: string | undefined,
  payload: LLMExtractPayload
): Promise<LLMExtracted | null> {
  const systemPrompt = `You extract person contact data from Google search results.

Return ONLY a JSON object with exactly these keys: linkedinUrl, title, company, fullName, location, confidence.
- linkedinUrl: the most likely linkedin.com/in/... URL for this person, or null
- title: their job title if present in a snippet, or null
- company: their employer if clear, or null
- fullName: their full name as written on LinkedIn, or null
- location: city/country if mentioned, or null
- confidence: 0-1 score for how certain you are this is the right person

Rules:
- Only use linkedin.com/in/ URLs, never /company/ or /pub/.
- If multiple candidates match, pick the one matching the provided company.
- If no result clearly matches the target person, set confidence below 0.5 and most fields to null.
- Do not fabricate data. If a field is not in the search results, return null.`;

  const userPrompt = `Query: ${JSON.stringify(payload)}

Return the JSON object only, no markdown, no prose.`;

  try {
    const response = await callLLM(httpRequest, llmProvider, llmApiKey, systemPrompt, userPrompt);
    const parsed = JSON.parse(response);
    if (typeof parsed.confidence !== 'number') return null;
    return parsed;
  } catch {
    return null;
  }
}

async function callLLM(
  httpRequest: HttpRequest,
  llmProvider: LLMProvider,
  llmApiKey: string | undefined,
  systemPrompt: string,
  userPrompt: string
): Promise<string> {
  if (llmProvider === 'groq') {
    const res = await httpRequest<{ choices: Array<{ message: { content: string } }> }>(
      'https://api.groq.com/openai/v1/chat/completions',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${llmApiKey}`,
        },
        body: JSON.stringify({
          model: 'llama-3.3-70b-versatile',
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt },
          ],
          temperature: 0,
          response_format: { type: 'json_object' },
          max_tokens: 400,
        }),
        timeoutMs: 15_000,
      }
    );
    return res.choices[0]?.message?.content ?? '{}';
  }
  if (llmProvider === 'openai') {
    const res = await httpRequest<{ choices: Array<{ message: { content: string } }> }>(
      'https://api.openai.com/v1/chat/completions',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${llmApiKey}`,
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt },
          ],
          temperature: 0,
          response_format: { type: 'json_object' },
          max_tokens: 400,
        }),
        timeoutMs: 15_000,
      }
    );
    return res.choices[0]?.message?.content ?? '{}';
  }
  const res = await httpRequest<{ content: Array<{ text: string }> }>(
    'https://api.anthropic.com/v1/messages',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': llmApiKey ?? '',
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 400,
        system: systemPrompt,
        messages: [{ role: 'user', content: userPrompt }],
      }),
      timeoutMs: 20_000,
    }
  );
  return res.content[0]?.text ?? '{}';
}
