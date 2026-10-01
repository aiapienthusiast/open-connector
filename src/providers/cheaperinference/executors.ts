import type { CredentialValidators, ProviderExecutors, ProviderProxyExecutor } from "../../core/types.ts";
import type { ProviderActionHandlers } from "../provider-runtime.ts";
import type { ApiKeyProviderContext } from "../provider-runtime.ts";

import { compactObject, optionalString } from "../../core/cast.ts";
import {
  defineApiKeyProviderExecutors,
  defineProviderProxy,
  ProviderRequestError,
  providerUserAgent,
} from "../provider-runtime.ts";

const service = "cheaperinference";
const cheaperinferenceApiBaseUrl = "https://api.cheaperinference.com/v1";
const anthropicApiVersion = "2023-06-01";

type QueryValue = string | number | boolean | undefined;
type ActionHandler = (input: Record<string, unknown>, context: ApiKeyProviderContext) => Promise<unknown>;

interface CheaperinferenceRequestInput {
  method?: "GET" | "POST";
  path: string;
  query?: Record<string, QueryValue>;
  body?: Record<string, unknown>;
  anthropicVersion?: string;
  mode?: "validate" | "execute";
}

const cheaperinferenceActionHandlers: ProviderActionHandlers<"cheaperinference", ActionHandler> = {
  create_chat_completion(input, context) {
    assertStreamingDisabled(input);
    return cheaperinferenceRequest(
      context.apiKey,
      {
        method: "POST",
        path: "/chat/completions",
        body: compactObject(input),
      },
      context,
    );
  },
  create_message(input, context) {
    assertStreamingDisabled(input);
    return cheaperinferenceRequest(
      context.apiKey,
      {
        method: "POST",
        path: "/messages",
        body: compactObject(input),
        anthropicVersion: anthropicApiVersion,
      },
      context,
    );
  },
  list_models(_input, context) {
    return cheaperinferenceRequest(context.apiKey, { path: "/models" }, context);
  },
};

export const executors: ProviderExecutors = defineApiKeyProviderExecutors(service, cheaperinferenceActionHandlers);

export const proxy: ProviderProxyExecutor = defineProviderProxy({
  service,
  baseUrl: cheaperinferenceApiBaseUrl,
  auth: { type: "api_key_authorization", prefix: "Bearer " },
  skipDnsValidation: true,
});

export const credentialValidators: CredentialValidators = {
  async apiKey(input, { fetcher, signal }) {
    const payload = (await cheaperinferenceRequest(
      input.apiKey,
      {
        path: "/models",
        mode: "validate",
      },
      {
        apiKey: input.apiKey,
        fetcher,
        signal,
      },
    )) as {
      data?: Array<{ id?: unknown }>;
    };

    return {
      profile: {
        displayName: "Cheaper Inference API Key",
      },
      grantedScopes: [],
      metadata: {
        validationEndpoint: "/models",
        availableModels: (payload.data ?? [])
          .map((model) => model.id)
          .filter((model): model is string => typeof model === "string"),
      },
    };
  },
};

async function cheaperinferenceRequest(
  apiKey: string,
  input: CheaperinferenceRequestInput,
  context: Pick<ApiKeyProviderContext, "apiKey" | "fetcher" | "signal">,
): Promise<unknown> {
  const url = buildCheaperinferenceUrl(input.path, input.query ?? {});
  let response: Response;
  try {
    response = await context.fetcher(url, {
      method: input.method ?? "GET",
      headers: buildCheaperinferenceHeaders(apiKey, input.body != null, input.anthropicVersion),
      body: input.body == null ? undefined : JSON.stringify(input.body),
      signal: context.signal,
    });
  } catch (error) {
    throw new ProviderRequestError(
      502,
      error instanceof Error
        ? `Cheaper Inference request failed: ${error.message}`
        : "Cheaper Inference request failed",
    );
  }

  await assertCheaperinferenceResponse(response, input.mode ?? "execute");
  return response.json() as Promise<unknown>;
}

function buildCheaperinferenceHeaders(
  apiKey: string,
  includeJsonContentType: boolean,
  anthropicVersion?: string,
): Headers {
  const headers = new Headers({
    authorization: `Bearer ${apiKey}`,
    "user-agent": providerUserAgent,
  });

  if (includeJsonContentType) {
    headers.set("content-type", "application/json");
  }
  if (anthropicVersion) {
    headers.set("anthropic-version", anthropicVersion);
  }

  return headers;
}

function buildCheaperinferenceUrl(path: string, query: Record<string, QueryValue>): string {
  const url = new URL(`${cheaperinferenceApiBaseUrl}${path}`);
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined) {
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

function assertStreamingDisabled(input: Record<string, unknown>): void {
  if (input.stream === true) {
    throw new ProviderRequestError(400, "stream=true is not supported by connector actions");
  }
}

async function assertCheaperinferenceResponse(response: Response, mode: "validate" | "execute"): Promise<void> {
  if (response.ok) {
    return;
  }

  const error = await readCheaperinferenceError(response);
  if (response.status === 429) {
    throw new ProviderRequestError(429, error.message, error);
  }
  if (mode === "validate" && (response.status === 401 || response.status === 403)) {
    throw new ProviderRequestError(400, error.message, error);
  }
  if (mode === "execute" && response.status === 401) {
    throw new ProviderRequestError(401, error.message, error);
  }
  if (response.status === 400 || response.status === 404 || response.status === 413 || response.status === 422) {
    throw new ProviderRequestError(400, error.message, error);
  }

  throw new ProviderRequestError(response.status || 502, error.message, error);
}

async function readCheaperinferenceError(response: Response): Promise<{
  type: string;
  code?: string | number;
  message: string;
}> {
  const rawText =
    (await response.text().catch(() => "")) || `Cheaper Inference request failed with status ${response.status}`;

  try {
    const payload = JSON.parse(rawText) as Record<string, unknown>;
    const nestedError = optionalRecordFrom(payload.error);

    return {
      type: optionalString(nestedError?.type) ?? optionalString(payload.type) ?? "provider_error",
      code: readErrorCode(nestedError?.code),
      message: optionalString(nestedError?.message) ?? optionalString(payload.message) ?? rawText,
    };
  } catch {
    return {
      type: "provider_error",
      message: rawText,
    };
  }
}

function readErrorCode(value: unknown): string | number | undefined {
  return typeof value === "string" || typeof value === "number" ? value : undefined;
}

function optionalRecordFrom(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}
