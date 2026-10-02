import type { CredentialValidators, ProviderExecutors, ProviderProxyExecutor } from "../../core/types.ts";
import type { ProviderActionHandlers } from "../provider-runtime.ts";
import type { ApiKeyProviderContext } from "../provider-runtime.ts";

import { compactObject, looseArray, optionalRecord, optionalString } from "../../core/cast.ts";
import {
  defineApiKeyProviderExecutors,
  defineProviderProxy,
  isAbortLikeError,
  ProviderRequestError,
  providerUserAgent,
  runProviderRequest,
} from "../provider-runtime.ts";

const service = "cheaperinference";
const cheaperinferenceApiBaseUrl = "https://api.cheaperinference.com/v1";
const anthropicApiVersion = "2023-06-01";
const requestLabel = "Cheaper Inference";

type ActionHandler = (input: Record<string, unknown>, context: ApiKeyProviderContext) => Promise<unknown>;
type CheaperinferenceRequestContext = Pick<ApiKeyProviderContext, "apiKey" | "fetcher" | "signal">;

interface CheaperinferenceInferenceInput {
  path: string;
  body: Record<string, unknown>;
  anthropicVersion?: string;
}

const cheaperinferenceActionHandlers: ProviderActionHandlers<"cheaperinference", ActionHandler> = {
  create_chat_completion(input, context) {
    assertStreamingDisabled(input);
    return cheaperinferenceInferenceRequest({ path: "/chat/completions", body: compactObject(input) }, context);
  },
  create_message(input, context) {
    assertStreamingDisabled(input);
    return cheaperinferenceInferenceRequest(
      { path: "/messages", body: compactObject(input), anthropicVersion: anthropicApiVersion },
      context,
    );
  },
  list_models(_input, context) {
    return listCheaperinferenceModels(context, "execute");
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
    const payload = optionalRecord(
      await listCheaperinferenceModels({ apiKey: input.apiKey, fetcher, signal }, "validate"),
    );

    return {
      profile: {
        displayName: "Cheaper Inference API Key",
      },
      grantedScopes: [],
      metadata: {
        validationEndpoint: "/models",
        availableModels: looseArray(payload?.data)
          .map((model) => optionalString(optionalRecord(model)?.id))
          .filter((model): model is string => model !== undefined),
      },
    };
  },
};

function listCheaperinferenceModels(
  context: CheaperinferenceRequestContext,
  mode: "validate" | "execute",
): Promise<unknown> {
  return runProviderRequest({ signal: context.signal, label: requestLabel }, async (signal) => {
    const response = await context.fetcher(`${cheaperinferenceApiBaseUrl}/models`, {
      method: "GET",
      headers: buildCheaperinferenceHeaders(context.apiKey, false),
      signal,
    });
    return readCheaperinferenceResponse(response, mode);
  });
}

// Inference is intentionally not capped by the shared 30 second provider timeout, because long
// completions routinely take longer. Only the caller signal bounds it.
async function cheaperinferenceInferenceRequest(
  input: CheaperinferenceInferenceInput,
  context: CheaperinferenceRequestContext,
): Promise<unknown> {
  let response: Response;
  try {
    response = await context.fetcher(`${cheaperinferenceApiBaseUrl}${input.path}`, {
      method: "POST",
      headers: buildCheaperinferenceHeaders(context.apiKey, true, input.anthropicVersion),
      body: JSON.stringify(input.body),
      signal: context.signal,
    });
  } catch (error) {
    if (isAbortLikeError(error)) {
      throw new ProviderRequestError(504, `${requestLabel} request timed out`);
    }
    throw new ProviderRequestError(
      502,
      error instanceof Error ? `${requestLabel} request failed: ${error.message}` : `${requestLabel} request failed`,
    );
  }

  return readCheaperinferenceResponse(response, "execute");
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

function assertStreamingDisabled(input: Record<string, unknown>): void {
  if (input.stream === true) {
    throw new ProviderRequestError(400, "stream=true is not supported by connector actions");
  }
}

async function readCheaperinferenceResponse(response: Response, mode: "validate" | "execute"): Promise<unknown> {
  await assertCheaperinferenceResponse(response, mode);
  try {
    return await response.json();
  } catch (error) {
    if (isAbortLikeError(error)) {
      throw new ProviderRequestError(504, `${requestLabel} request timed out`);
    }
    throw new ProviderRequestError(502, `${requestLabel} returned malformed JSON`);
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
    (await response.text().catch(() => "")) || `${requestLabel} request failed with status ${response.status}`;

  try {
    const payload = JSON.parse(rawText) as Record<string, unknown>;
    const nestedError = optionalRecord(payload.error);

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
