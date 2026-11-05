import type { CellResponse, LayersResponse } from "./types";

const DEFAULT_API_URL = "http://localhost:8000";

export const API_URL = (import.meta.env.VITE_API_URL || DEFAULT_API_URL).replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError(0, `API недоступен (${API_URL}): ${String(error)}`);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail) detail = JSON.stringify(body.detail);
    } catch {
      // Тело не JSON: оставляем statusText.
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export function fetchLayers(signal?: AbortSignal): Promise<LayersResponse> {
  return getJson<LayersResponse>("/api/layers", signal);
}

export function fetchCell(lon: number, lat: number, signal?: AbortSignal): Promise<CellResponse> {
  const query = new URLSearchParams({ lon: lon.toFixed(6), lat: lat.toFixed(6) });
  return getJson<CellResponse>(`/api/cell?${query}`, signal);
}
