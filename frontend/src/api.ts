import type {
  LibraryLyricsStatus,
  LibraryLyricsSummary,
  LibraryStatus,
  MeResponse,
  PlaylistSummary,
  RecommendJobStarted,
  RecommendJobStatus,
  SavePlaylistResponse,
} from "./types/api";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function parseErrorDetail(response: Response): Promise<string> {
  const body = await response.json().catch(() => ({ detail: response.statusText }));
  return body.detail ?? response.statusText ?? "Request failed";
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    credentials: "include",
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorDetail(response));
  }

  return response.json();
}

export function loginUrl(): string {
  return `${API_BASE_URL}/auth/login`;
}

export async function fetchMe(): Promise<MeResponse> {
  return requestJson<MeResponse>("/auth/me");
}

export async function logout(): Promise<void> {
  await requestJson<{ logged_out: boolean }>("/auth/logout", { method: "POST" });
}

export async function fetchPlaylists(): Promise<PlaylistSummary[]> {
  return requestJson<PlaylistSummary[]>("/playlists");
}

export async function prepareLibrary(): Promise<{ started: boolean }> {
  return requestJson<{ started: boolean }>("/library/prepare", {
    method: "POST",
  });
}

export async function fetchLibraryStatus(): Promise<LibraryStatus> {
  return requestJson<LibraryStatus>("/library/status");
}

export async function fetchLibraryLyricsStatus(offset?: number, limit = 50): Promise<LibraryLyricsStatus> {
  const params = new URLSearchParams();
  if (offset !== undefined) {
    params.set("offset", String(offset));
  }
  params.set("limit", String(limit));
  return requestJson<LibraryLyricsStatus>(`/library/lyrics/status?${params.toString()}`);
}

export async function fetchLibraryLyricsSummary(): Promise<LibraryLyricsSummary> {
  return requestJson<LibraryLyricsSummary>("/library/lyrics/summary");
}

export async function startLibraryRecommendJob(prompt: string): Promise<RecommendJobStarted> {
  return requestJson<RecommendJobStarted>("/library/recommend", {
    method: "POST",
    body: JSON.stringify({ prompt }),
  });
}

export async function fetchRecommendJob(jobId: string): Promise<RecommendJobStatus> {
  return requestJson<RecommendJobStatus>(`/recommend-jobs/${jobId}`);
}

export async function savePlaylistToSpotify(
  name: string,
  trackIds: string[],
): Promise<SavePlaylistResponse> {
  return requestJson<SavePlaylistResponse>("/playlists/save", {
    method: "POST",
    body: JSON.stringify({ name, track_ids: trackIds }),
  });
}
