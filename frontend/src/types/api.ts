export interface Track {
  id: string;
  name: string;
  artist: string;
  album: string;
  energy: number;
  valence: number;
  tempo: number;
  danceability: number;
  acousticness: number;
  instrumentalness: number;
  cover_url: string | null;
  external_url: string | null;
  keep_probability: number | null;
}

export interface Profile {
  energy: number;
  valence: number;
  tempo: number;
  instrumentalness: number;
}

export interface Stage {
  name: string;
  profile: Profile;
  tracks: Track[];
}

export interface RecommendResponse {
  strategy: string;
  strategy_probabilities: Record<string, number>;
  stages: Stage[];
}

export interface MeResponse {
  logged_in: boolean;
  display_name: string | null;
}

export interface PlaylistSummary {
  id: string;
  name: string;
  image_url: string | null;
  track_count: number;
  snapshot_id: string;
}

export type LibraryPrepareState = "idle" | "running" | "done" | "partial" | "error";

export type LibraryPhase = "reading playlists" | "fetching lyrics" | "reading mood";

export interface LibraryStatus {
  state: LibraryPrepareState;
  phase: LibraryPhase | string;
  processed: number;
  total: number;
  cached: number;
  playlists: number;
  tracks: number;
  with_lyrics: number;
  instrumental: number;
  missing: number;
  playlists_processed: number;
  playlists_total: number;
  tracks_processed: number;
  tracks_total: number;
  profiles_processed: number;
  profiles_total: number;
  pending: number;
  failed_transient: number;
  error: string | null;
}

export type LyricsRowStatus = "downloaded" | "downloading" | "pending" | "missing" | "instrumental";

export interface LibraryLyricsRow {
  index: number;
  track_id: string;
  name: string;
  artist: string;
  status: LyricsRowStatus;
}

export interface LibraryLyricsStatus {
  state: LibraryPrepareState;
  phase: LibraryPhase | string;
  total: number;
  processed: number;
  with_lyrics: number;
  instrumental: number;
  missing: number;
  pending: number;
  playlists_processed: number;
  playlists_total: number;
  tracks_processed: number;
  tracks_total: number;
  failed_transient: number;
  rows: LibraryLyricsRow[];
}

export interface LibraryLyricsSummary {
  all_cached: boolean;
}

export interface PlaylistTrack {
  id: string;
  name: string;
  artist: string;
  album: string;
  cover_url: string | null;
  external_url: string | null;
  keep_probability: number;
}

export interface PlaylistStage {
  name: string;
  tracks: PlaylistTrack[];
}

export interface Excluded {
  no_lyrics: number;
  instrumental: number;
  no_profile: number;
}

export interface DetectedEmotion {
  id: string;
  label: string;
  confidence: number;
}

export interface DetectedSituation {
  id: string;
  label: string;
  confidence: number;
}

export interface DetectedTarget {
  valence: number;
  arousal: number;
}

export interface Detected {
  emotion: DetectedEmotion;
  family_id: string;
  situation: DetectedSituation;
  target: DetectedTarget;
}

export interface PlaylistRecommendResponse {
  strategy: string;
  signals: Record<string, number>;
  stages: PlaylistStage[];
  detected: Detected;
  excluded: Excluded;
  qualifying_count: number;
  threshold: number;
}

export interface RecommendJobStarted {
  job_id: string;
}

export interface SavePlaylistResponse {
  playlist_id: string;
}

export type RecommendJobState = "running" | "done" | "error";

export type RecommendPhase = "understanding your mood" | "ranking tracks";

export interface RecommendJobStatus {
  state: RecommendJobState;
  phase: RecommendPhase | string;
  processed: number;
  total: number;
  result: PlaylistRecommendResponse | null;
  error: string | null;
}
