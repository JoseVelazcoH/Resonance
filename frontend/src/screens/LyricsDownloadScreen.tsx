import checkCircleIcon from "../assets/check-circle.svg";
import loaderIcon from "../assets/loader.svg";
import type { LibraryLyricsStatus, LibraryStatus, LyricsRowStatus } from "../types/api";

interface LyricsDownloadScreenProps {
  status: LibraryLyricsStatus | null;
  libraryStatus: LibraryStatus | null;
  onRetry: () => void;
  onContinueAnyway: () => void;
  isRetrying: boolean;
}

// Preparation has three phases: reading playlists (fast), fetching lyrics
// (usually the longest), then reading the mood of every track that has
// lyrics. The overall bar below blends all three by these weights. Note the
// backend does not expose which playlist each track/lyrics/profile belongs
// to for the lyrics/mood phases, only playlist-level counts for the
// "reading playlists" phase, so there is no per-playlist grid for those.
const PLAYLISTS_PHASE_WEIGHT = 0.1;
const LYRICS_PHASE_WEIGHT = 0.6;
const MOOD_PHASE_WEIGHT = 0.3;

function statusBadge(status: LyricsRowStatus) {
  switch (status) {
    case "downloaded":
      return (
        <div className="lyrics-badge is-downloaded">
          <img src={checkCircleIcon} alt="" width={10} height={10} />
          <span>Downloaded</span>
        </div>
      );
    case "downloading":
      return (
        <div className="lyrics-badge is-downloading">
          <img src={loaderIcon} alt="" width={10} height={10} className="lyrics-loader-icon" />
          <span>Downloading</span>
        </div>
      );
    case "missing":
      return (
        <div className="lyrics-badge is-subtle">
          <span>Missing</span>
        </div>
      );
    case "instrumental":
      return (
        <div className="lyrics-badge is-subtle">
          <span>Instrumental</span>
        </div>
      );
    default:
      return (
        <div className="lyrics-badge is-pending">
          <span>Pending</span>
        </div>
      );
  }
}

function screenTitle(status: LibraryLyricsStatus | null, libraryStatus: LibraryStatus | null): string {
  if (!status || status.phase === "reading playlists") {
    return "Reading your playlists...";
  }

  if (libraryStatus && libraryStatus.phase === "reading mood") {
    return "Reading the mood of your songs...";
  }

  return "Downloading lyrics...";
}

function progressLabel(status: LibraryLyricsStatus | null, libraryStatus: LibraryStatus | null): string {
  if (!status) {
    return "Reading your playlists...";
  }

  if (status.phase === "reading playlists") {
    return `Reading your playlists: ${status.playlists_processed} of ${status.playlists_total}`;
  }

  if (libraryStatus && libraryStatus.phase === "reading mood") {
    return `Reading the mood of ${libraryStatus.profiles_processed} of ${libraryStatus.profiles_total} tracks`;
  }

  const total = status.tracks_total || status.total;
  const processed = status.tracks_processed || status.processed;
  const pendingSuffix = status.pending > 0 ? ` (${status.pending} pending)` : "";
  return `Downloading lyrics: ${processed} of ${total} tracks${pendingSuffix}`;
}

function overallPercent(status: LibraryLyricsStatus | null, libraryStatus: LibraryStatus | null): number {
  if (!status) {
    return 0;
  }

  const playlistsPct =
    status.playlists_total > 0
      ? Math.min(1, status.playlists_processed / status.playlists_total)
      : status.phase === "reading playlists"
        ? 0
        : 1;

  const tracksTotal = status.tracks_total || status.total;
  const tracksProcessed = status.tracks_processed || status.processed;
  const lyricsPct = tracksTotal > 0 ? Math.min(1, tracksProcessed / tracksTotal) : 0;

  const profilesTotal = libraryStatus?.profiles_total ?? 0;
  const profilesProcessed = libraryStatus?.profiles_processed ?? 0;
  const moodPct = profilesTotal > 0 ? Math.min(1, profilesProcessed / profilesTotal) : 0;

  const combined =
    PLAYLISTS_PHASE_WEIGHT * playlistsPct + LYRICS_PHASE_WEIGHT * lyricsPct + MOOD_PHASE_WEIGHT * moodPct;
  return Math.round(Math.min(100, combined * 100));
}

export function LyricsDownloadScreen({
  status,
  libraryStatus,
  onRetry,
  onContinueAnyway,
  isRetrying,
}: LyricsDownloadScreenProps) {
  const percent = overallPercent(status, libraryStatus);
  const withLyrics = status?.with_lyrics ?? 0;
  const missing = status?.missing ?? 0;
  const rows = status?.rows ?? [];
  const isPartial = status?.state === "partial";
  const pending = status?.pending ?? 0;

  return (
    <div className="lyrics-download-content">
      <div className="lyrics-download-status">
        <h1>{screenTitle(status, libraryStatus)}</h1>
        <div className="progress-container">
          <div className="progress-bar-track">
            <div className="progress-bar-fill" style={{ width: `${percent}%` }} />
          </div>
          <p className="progress-label">{progressLabel(status, libraryStatus)}</p>
          {status && status.phase !== "reading playlists" && (
            <p className="progress-sublabel">
              {withLyrics} with lyrics, {missing} missing
            </p>
          )}
        </div>

        {isPartial && (
          <div className="lyrics-partial-banner">
            <p>
              LRCLIB is busy. {pending} tracks pending, retrying...
            </p>
            <div className="lyrics-partial-actions">
              <button type="button" className="download-cta-button" onClick={onRetry} disabled={isRetrying}>
                {isRetrying ? "Retrying..." : "Retry now"}
              </button>
              <button type="button" className="lyrics-continue-link" onClick={onContinueAnyway}>
                Continue anyway
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="lyrics-table">
        <div className="lyrics-table-header">
          <span className="lyrics-col-index">#</span>
          <span className="lyrics-col-track">Track info</span>
          <span className="lyrics-col-status">Status</span>
        </div>
        <div className="lyrics-table-rows">
          {rows.map((row) => (
            <div className={`lyrics-table-row ${row.status === "pending" ? "is-pending" : ""}`} key={row.track_id}>
              <p className="lyrics-row-index">{String(row.index + 1).padStart(2, "0")}</p>
              <div className="lyrics-row-details">
                <p className="lyrics-row-name">{row.name}</p>
                <p className="lyrics-row-artist">{row.artist}</p>
              </div>
              {statusBadge(row.status)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
