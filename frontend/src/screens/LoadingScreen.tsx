import type { PlaylistSummary, RecommendJobStatus } from "../types/api";

const CARDS_PER_ROW = 5;

interface LoadingScreenProps {
  playlists: PlaylistSummary[];
  recommendStatus: RecommendJobStatus | null;
}

function chunk<T>(items: T[], size: number): T[][] {
  const rows: T[][] = [];
  for (let i = 0; i < items.length; i += size) {
    rows.push(items.slice(i, i + size));
  }
  return rows;
}

function phaseLabel(status: RecommendJobStatus | null): string {
  if (!status) {
    return "Starting...";
  }
  switch (status.phase) {
    case "understanding your mood":
      return "Understanding your mood...";
    case "ranking tracks":
      return "Ranking your tracks...";
    default:
      return "Working on your playlist...";
  }
}

function overallPercent(status: RecommendJobStatus | null): number {
  if (!status || status.total <= 0) {
    return 0;
  }
  return Math.min(100, Math.round((status.processed / status.total) * 100));
}

// This step usually finishes in about 6-20 seconds, so the playlist grid below
// is shown only as static context (which parts of your library are involved),
// not as a live per-playlist progress indicator.
export function LoadingScreen({ playlists, recommendStatus }: LoadingScreenProps) {
  const percent = overallPercent(recommendStatus);
  const rows = chunk(playlists, CARDS_PER_ROW);

  return (
    <div className="loading-content">
      <div className="loading-headings">
        <h1>Analyzing your mood...</h1>
        <p>Moodify is building a playlist from your whole music library</p>
      </div>

      <div className="progress-container">
        <div className="progress-bar-track">
          <div className="progress-bar-fill" style={{ width: `${percent}%` }} />
        </div>
        <p className="progress-label">{phaseLabel(recommendStatus)}</p>
      </div>

      <div className="playlist-grid">
        {rows.map((row, rowIndex) => (
          <div className="playlist-grid-row" key={rowIndex}>
            {row.map((playlist) => (
              <div key={playlist.id} className="playlist-card">
                {playlist.image_url ? (
                  <img className="cover-placeholder" src={playlist.image_url} alt="" />
                ) : (
                  <div className="cover-placeholder" />
                )}
                <p className="playlist-card-name">{playlist.name}</p>
                <p className="playlist-card-count">{playlist.track_count} tracks</p>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
