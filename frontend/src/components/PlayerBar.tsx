import { loginUrl } from "../api";
import type { PlaybackErrorKind } from "../hooks/usePlaybackSdk";
import type { PlaylistTrack } from "../types/api";

interface PlayerBarProps {
  track: PlaylistTrack | null;
  isPaused: boolean;
  position: number;
  duration: number;
  hasPrev: boolean;
  hasNext: boolean;
  errorMessage: string | null;
  errorKind: PlaybackErrorKind;
  onTogglePlay: () => void;
  onSeek: (ms: number) => void;
  onPrev: () => void;
  onNext: () => void;
}

function formatTime(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

export function PlayerBar({
  track,
  isPaused,
  position,
  duration,
  hasPrev,
  hasNext,
  errorMessage,
  errorKind,
  onTogglePlay,
  onSeek,
  onPrev,
  onNext,
}: PlayerBarProps) {
  if (!track) {
    return null;
  }

  const needsReconnect = errorKind === "authentication" || errorKind === "account";

  const handleSeek = (event: React.MouseEvent<HTMLDivElement>) => {
    if (duration <= 0) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    onSeek(Math.floor(ratio * duration));
  };

  return (
    <div className="player-bar">
      <div className="player-bar-track-info">
        {track.cover_url ? (
          <img className="player-bar-cover" src={track.cover_url} alt="" />
        ) : (
          <div className="player-bar-cover" />
        )}
        <div className="player-bar-track-text">
          <p className="player-bar-track-name">{track.name}</p>
          <p className="player-bar-track-artist">{track.artist}</p>
        </div>
      </div>

      <div className="player-bar-center">
        <div className="player-bar-controls">
          <button
            type="button"
            className="player-bar-secondary-button"
            aria-label="Previous track"
            onClick={onPrev}
            disabled={!hasPrev}
          >
            ⏮
          </button>
          <button type="button" className="player-bar-play-button" aria-label={isPaused ? "Play" : "Pause"} onClick={onTogglePlay}>
            {isPaused ? "▶" : "❚❚"}
          </button>
          <button
            type="button"
            className="player-bar-secondary-button"
            aria-label="Next track"
            onClick={onNext}
            disabled={!hasNext}
          >
            ⏭
          </button>
        </div>
        <div className="player-bar-progress-row">
          <span className="player-bar-time">{formatTime(position)}</span>
          <div className="player-bar-progress-track" onClick={handleSeek}>
            <div
              className="player-bar-progress-fill"
              style={{ width: duration > 0 ? `${Math.min(100, (position / duration) * 100)}%` : "0%" }}
            />
          </div>
          <span className="player-bar-time">{formatTime(duration)}</span>
        </div>
        {errorMessage && (
          <p className="player-bar-error">
            {errorMessage}
            {needsReconnect && (
              <>
                {" "}
                <a className="player-bar-reconnect-link" href={loginUrl()}>
                  Reconnect Spotify
                </a>
              </>
            )}
          </p>
        )}
      </div>
    </div>
  );
}
