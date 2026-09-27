import { useMemo, useState } from "react";
import musicIcon from "../assets/music.svg";
import { PlayerBar } from "../components/PlayerBar";
import { usePlaybackSdk } from "../hooks/usePlaybackSdk";
import { strategyTitle } from "../lib/strategyTitles";
import type { PlaylistRecommendResponse, PlaylistStage, PlaylistTrack } from "../types/api";

function formatDuration(seconds: number): string {
  const totalSeconds = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = totalSeconds % 60;
  return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

interface PlaylistScreenProps {
  result: PlaylistRecommendResponse;
  moodLabel: string;
  onTryAnotherMood: () => void;
  onSave: (name: string, trackIds: string[]) => void;
  isSaving: boolean;
  saveError: string | null;
  savedPlaylistId: string | null;
  savedPlaylistName: string | null;
}

interface RemovedTrack {
  stageIndex: number;
  position: number;
  track: PlaylistTrack;
}

function buildSuggestedName(strategyTitle: string, moodLabel: string): string {
  const label = moodLabel.trim() || strategyTitle;
  const shortDate = new Date().toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return `Resonance · ${label} · ${shortDate}`;
}

export function PlaylistScreen({
  result,
  moodLabel,
  onTryAnotherMood,
  onSave,
  isSaving,
  saveError,
  savedPlaylistId,
  savedPlaylistName,
}: PlaylistScreenProps) {
  const title = strategyTitle(result.strategy);
  const [stages, setStages] = useState<PlaylistStage[]>(() =>
    result.stages.map((stage) => ({ ...stage, tracks: [...stage.tracks] })),
  );
  const {
    currentTrackId,
    isPaused,
    position,
    duration,
    errorMessage,
    errorKind,
    playTracks,
    togglePlay,
    seek,
    next: playNext,
    previous: playPrevious,
  } = usePlaybackSdk();
  const [lastRemoved, setLastRemoved] = useState<RemovedTrack | null>(null);
  const [playlistName, setPlaylistName] = useState(() => buildSuggestedName(title, moodLabel));

  const singleStage = stages.length === 1;
  const eyebrow = singleStage ? `Stage 1: ${stages[0].name}` : `${stages.length} Stages`;
  const thresholdPercent = Math.round(result.threshold * 100);
  const hasQualifyingTracks = result.qualifying_count > 0;
  const trackIds = useMemo(() => stages.flatMap((stage) => stage.tracks.map((track) => track.id)), [stages]);

  const playingTrack = useMemo(() => {
    if (!currentTrackId) {
      return null;
    }
    for (const stage of stages) {
      const found = stage.tracks.find((track) => track.id === currentTrackId);
      if (found) {
        return found;
      }
    }
    return null;
  }, [currentTrackId, stages]);

  const handleRowPlay = (trackId: string) => {
    if (currentTrackId === trackId) {
      togglePlay();
      return;
    }
    const startIndex = trackIds.indexOf(trackId);
    if (startIndex === -1) {
      return;
    }
    void playTracks(trackIds, startIndex);
  };

  const handleRemove = (stageIndex: number, trackId: string) => {
    setStages((prev) => {
      const stage = prev[stageIndex];
      const position = stage.tracks.findIndex((track) => track.id === trackId);
      if (position === -1) {
        return prev;
      }
      setLastRemoved({ stageIndex, position, track: stage.tracks[position] });
      const nextStages = [...prev];
      nextStages[stageIndex] = {
        ...stage,
        tracks: stage.tracks.filter((track) => track.id !== trackId),
      };
      return nextStages;
    });
  };

  const handleUndo = () => {
    if (!lastRemoved) {
      return;
    }
    setStages((prev) => {
      const nextStages = [...prev];
      const stage = nextStages[lastRemoved.stageIndex];
      const nextTracks = [...stage.tracks];
      nextTracks.splice(lastRemoved.position, 0, lastRemoved.track);
      nextStages[lastRemoved.stageIndex] = { ...stage, tracks: nextTracks };
      return nextStages;
    });
    setLastRemoved(null);
  };

  const handleSave = () => {
    onSave(playlistName.trim() || buildSuggestedName(title, moodLabel), trackIds);
  };

  const savedPlaylistUrl = savedPlaylistId ? `https://open.spotify.com/playlist/${savedPlaylistId}` : null;

  let index = 0;

  return (
    <div className="playlist-layout">
      <div className="playlist-summary">
        <div className="metadata-group">
          <p className="metadata-eyebrow">{eyebrow}</p>
          <h1 className="playlist-title">Your Curated Playlist</h1>
          <p className="playlist-description">
            We analyzed your input and calibrated a perfectly harmonized setlist suited for a{" "}
            {title.toLowerCase()} state of mind.
          </p>
        </div>
        <div className="action-group">
          <label className="playlist-name-field">
            <span className="playlist-name-label">Playlist name</span>
            <input
              className="playlist-name-input"
              type="text"
              value={playlistName}
              onChange={(event) => setPlaylistName(event.target.value)}
              disabled={isSaving}
            />
          </label>
          <button
            type="button"
            className="save-button"
            onClick={handleSave}
            disabled={isSaving || trackIds.length === 0}
          >
            <img className="save-button-icon" src={musicIcon} alt="" width={18} height={18} />
            {isSaving ? "Saving..." : "Save to Spotify"}
          </button>
          <button type="button" className="regenerate-button" onClick={onTryAnotherMood}>
            Try another mood
          </button>
          {saveError && <p className="save-feedback is-error">{saveError}</p>}
          {savedPlaylistId && !saveError && (
            <p className="save-feedback">
              Saved as {savedPlaylistName ?? playlistName}.{" "}
              {savedPlaylistUrl && (
                <a className="open-in-spotify-link" href={savedPlaylistUrl} target="_blank" rel="noopener noreferrer">
                  Open in Spotify
                </a>
              )}
            </p>
          )}
        </div>
      </div>

      <div className="songs-container">
        {!hasQualifyingTracks && (
          <div className="no-matches-message">
            <p className="no-matches-title">
              Only {result.qualifying_count} songs matched at {thresholdPercent}% or more
            </p>
            <button type="button" className="regenerate-button" onClick={onTryAnotherMood}>
              Try another mood
            </button>
          </div>
        )}
        {lastRemoved && (
          <div className="undo-banner">
            <p className="undo-banner-text">Removed &ldquo;{lastRemoved.track.name}&rdquo;</p>
            <button type="button" className="undo-button" onClick={handleUndo}>
              Undo
            </button>
          </div>
        )}
        {!singleStage ? null : (
          <div className="song-table-header">
            <span className="song-table-header-index">#</span>
            <span className="song-table-header-title">Title</span>
            <span className="song-table-header-album">Album</span>
            <span className="song-table-header-match">Match</span>
            <span className="song-table-header-duration">Duration</span>
          </div>
        )}
        {stages.map((stage, stageIndex) => (
          <div key={stage.name}>
            {!singleStage && <p className="stage-label">{stage.name}</p>}
            {stage.tracks.map((track) => {
              index += 1;
              const matchPercent = Math.round(track.keep_probability * 100);
              const isCurrentTrack = currentTrackId === track.id;
              const isPlaying = isCurrentTrack && !isPaused;
              return (
                <div key={track.id} className={`song-row${isCurrentTrack ? " is-playing" : ""}`}>
                  <div className="song-index-cell">
                    <span className="song-index">{String(index).padStart(2, "0")}</span>
                    <button
                      type="button"
                      className="song-play-button"
                      aria-label={isPlaying ? "Pause" : "Play"}
                      onClick={() => handleRowPlay(track.id)}
                    >
                      {isPlaying ? "❚❚" : "▶"}
                    </button>
                  </div>
                  {track.cover_url ? (
                    <img className="album-art" src={track.cover_url} alt="" />
                  ) : (
                    <div className="album-art" />
                  )}
                  <div className="song-details">
                    <span className="song-name">
                      {track.name}
                      {track.external_url && (
                        <a
                          className="song-external-link"
                          href={track.external_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          aria-label="Open in Spotify"
                        >
                          ↗
                        </a>
                      )}
                    </span>
                    <p className="song-artist">{track.artist}</p>
                  </div>
                  <p className="song-album">{track.album}</p>
                  <div className="match-badge">{matchPercent}% match</div>
                  <p className="song-duration">{formatDuration(track.duration_s)}</p>
                  <button
                    type="button"
                    className="song-remove-button"
                    aria-label="Remove from playlist"
                    onClick={() => handleRemove(stageIndex, track.id)}
                  >
                    ×
                  </button>
                </div>
              );
            })}
          </div>
        ))}
      </div>

      <PlayerBar
        track={playingTrack}
        isPaused={isPaused}
        position={position}
        duration={duration}
        hasPrev={currentTrackId !== null && trackIds.indexOf(currentTrackId) > 0}
        hasNext={currentTrackId !== null && trackIds.indexOf(currentTrackId) < trackIds.length - 1}
        errorMessage={errorMessage}
        errorKind={errorKind}
        onTogglePlay={togglePlay}
        onSeek={seek}
        onPrev={playPrevious}
        onNext={playNext}
      />
    </div>
  );
}
