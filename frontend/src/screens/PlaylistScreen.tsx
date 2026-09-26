import { useMemo, useState } from "react";
import musicIcon from "../assets/music.svg";
import type { PlaylistRecommendResponse, PlaylistStage, PlaylistTrack } from "../types/api";

const STRATEGY_TITLES: Record<string, string> = {
  lift: "Lift Me Up",
  accompany: "Keep Me Company",
  energize: "Energize",
  calm: "Calm & Reflective",
};

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
  return `Moodify · ${label} · ${shortDate}`;
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
  const title = STRATEGY_TITLES[result.strategy] ?? result.strategy;
  const [stages, setStages] = useState<PlaylistStage[]>(() =>
    result.stages.map((stage) => ({ ...stage, tracks: [...stage.tracks] })),
  );
  const [playingTrackId, setPlayingTrackId] = useState<string | null>(null);
  const [lastRemoved, setLastRemoved] = useState<RemovedTrack | null>(null);
  const [playlistName, setPlaylistName] = useState(() => buildSuggestedName(title, moodLabel));

  const singleStage = stages.length === 1;
  const eyebrow = singleStage ? `Stage 1: ${stages[0].name}` : `${stages.length} Stages`;
  const thresholdPercent = Math.round(result.threshold * 100);
  const hasQualifyingTracks = result.qualifying_count > 0;
  const trackIds = useMemo(() => stages.flatMap((stage) => stage.tracks.map((track) => track.id)), [stages]);

  const playingTrack = useMemo(() => {
    if (!playingTrackId) {
      return null;
    }
    for (const stage of stages) {
      const found = stage.tracks.find((track) => track.id === playingTrackId);
      if (found) {
        return found;
      }
    }
    return null;
  }, [playingTrackId, stages]);

  const handlePlay = (trackId: string) => {
    setPlayingTrackId((current) => (current === trackId ? null : trackId));
  };

  const handleClosePlayer = () => {
    setPlayingTrackId(null);
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
    if (playingTrackId === trackId) {
      setPlayingTrackId(null);
    }
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
        {stages.map((stage, stageIndex) => (
          <div key={stage.name}>
            {!singleStage && <p className="stage-label">{stage.name}</p>}
            {stage.tracks.map((track) => {
              index += 1;
              const matchPercent = Math.round(track.keep_probability * 100);
              const isPlaying = playingTrackId === track.id;
              return (
                <div
                  key={track.id}
                  className={`song-row${isPlaying ? " is-playing" : ""}`}
                  onClick={() => handlePlay(track.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      handlePlay(track.id);
                    }
                  }}
                >
                  <p className="song-index">{String(index).padStart(2, "0")}</p>
                  {track.cover_url ? (
                    <img className="album-art" src={track.cover_url} alt="" />
                  ) : (
                    <div className="album-art" />
                  )}
                  <div className="song-details">
                    <p className="song-name">{track.name}</p>
                    <p className="song-artist">{track.artist}</p>
                  </div>
                  <div className="match-badge">{matchPercent}% match</div>
                  {track.external_url && (
                    <a
                      className="song-external-link"
                      href={track.external_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label="Open in Spotify"
                      onClick={(event) => event.stopPropagation()}
                    >
                      ↗
                    </a>
                  )}
                  <button
                    type="button"
                    className="song-remove-button"
                    aria-label="Remove from playlist"
                    onClick={(event) => {
                      event.stopPropagation();
                      handleRemove(stageIndex, track.id);
                    }}
                  >
                    ×
                  </button>
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {playingTrack && (
        <div className="track-player">
          <iframe
            key={playingTrack.id}
            title={`Now playing: ${playingTrack.name}`}
            className="track-player-embed"
            src={`https://open.spotify.com/embed/track/${playingTrack.id}`}
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
            loading="lazy"
          />
          <button
            type="button"
            className="track-player-close"
            aria-label="Close player"
            onClick={handleClosePlayer}
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
}
