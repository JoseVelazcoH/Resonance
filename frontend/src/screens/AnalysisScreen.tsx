import { useEffect, useState } from "react";
import type { PlaylistRecommendResponse } from "../types/api";

const STRATEGY_TITLES: Record<string, string> = {
  lift: "Lift Me Up",
  accompany: "Keep Me Company",
  energize: "Energize",
  calm: "Calm & Reflective",
};

const SIGNAL_BARS: { key: string; label: string }[] = [
  { key: "feels_bad", label: "Feels Down" },
  { key: "wants_change", label: "Wants A Change" },
  { key: "wants_energy", label: "Wants Energy" },
  { key: "wants_rest", label: "Wants Rest" },
];

const AUTO_ADVANCE_SECONDS = 10;
const SITUATION_CONFIDENCE_THRESHOLD = 0.2;

interface AnalysisScreenProps {
  result: PlaylistRecommendResponse;
  onViewPlaylist: () => void;
}

export function AnalysisScreen({ result, onViewPlaylist }: AnalysisScreenProps) {
  const title = STRATEGY_TITLES[result.strategy] ?? result.strategy;
  const excludedTotal = result.excluded.no_lyrics + result.excluded.instrumental + result.excluded.no_profile;
  const [secondsLeft, setSecondsLeft] = useState(AUTO_ADVANCE_SECONDS);

  const emotion = result.detected.emotion;
  const situation = result.detected.situation;
  const showSituation = situation.confidence >= SITUATION_CONFIDENCE_THRESHOLD;

  useEffect(() => {
    if (secondsLeft <= 0) {
      onViewPlaylist();
      return;
    }
    const timer = setTimeout(() => setSecondsLeft((value) => value - 1), 1000);
    return () => clearTimeout(timer);
  }, [secondsLeft, onViewPlaylist]);

  return (
    <div className="analysis-content">
      <div className="analysis-header">
        <p className="analysis-eyebrow">Detected Strategy</p>
        <h1 className="analysis-title">{title}</h1>
      </div>

      <div className="detected-summary">
        <div className="detected-row">
          <span className="detected-label">Emotion</span>
          <span className="detected-value">
            {emotion.label} ({Math.round(emotion.confidence * 100)}%)
          </span>
        </div>
        {showSituation && (
          <div className="detected-row">
            <span className="detected-label">Situation</span>
            <span className="detected-value">
              {situation.label} ({Math.round(situation.confidence * 100)}%)
            </span>
          </div>
        )}
      </div>

      <div className="parameters-card">
        {SIGNAL_BARS.map(({ key, label }) => {
          const value = result.signals[key] ?? 0;
          const percent = Math.round(value * 100);
          return (
            <div className="param-row" key={key}>
              <div className="bar-labels">
                <span className="bar-name">{label}</span>
                <span className="bar-value">{percent}%</span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${percent}%` }} />
              </div>
            </div>
          );
        })}
      </div>

      <p className="analysis-disclaimer">
        Excluded {excludedTotal} tracks from the mood analysis (missing lyrics, instrumental, or no mood profile
        yet).
      </p>

      <button type="button" className="view-playlist-button" onClick={onViewPlaylist}>
        View playlist now ({secondsLeft}s)
      </button>
    </div>
  );
}
