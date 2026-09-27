import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import arrowIcon from "../assets/arrow-icon.svg";
import playIcon from "../assets/play-icon.svg";
import { strategyTitle } from "../lib/strategyTitles";
import type { LibraryTrackSummary, PlaylistRecommendResponse, RankedTrack, RecommendJobStatus } from "../types/api";

// The reveal plays in waves rather than one tile at a time (which would take
// far too long for ~1200 tracks): tracks are split into REVEAL_BATCHES groups,
// each group revealed together, spread evenly across REVEAL_TOTAL_MS.
const REVEAL_BATCHES = 16;
const REVEAL_TOTAL_MS = 2000;

// How long the "mood analyzed" morph (heading cross-fade, progress-bar ->
// pill, context strip reveal, grid re-flow) takes end to end. Kept within the
// 1.2-1.8s range asked for; every stage below is a fraction of this budget.
const MORPH_TOTAL_MS = 1500;
const GRID_SETTLE_MS = 700;

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
}

interface AnalysisScreenProps {
  recommendStatus: RecommendJobStatus | null;
  result: PlaylistRecommendResponse | null;
  onViewPlaylist: () => void;
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

export interface TileState {
  id: string;
  name: string;
  cover_url: string | null;
  selected: boolean | null;
}

const Tile = memo(function Tile({
  tile,
  revealed,
  exiting,
}: {
  tile: TileState;
  revealed: boolean;
  exiting: boolean;
}) {
  const outcomeClass = !revealed || tile.selected === null ? "is-pending" : tile.selected ? "is-selected" : "is-rejected";
  return (
    <div className={`track-tile ${outcomeClass}${exiting ? " is-exiting" : ""}`} title={tile.name}>
      {tile.cover_url ? (
        <img className="track-tile-cover" src={tile.cover_url} alt="" loading="lazy" decoding="async" />
      ) : null}
    </div>
  );
});

const SITUATION_CONFIDENCE_THRESHOLD = 0.2;

interface SignalMeter {
  key: string;
  label: string;
  variant: "large" | "small";
}

const SIGNAL_METERS: SignalMeter[] = [
  { key: "feels_bad", label: "Feels Down", variant: "large" },
  { key: "wants_change", label: "Wants A Change", variant: "large" },
  { key: "wants_energy", label: "Energy", variant: "small" },
  { key: "wants_rest", label: "Rest", variant: "small" },
];

export function AnalysisScreen({ recommendStatus, result, onViewPlaylist }: AnalysisScreenProps) {
  const percent = overallPercent(recommendStatus);
  const libraryTracks: LibraryTrackSummary[] | null = recommendStatus?.library_tracks ?? null;
  const rankedTracks: RankedTrack[] | null = result?.ranked_tracks ?? recommendStatus?.result?.ranked_tracks ?? null;

  const tiles: TileState[] = useMemo(() => {
    if (rankedTracks) {
      return rankedTracks.map((track) => ({
        id: track.id,
        name: track.name,
        cover_url: track.cover_url,
        selected: track.selected,
      }));
    }
    if (libraryTracks) {
      return libraryTracks.map((track) => ({
        id: track.id,
        name: track.name,
        cover_url: track.cover_url,
        selected: null,
      }));
    }
    return [];
  }, [rankedTracks, libraryTracks]);

  // Coming back to this screen after the analysis finished must show the final
  // result directly instead of replaying the reveal and morph animations.
  const initiallyDone = useRef(rankedTracks !== null && result !== null).current;

  const [revealedCount, setRevealedCount] = useState(initiallyDone && rankedTracks ? rankedTracks.length : 0);
  const revealStarted = useRef(initiallyDone);

  useEffect(() => {
    if (!rankedTracks || revealStarted.current) {
      return;
    }
    revealStarted.current = true;

    const total = rankedTracks.length;
    const batchSize = Math.max(1, Math.ceil(total / REVEAL_BATCHES));
    const batchCount = Math.ceil(total / batchSize);
    const intervalMs = REVEAL_TOTAL_MS / batchCount;

    const timers: ReturnType<typeof setTimeout>[] = [];
    for (let batch = 1; batch <= batchCount; batch += 1) {
      timers.push(
        setTimeout(() => {
          setRevealedCount(Math.min(total, batch * batchSize));
        }, batch * intervalMs),
      );
    }

    return () => {
      timers.forEach(clearTimeout);
    };
  }, [rankedTracks]);

  const revealDone = rankedTracks !== null && revealedCount >= rankedTracks.length;

  // Stage machine for the in-place morph: "analyzing" (progress bar + full
  // grid) -> "morphing" (non-selected tiles batch-fade out via one CSS rule,
  // heading cross-fades, context strip expands in) -> "done" (final layout:
  // selected tracks as cards, FLIP-animated in from their old tile rect).
  const [stage, setStage] = useState<"analyzing" | "morphing" | "done">(initiallyDone ? "done" : "analyzing");
  const morphStarted = useRef(initiallyDone);
  const tileRectsRef = useRef<Map<string, DOMRect>>(new Map());
  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  useEffect(() => {
    if (!revealDone || morphStarted.current || !result) {
      return;
    }
    morphStarted.current = true;

    if (prefersReducedMotion()) {
      setStage("done");
      return;
    }

    setStage("morphing");
    const timer = setTimeout(() => {
      // Capture the on-screen rect of every selected tile right before the
      // final grid replaces them, so the FLIP effect below can animate from
      // "where the small tile was" to "where the final card lands".
      const rects = new Map<string, DOMRect>();
      document.querySelectorAll<HTMLElement>("[data-tile-id]").forEach((el) => {
        const id = el.dataset.tileId;
        if (id) {
          rects.set(id, el.getBoundingClientRect());
        }
      });
      tileRectsRef.current = rects;
      setStage("done");
    }, GRID_SETTLE_MS);

    return () => clearTimeout(timer);
  }, [revealDone, result]);

  // FLIP: once the final card grid mounts, invert each card from its final
  // position back to the captured tile rect (no transition), then release it
  // to the identity transform on the next frame (with a transition) so the
  // browser animates the reflow instead of jump-cutting to it.
  useLayoutEffect(() => {
    if (stage !== "done" || prefersReducedMotion()) {
      return;
    }
    const rects = tileRectsRef.current;
    if (rects.size === 0) {
      return;
    }

    const cards = Array.from(cardRefs.current.entries());
    cards.forEach(([id, el]) => {
      const from = rects.get(id);
      if (!from) {
        return;
      }
      const to = el.getBoundingClientRect();
      const dx = from.left + from.width / 2 - (to.left + to.width / 2);
      const dy = from.top + from.height / 2 - (to.top + to.height / 2);
      const scale = from.width / to.width;
      el.style.transition = "none";
      el.style.transform = `translate(${dx}px, ${dy}px) scale(${scale})`;
      el.style.opacity = "0.6";
    });

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        cards.forEach(([, el]) => {
          el.style.transition = `transform ${MORPH_TOTAL_MS}ms cubic-bezier(0.22, 1, 0.36, 1), opacity ${MORPH_TOTAL_MS}ms ease`;
          el.style.transform = "none";
          el.style.opacity = "1";
        });
      });
    });

    tileRectsRef.current = new Map();
  }, [stage]);

  const analyzing = stage === "analyzing";
  const morphingOrDone = stage !== "analyzing";
  const done = stage === "done";

  const selectedTracks = result ? result.ranked_tracks.filter((track) => track.selected) : [];
  const showThreshold = done && result !== null && selectedTracks.length === 0;

  const strategy = result ? strategyTitle(result.strategy) : "";
  const emotion = result?.detected.emotion;
  const situation = result?.detected.situation;
  const showSituation = situation !== undefined && situation.confidence >= SITUATION_CONFIDENCE_THRESHOLD;

  return (
    <div className="analysis-content">
      <div className="analysis-heading-crossfade">
        <div className={`analysis-heading-layer${morphingOrDone ? " is-hidden" : ""}`} aria-hidden={morphingOrDone}>
          <h1>Analyzing your mood...</h1>
          <p>Resonance is building a playlist from your whole music library</p>
        </div>
        <div className={`analysis-heading-layer${morphingOrDone ? "" : " is-hidden"}`} aria-hidden={!morphingOrDone}>
          <h1>Mood analyzed</h1>
          <p>Your playlist is ready</p>
        </div>
      </div>

      <div className="progress-container">
        <button
          type="button"
          className={`progress-morph${done ? " is-done" : ""}`}
          onClick={done ? onViewPlaylist : undefined}
          disabled={!done}
          aria-label={done ? "View playlist now" : `Analyzing your mood, ${percent}% complete`}
        >
          <span className="progress-morph-fill" style={{ width: `${done ? 100 : percent}%` }} />
          <span className="progress-morph-label" aria-hidden={!done}>
            {done ? (
              <>
                <img className="view-playlist-button-icon" src={playIcon} alt="" width={16} height={16} />
                <span>View playlist now</span>
                <img className="view-playlist-button-arrow" src={arrowIcon} alt="" width={11.9583} height={10.2083} />
              </>
            ) : (
              "View analysis"
            )}
          </span>
        </button>
        {analyzing && <p className="progress-label">{phaseLabel(recommendStatus)}</p>}
      </div>

      {result && (
        <div className={`analysis-context-strip${morphingOrDone ? " is-expanded" : ""}`}>
          <div className="analysis-context-row">
            <span className="analysis-context-label">ANALYSIS CONTEXT:</span>
            {emotion && (
              <span className="analysis-context-chip">
                <span className="analysis-context-chip-key">Emotion</span>
                <span className="analysis-context-chip-value">{emotion.label}</span>
                <span className="analysis-context-chip-confidence">({Math.round(emotion.confidence * 100)}%)</span>
              </span>
            )}
            {showSituation && situation && (
              <span className="analysis-context-chip">
                <span className="analysis-context-chip-key">Situation</span>
                <span className="analysis-context-chip-value">{situation.label}</span>
                <span className="analysis-context-chip-confidence">({Math.round(situation.confidence * 100)}%)</span>
              </span>
            )}
            <span className="analysis-context-chip">
              <span className="analysis-context-chip-key">Strategy:</span>
              <span className="analysis-context-chip-value">{strategy}</span>
            </span>
          </div>
          <div className="analysis-context-row analysis-meters-row">
            {SIGNAL_METERS.map(({ key, label, variant }) => {
              const value = result.signals[key] ?? 0;
              const meterPercent = Math.round(value * 100);
              return (
                <div className="analysis-meter" key={key}>
                  <span className={`analysis-meter-label${variant === "small" ? " is-small" : ""}`}>{label}</span>
                  <div className={`analysis-meter-track${variant === "small" ? " is-small" : ""}`}>
                    <div
                      className={`analysis-meter-fill${variant === "small" ? " is-small" : ""}`}
                      style={{ width: `${meterPercent}%` }}
                    />
                  </div>
                  <span className={`analysis-meter-value${variant === "small" ? " is-small" : ""}`}>
                    {meterPercent}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {!done && tiles.length > 0 && (
        <div className={`track-grid${stage === "morphing" ? " is-morphing" : ""}`}>
          {tiles.map((tile, index) => (
            <div key={tile.id} data-tile-id={tile.id}>
              <Tile
                tile={tile}
                revealed={rankedTracks !== null && index < revealedCount}
                exiting={stage === "morphing" && tile.selected === false}
              />
            </div>
          ))}
        </div>
      )}

      {done && selectedTracks.length > 0 && (
        <div className="analysis-playlist-grid">
          {selectedTracks.map((track) => (
            <div
              className="analysis-playlist-card"
              key={track.id}
              data-tile-id={track.id}
              ref={(el) => {
                if (el) {
                  cardRefs.current.set(track.id, el);
                } else {
                  cardRefs.current.delete(track.id);
                }
              }}
            >
              <div className="analysis-playlist-cover">
                {track.cover_url ? <img src={track.cover_url} alt="" loading="lazy" decoding="async" /> : null}
              </div>
              <p className="analysis-playlist-card-title">{track.name}</p>
              <p className="analysis-playlist-card-artist">{track.artist}</p>
            </div>
          ))}
        </div>
      )}

      {showThreshold && result && (
        <div className="no-matches-message">
          <p className="no-matches-title">
            Only {result.qualifying_count} songs matched at {Math.round(result.threshold * 100)}% or more
          </p>
        </div>
      )}
    </div>
  );
}
