import { useCallback, useEffect, useRef, useState } from "react";
import { fetchPlaybackToken, startPlayback } from "../api";

// Spotify Web Playback SDK reference:
// https://developer.spotify.com/documentation/web-playback-sdk
//
// Script: https://sdk.scdn.co/spotify-player.js
// The script calls window.onSpotifyWebPlaybackSDKReady() once loaded, after which
// window.Spotify.Player becomes available. `new Spotify.Player({ name, getOAuthToken,
// volume })` creates a player; player.connect() registers it as a Spotify Connect
// device and fires the "ready" event with { device_id } once registered.
// Events used: ready, not_ready, player_state_changed, initialization_error,
// authentication_error, account_error, playback_error.
// Methods used: connect, disconnect, getCurrentState, togglePlay, seek(position_ms),
// nextTrack, previousTrack, activateElement (required by autoplay policies: must be
// called from within a user gesture handler before the first playback command).
//
// Starting playback itself is done through the Web API's "Start/Resume Playback"
// endpoint (PUT /v1/me/player/play?device_id=...), proxied here through our backend's
// PUT /player/play so the access token never has to be exposed beyond the
// getOAuthToken callback. Because that call sends the full ordered list of track
// URIs as the playback context, the SDK's own nextTrack()/previousTrack() then walk
// that same context, so per-track "next/previous" naturally follows playlist order.

const SDK_SRC = "https://sdk.scdn.co/spotify-player.js";
const PLAYER_NAME = "Resonance";
const POSITION_TICK_MS = 500;

interface SpotifyPlayerTrack {
  id: string | null;
}

interface SpotifyPlaybackState {
  paused: boolean;
  position: number;
  duration: number;
  track_window: { current_track: SpotifyPlayerTrack };
}

interface SpotifyPlayerErrorPayload {
  message: string;
}

interface SpotifyPlayerReadyPayload {
  device_id: string;
}

interface SpotifyPlayer {
  connect: () => Promise<boolean>;
  disconnect: () => void;
  addListener: (event: string, callback: (payload: never) => void) => void;
  removeListener: (event: string) => void;
  getCurrentState: () => Promise<SpotifyPlaybackState | null>;
  togglePlay: () => Promise<void>;
  seek: (positionMs: number) => Promise<void>;
  nextTrack: () => Promise<void>;
  previousTrack: () => Promise<void>;
  activateElement: () => Promise<void>;
}

interface SpotifyPlayerConstructorOptions {
  name: string;
  getOAuthToken: (callback: (token: string) => void) => void;
  volume?: number;
}

declare global {
  interface Window {
    onSpotifyWebPlaybackSDKReady?: () => void;
    Spotify?: {
      Player: new (options: SpotifyPlayerConstructorOptions) => SpotifyPlayer;
    };
  }
}

let sharedSdkLoadPromise: Promise<void> | null = null;

function loadPlaybackSdk(): Promise<void> {
  if (sharedSdkLoadPromise) {
    return sharedSdkLoadPromise;
  }

  sharedSdkLoadPromise = new Promise((resolve, reject) => {
    const previousReady = window.onSpotifyWebPlaybackSDKReady;
    window.onSpotifyWebPlaybackSDKReady = () => {
      previousReady?.();
      resolve();
    };

    if (document.querySelector<HTMLScriptElement>(`script[src="${SDK_SRC}"]`)) {
      return;
    }

    const script = document.createElement("script");
    script.src = SDK_SRC;
    script.async = true;
    script.onerror = () => reject(new Error("Failed to load the Spotify Web Playback SDK script."));
    document.body.appendChild(script);
  });

  return sharedSdkLoadPromise;
}

export type PlaybackErrorKind = "initialization" | "authentication" | "account" | "playback" | null;

export interface UsePlaybackSdkResult {
  isReady: boolean;
  currentTrackId: string | null;
  position: number;
  duration: number;
  isPaused: boolean;
  errorMessage: string | null;
  errorKind: PlaybackErrorKind;
  playTracks: (trackIds: string[], startIndex: number) => Promise<void>;
  togglePlay: () => void;
  seek: (ms: number) => void;
  next: () => void;
  previous: () => void;
}

/**
 * Owns a single Spotify Web Playback SDK player instance for the app's lifetime,
 * mirroring its state and exposing playback actions. Replaces the old Spotify Embed
 * iFrame player: there is now exactly one custom-rendered player (PlayerBar).
 */
export function usePlaybackSdk(): UsePlaybackSdkResult {
  const playerRef = useRef<SpotifyPlayer | null>(null);
  const deviceIdRef = useRef<string | null>(null);
  const activatedRef = useRef(false);

  const [isReady, setIsReady] = useState(false);
  const [currentTrackId, setCurrentTrackId] = useState<string | null>(null);
  const [position, setPosition] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPaused, setIsPaused] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [errorKind, setErrorKind] = useState<PlaybackErrorKind>(null);

  useEffect(() => {
    let cancelled = false;

    loadPlaybackSdk()
      .then(() => {
        if (cancelled || !window.Spotify) {
          return;
        }

        const player = new window.Spotify.Player({
          name: PLAYER_NAME,
          getOAuthToken: (callback) => {
            fetchPlaybackToken()
              .then((token) => callback(token.access_token))
              .catch(() => {
                setErrorKind("authentication");
                setErrorMessage("Could not refresh the Spotify session. Please reconnect Spotify.");
              });
          },
          volume: 0.8,
        });

        player.addListener("ready", (payload: SpotifyPlayerReadyPayload) => {
          deviceIdRef.current = payload.device_id;
          setIsReady(true);
        });
        player.addListener("not_ready", () => {
          setIsReady(false);
        });
        player.addListener("player_state_changed", (state: SpotifyPlaybackState | null) => {
          if (!state) {
            return;
          }
          setIsPaused(state.paused);
          setPosition(state.position);
          setDuration(state.duration);
          setCurrentTrackId(state.track_window.current_track?.id ?? null);
        });
        player.addListener("initialization_error", (payload: SpotifyPlayerErrorPayload) => {
          setErrorKind("initialization");
          setErrorMessage(payload.message);
        });
        player.addListener("authentication_error", (payload: SpotifyPlayerErrorPayload) => {
          setErrorKind("authentication");
          setErrorMessage(payload.message);
        });
        player.addListener("account_error", (payload: SpotifyPlayerErrorPayload) => {
          setErrorKind("account");
          setErrorMessage(payload.message || "Spotify Premium is required to play music in Resonance.");
        });
        player.addListener("playback_error", (payload: SpotifyPlayerErrorPayload) => {
          setErrorKind("playback");
          setErrorMessage(payload.message);
        });

        playerRef.current = player;
        player.connect();
      })
      .catch((error: Error) => {
        if (!cancelled) {
          setErrorKind("initialization");
          setErrorMessage(error.message);
        }
      });

    return () => {
      cancelled = true;
      playerRef.current?.disconnect();
      playerRef.current = null;
      deviceIdRef.current = null;
    };
  }, []);

  // player_state_changed does not fire continuously while a track plays, so poll the
  // current position at a fixed interval to keep the progress bar moving smoothly.
  useEffect(() => {
    if (isPaused) {
      return;
    }
    const intervalId = window.setInterval(() => {
      playerRef.current?.getCurrentState().then((state) => {
        if (state) {
          setPosition(state.position);
        }
      });
    }, POSITION_TICK_MS);
    return () => window.clearInterval(intervalId);
  }, [isPaused]);

  // Browsers require a DOM element to be "activated" from within a user gesture
  // before the SDK is allowed to play audio through it.
  const activateElementOnce = useCallback(() => {
    if (!activatedRef.current && playerRef.current) {
      activatedRef.current = true;
      playerRef.current.activateElement();
    }
  }, []);

  const playTracks = useCallback(
    async (trackIds: string[], startIndex: number) => {
      activateElementOnce();
      if (!deviceIdRef.current) {
        setErrorKind("initialization");
        setErrorMessage("Spotify player is not ready yet. Please wait a moment and try again.");
        return;
      }
      try {
        await startPlayback(deviceIdRef.current, trackIds, trackIds[startIndex]);
        setErrorMessage(null);
        setErrorKind(null);
      } catch (error) {
        setErrorKind("playback");
        setErrorMessage(error instanceof Error ? error.message : "Could not start playback.");
      }
    },
    [activateElementOnce],
  );

  const togglePlay = useCallback(() => {
    activateElementOnce();
    playerRef.current?.togglePlay();
  }, [activateElementOnce]);

  const seek = useCallback((ms: number) => {
    playerRef.current?.seek(ms);
    setPosition(ms);
  }, []);

  const next = useCallback(() => {
    playerRef.current?.nextTrack();
  }, []);

  const previous = useCallback(() => {
    playerRef.current?.previousTrack();
  }, []);

  return {
    isReady,
    currentTrackId,
    position,
    duration,
    isPaused,
    errorMessage,
    errorKind,
    playTracks,
    togglePlay,
    seek,
    next,
    previous,
  };
}
