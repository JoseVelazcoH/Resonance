import { useEffect, useRef, useState } from "react";
import { usePlaybackSdk } from "./hooks/usePlaybackSdk";
import "./App.css";
import {
  ApiError,
  fetchLibraryLyricsStatus,
  fetchLibraryStatus,
  fetchMe,
  fetchPlaylists,
  fetchRecommendJob,
  loginUrl,
  logout,
  prepareLibrary,
  savePlaylistToSpotify,
  startLibraryRecommendJob,
} from "./api";
import { Header, type Screen } from "./components/Header";
import { AnalysisScreen } from "./screens/AnalysisScreen";
import { HomeScreen } from "./screens/HomeScreen";
import { LyricsDownloadScreen } from "./screens/LyricsDownloadScreen";
import { PlaylistScreen } from "./screens/PlaylistScreen";
import { StartDownloadScreen } from "./screens/StartDownloadScreen";
import type {
  LibraryLyricsStatus,
  LibraryStatus,
  PlaylistRecommendResponse,
  RecommendJobStatus,
} from "./types/api";

const LIBRARY_POLL_INTERVAL_MS = 1500;
const LYRICS_POLL_INTERVAL_MS = 1000;
const RECOMMEND_POLL_INTERVAL_MS = 1000;

function App() {
  // One Spotify Web Playback SDK player for the whole app lifetime. Creating and
  // destroying players per screen visit left the second player without a device,
  // so playback failed after going Home and generating another playlist.
  const playback = usePlaybackSdk();
  const [authChecked, setAuthChecked] = useState(false);
  const [loggedIn, setLoggedIn] = useState(false);
  const [displayName, setDisplayName] = useState<string | null>(null);

  const [screen, setScreen] = useState<Screen>("home");
  const [prompt, setPrompt] = useState("");
  const [libraryError, setLibraryError] = useState<string | null>(null);
  const [lyricsStatus, setLyricsStatus] = useState<LibraryLyricsStatus | null>(null);
  const [libraryPrepareStatus, setLibraryPrepareStatus] = useState<LibraryStatus | null>(null);
  const [recommendStatus, setRecommendStatus] = useState<RecommendJobStatus | null>(null);
  const [playlistResult, setPlaylistResult] = useState<PlaylistRecommendResponse | null>(null);

  const [isRetryingLibrary, setIsRetryingLibrary] = useState(false);

  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedPlaylistId, setSavedPlaylistId] = useState<string | null>(null);
  const [savedPlaylistName, setSavedPlaylistName] = useState<string | null>(null);

  const libraryPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lyricsPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const recommendPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const libraryChecked = useRef(false);

  useEffect(() => {
    if (authChecked) {
      return;
    }
    fetchMe()
      .then((me) => {
        setLoggedIn(me.logged_in);
        setDisplayName(me.display_name);
        setAuthChecked(true);
      })
      .catch(() => {
        setLoggedIn(false);
        setAuthChecked(true);
      });
  }, [authChecked]);

  useEffect(() => {
    if (!authChecked || !loggedIn) {
      return;
    }
    setLibraryError(null);
    fetchPlaylists()
      .then(() => {})
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          setLoggedIn(false);
          return;
        }
        setLibraryError(err instanceof Error ? err.message : "Failed to load playlists");
      });
  }, [authChecked, loggedIn]);

  const stopLibraryPolling = () => {
    if (libraryPollRef.current) {
      clearInterval(libraryPollRef.current);
      libraryPollRef.current = null;
    }
  };

  const stopLyricsPolling = () => {
    if (lyricsPollRef.current) {
      clearInterval(lyricsPollRef.current);
      lyricsPollRef.current = null;
    }
  };

  const stopRecommendPolling = () => {
    if (recommendPollRef.current) {
      clearInterval(recommendPollRef.current);
      recommendPollRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      stopLibraryPolling();
      stopLyricsPolling();
      stopRecommendPolling();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // On first login, call POST /library/prepare once (idempotent: fast when everything is
  // already cached in SQLite) so GET /library/status reflects the real state instead of
  // an "idle" job that never ran. The job always starts with a "reading playlists" phase
  // (a few seconds against the Spotify API) before it knows whether anything actually
  // needs downloading, so we show a neutral "checking" screen during that phase instead
  // of jumping straight to Start Download. Once playlists are read, real counts decide:
  // missing lyrics -> Start Download, lyrics done but profiles pending -> the lyrics/mood
  // progress screen directly, nothing left -> Home.
  useEffect(() => {
    if (!authChecked || !loggedIn || libraryChecked.current) {
      return;
    }
    libraryChecked.current = true;
    setScreen("checking");

    prepareLibrary()
      .then(() => fetchLibraryStatus())
      .then((status) => {
        setLibraryPrepareStatus(status);
        advanceFromCheckingStatus(status);
      })
      .catch((err) => {
        setLibraryError(err instanceof Error ? err.message : "Failed to prepare your library");
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authChecked, loggedIn]);

  // While the job is still only reading playlists (or hasn't reported a real phase yet),
  // its counts are not meaningful, so keep polling and stay on the "checking" screen.
  const isStillReadingPlaylists = (status: LibraryStatus): boolean => {
    return status.state === "idle" || status.phase === "reading playlists";
  };

  const hasMissingLyrics = (status: LibraryStatus): boolean => {
    return status.tracks_total - status.cached > 0 || status.pending > 0;
  };

  const advanceFromCheckingStatus = (status: LibraryStatus) => {
    if (status.state === "error") {
      stopLibraryPolling();
      setLibraryError(status.error ?? "Failed to prepare your library");
      return;
    }

    if (isStillReadingPlaylists(status)) {
      pollCheckingStatus();
      return;
    }

    stopLibraryPolling();
    if (hasMissingLyrics(status)) {
      setScreen("start-download");
      return;
    }
    if (!isMoodProfilingDone(status)) {
      setScreen("lyrics-download");
      pollLyricsStatus();
      return;
    }
    setScreen("home");
  };

  const pollCheckingStatus = () => {
    stopLibraryPolling();
    libraryPollRef.current = setInterval(async () => {
      try {
        const status = await fetchLibraryStatus();
        setLibraryPrepareStatus(status);
        if (status.state === "error" || !isStillReadingPlaylists(status)) {
          stopLibraryPolling();
          advanceFromCheckingStatus(status);
        }
      } catch (err) {
        stopLibraryPolling();
        setLibraryError(err instanceof Error ? err.message : "Failed to check your library");
      }
    }, LIBRARY_POLL_INTERVAL_MS);
  };

  // Mood profiling ("reading mood") happens after lyrics are fetched, so lyrics
  // being "done" is not enough: we also need profiles_processed to have caught
  // up to profiles_total before it's safe to move on to Home.
  const isMoodProfilingDone = (status: LibraryStatus | null): boolean => {
    if (!status) {
      return false;
    }
    return status.profiles_total <= 0 || status.profiles_processed >= status.profiles_total;
  };

  const pollLyricsStatus = () => {
    stopLyricsPolling();
    lyricsPollRef.current = setInterval(async () => {
      try {
        const [status, libraryStatus] = await Promise.all([fetchLibraryLyricsStatus(), fetchLibraryStatus()]);
        setLyricsStatus(status);
        setLibraryPrepareStatus(libraryStatus);
        // Only "done" (pending === 0) AND mood profiling caught up auto-advances
        // to Home; "partial" keeps the user on this screen with a Retry/Continue
        // anyway choice, since some tracks may still be missing lyrics.
        if (status.state === "done" && isMoodProfilingDone(libraryStatus)) {
          stopLyricsPolling();
          stopLibraryPolling();
          setScreen("home");
        } else if (status.state === "error" || libraryStatus.state === "error") {
          stopLyricsPolling();
          stopLibraryPolling();
          setLibraryError("Failed to download lyrics for your library");
        } else if (status.state === "partial") {
          setIsRetryingLibrary(false);
        }
      } catch (err) {
        stopLyricsPolling();
        setLibraryError(err instanceof Error ? err.message : "Failed to check lyrics download status");
      }
    }, LYRICS_POLL_INTERVAL_MS);
  };

  const handleStartDownload = () => {
    setScreen("lyrics-download");
    pollLyricsStatus();
  };

  const handleRetryLibrary = async () => {
    setIsRetryingLibrary(true);
    try {
      await prepareLibrary();
      pollLyricsStatus();
    } catch (err) {
      setLibraryError(err instanceof Error ? err.message : "Failed to retry lyrics download");
      setIsRetryingLibrary(false);
    }
  };

  const handleContinueAnyway = () => {
    stopLyricsPolling();
    stopLibraryPolling();
    setScreen("home");
  };

  const pollRecommendJob = (jobId: string) => {
    stopRecommendPolling();
    recommendPollRef.current = setInterval(async () => {
      try {
        const status = await fetchRecommendJob(jobId);
        setRecommendStatus(status);
        if (status.state === "done") {
          stopRecommendPolling();
          setPlaylistResult(status.result);
          // Stay on the loading screen: it plays the decisions reveal and the
          // playlist-grid highlight animation, then calls back to advance here.
        } else if (status.state === "error") {
          stopRecommendPolling();
          setLibraryError(status.error ?? "Failed to get recommendation");
        }
      } catch (err) {
        stopRecommendPolling();
        setLibraryError(err instanceof Error ? err.message : "Failed to check recommendation status");
      }
    }, RECOMMEND_POLL_INTERVAL_MS);
  };

  const handleSubmitPrompt = async (moodPrompt: string) => {
    setPrompt(moodPrompt);
    setLibraryError(null);
    setPlaylistResult(null);
    setRecommendStatus(null);
    setScreen("analysis");

    try {
      const { job_id } = await startLibraryRecommendJob(moodPrompt);
      pollRecommendJob(job_id);
    } catch (err) {
      setLibraryError(err instanceof Error ? err.message : "Failed to start recommendation");
    }
  };

  const handleTryAnotherMood = () => {
    stopRecommendPolling();
    setSaveError(null);
    setSavedPlaylistId(null);
    setSavedPlaylistName(null);
    setScreen("home");
  };

  const handleSave = async (name: string, trackIds: string[]) => {
    setIsSaving(true);
    setSaveError(null);
    setSavedPlaylistId(null);
    setSavedPlaylistName(null);

    try {
      const response = await savePlaylistToSpotify(name, trackIds);
      setSavedPlaylistId(response.playlist_id);
      setSavedPlaylistName(name);
    } catch (err) {
      if (err instanceof ApiError && (err.status === 401 || err.status === 403)) {
        setSaveError("Missing permission to save playlists. Please reconnect your Spotify account.");
      } else {
        setSaveError(err instanceof Error ? err.message : "Failed to save playlist");
      }
    } finally {
      setIsSaving(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
    } catch {
      // Ignore network errors on logout; we clear local state regardless.
    }
    stopLibraryPolling();
    stopLyricsPolling();
    stopRecommendPolling();
    libraryChecked.current = false;
    setLoggedIn(false);
    setDisplayName(null);
    setLyricsStatus(null);
    setLibraryPrepareStatus(null);
    setRecommendStatus(null);
    setPlaylistResult(null);
    setScreen("home");
  };

  const handleNavigate = (target: Screen) => {
    if (target === "analysis" && !recommendStatus) {
      return;
    }
    if (target === "playlist" && !playlistResult) {
      return;
    }
    setScreen(target);
  };

  const renderScreen = () => {
    if (!authChecked || !loggedIn) {
      return <HomeScreen onSubmitPrompt={handleSubmitPrompt} />;
    }

    if (screen === "checking") {
      return (
        <div className="checking-library">
          <div className="checking-spinner" aria-hidden="true" />
          <p className="checking-label">Checking your library...</p>
        </div>
      );
    }

    if (screen === "start-download") {
      return <StartDownloadScreen onStart={handleStartDownload} />;
    }

    if (screen === "lyrics-download") {
      return (
        <LyricsDownloadScreen
          status={lyricsStatus}
          libraryStatus={libraryPrepareStatus}
          onRetry={handleRetryLibrary}
          onContinueAnyway={handleContinueAnyway}
          isRetrying={isRetryingLibrary}
        />
      );
    }

    if (screen === "analysis") {
      return (
        <AnalysisScreen
          recommendStatus={recommendStatus}
          result={playlistResult}
          onViewPlaylist={() => setScreen("playlist")}
        />
      );
    }

    if (screen === "playlist" && playlistResult) {
      return (
        <PlaylistScreen
          result={playlistResult}
          moodLabel={prompt}
          onTryAnotherMood={handleTryAnotherMood}
          onSave={handleSave}
          isSaving={isSaving}
          saveError={saveError}
          savedPlaylistId={savedPlaylistId}
          savedPlaylistName={savedPlaylistName}
          playback={playback}
        />
      );
    }

    return <HomeScreen onSubmitPrompt={handleSubmitPrompt} />;
  };

  return (
    <div className="app-shell">
      <Header
        activeScreen={screen}
        loggedIn={loggedIn}
        displayName={displayName}
        hasResult={playlistResult !== null}
        hasJobStarted={recommendStatus !== null}
        onNavigate={handleNavigate}
        onConnect={() => {
          window.location.href = loginUrl();
        }}
        onLogout={handleLogout}
      />
      <div className="screen">
        {libraryError && <div className="error-banner">{libraryError}</div>}
        {renderScreen()}
      </div>
    </div>
  );
}

export default App;
