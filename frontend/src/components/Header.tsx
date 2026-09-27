import brandIndicator from "../assets/brand-indicator.svg";
import circleX from "../assets/circle-x.svg";

export type Screen = "checking" | "home" | "start-download" | "lyrics-download" | "analysis" | "playlist";

interface HeaderProps {
  activeScreen: Screen;
  loggedIn: boolean;
  displayName: string | null;
  hasResult: boolean;
  hasJobStarted: boolean;
  onNavigate: (screen: Screen) => void;
  onConnect: () => void;
  onLogout: () => void;
}

export function Header({
  activeScreen,
  loggedIn,
  displayName,
  hasResult,
  hasJobStarted,
  onNavigate,
  onConnect,
  onLogout,
}: HeaderProps) {
  const navTarget =
    activeScreen === "checking" || activeScreen === "start-download" || activeScreen === "lyrics-download"
      ? "home"
      : activeScreen;

  return (
    <div className="top-header">
      <div className="brand-group">
        <img className="brand-indicator" src={brandIndicator} alt="" width={8} height={8} />
        <p className="brand-name">Resonance</p>
      </div>
      <div className="nav-links">
        <button
          type="button"
          className={`nav-link ${navTarget === "home" ? "is-active" : ""}`}
          onClick={() => onNavigate("home")}
        >
          Home
        </button>
        <button
          type="button"
          className={`nav-link ${navTarget === "analysis" ? "is-active" : ""}`}
          disabled={!hasJobStarted}
          onClick={() => onNavigate("analysis")}
        >
          Analysis
        </button>
        <button
          type="button"
          className={`nav-link ${navTarget === "playlist" ? "is-active" : ""}`}
          disabled={!hasResult}
          onClick={() => onNavigate("playlist")}
        >
          Playlist
        </button>
      </div>
      {loggedIn ? (
        <div className="account-actions">
          <div className="spotify-status">
            <img className="spotify-status-icon" src={circleX} alt="" width={14} height={14} />
            <p className="spotify-status-label">Connected as @{displayName ?? "user"}</p>
          </div>
          <button type="button" className="logout-link" onClick={onLogout}>
            Log out
          </button>
        </div>
      ) : (
        <button type="button" className="connect-button" onClick={onConnect}>
          Connect Spotify
        </button>
      )}
    </div>
  );
}
