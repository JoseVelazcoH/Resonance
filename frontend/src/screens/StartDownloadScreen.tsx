import downloadIcon from "../assets/download.svg";

interface StartDownloadScreenProps {
  onStart: () => void;
}

export function StartDownloadScreen({ onStart }: StartDownloadScreenProps) {
  return (
    <div className="start-download-content">
      <div className="start-download-text">
        <h1>Start downloading your lyrics</h1>
        <p>We will analyze the lyrics of your saved tracks to match them perfectly with your mood.</p>
      </div>
      <button type="button" className="download-cta-button" onClick={onStart}>
        <img className="download-cta-icon" src={downloadIcon} alt="" width={16} height={16} />
        Start Download
      </button>
    </div>
  );
}
