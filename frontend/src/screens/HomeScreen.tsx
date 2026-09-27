import { useRef, useState } from "react";

const SUGGESTION_CHIPS = ["Energetic", "Melancholic", "Chill", "Happy", "I want to rest"];

interface HomeScreenProps {
  onSubmitPrompt: (prompt: string) => void;
}

export function HomeScreen({ onSubmitPrompt }: HomeScreenProps) {
  const [prompt, setPrompt] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const submit = () => {
    const trimmed = prompt.trim();
    if (!trimmed) {
      return;
    }
    onSubmitPrompt(trimmed);
  };

  return (
    <div className="home-content">
      <div className="brand-intro">
        <h1>Resonance</h1>
        <p>Your mood, your music.</p>
      </div>
      {/* The pill is much taller than the input; clicking anywhere on it focuses the input. */}
      <div className="input-container" onClick={() => inputRef.current?.focus()}>
        <input
          ref={inputRef}
          type="text"
          value={prompt}
          placeholder="How are you feeling today? Describe your mood..."
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              submit();
            }
          }}
        />
        <button type="button" className="cta-button" onClick={submit} disabled={!prompt.trim()}>
          Get Playlist
        </button>
      </div>
      <div className="suggestions-wrapper">
        <p className="suggestions-label">or try a suggestion</p>
        <div className="suggestion-chips">
          {SUGGESTION_CHIPS.map((chip) => (
            <button key={chip} type="button" className="chip" onClick={() => onSubmitPrompt(chip)}>
              {chip}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
