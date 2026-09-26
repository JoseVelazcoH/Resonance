import { useRef, useState } from "react";

const SUGGESTION_CHIPS = ["Energético", "Melancólico", "Chill", "Feliz", "Quiero descansar"];

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
        <h1>Moodify</h1>
        <p>Your mood, your music. Powered by AI and Spotify.</p>
      </div>
      {/* The pill is much taller than the input; clicking anywhere on it focuses the input. */}
      <div className="input-container" onClick={() => inputRef.current?.focus()}>
        <input
          ref={inputRef}
          type="text"
          value={prompt}
          placeholder="¿Cómo te sientes hoy? Describe tu estado de ánimo..."
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
        <p className="suggestions-label">o prueba con una sugerencia</p>
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
