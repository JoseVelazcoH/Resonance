# Resonance

<p align="center">
  <b>A playlist from your own library, decided by how you feel</b>
</p>
<p align="center">
  Describe your mood in plain words. Laya reads your songs, decides a strategy, and builds the playlist from music you already love.
</p>

<div align="center">

  <img src="https://img.shields.io/badge/Status-Proof%20of%20Concept-1DB954?style=for-the-badge&labelColor=0A0A0A" alt="Status" />
  <img src="https://img.shields.io/badge/Stack-Python%20%7C%20FastAPI%20%7C%20React%20%7C%20Laya-1DB954?style=for-the-badge&labelColor=0A0A0A" alt="Stack" />
  <img src="https://img.shields.io/badge/Requires-Spotify%20Premium-1DB954?style=for-the-badge&labelColor=0A0A0A" alt="Requires Spotify Premium" />

</div>

<br>

## <img src="https://api.iconify.design/lucide/telescope.svg?color=%231DB954" width="20" height="20">&nbsp; The Problem
Mood playlists exist, but they are built from someone else's catalog, and the signals that used to make them possible are gone.

<table width="100%">
  <tr>
    <td width="33%" valign="top" align="center">
      <br>
      <h3 align="center"><img src="https://api.iconify.design/lucide/user-x.svg?color=%231DB954" width="18" height="18">&nbsp; Not Your Music</h3>
      <p align="center">Mood playlists come from a global catalog. They rarely sound like the music you actually saved.</p>
      <br>
    </td>
    <td width="33%" valign="top" align="center">
      <br>
      <h3 align="center"><img src="https://api.iconify.design/lucide/ban.svg?color=%231DB954" width="18" height="18">&nbsp; No Audio Features</h3>
      <p align="center">Since late 2024, new Spotify apps get no audio features, no recommendations and no previews from the Web API.</p>
      <br>
    </td>
    <td width="33%" valign="top" align="center">
      <br>
      <h3 align="center"><img src="https://api.iconify.design/lucide/message-square-dashed.svg?color=%231DB954" width="18" height="18">&nbsp; Words, Not Sliders</h3>
      <p align="center">People say "I'm sad but I want to feel better", not "valence 0.4, energy 0.6".</p>
      <br>
    </td>
  </tr>
</table>

<br>

## <img src="https://api.iconify.design/lucide/cpu.svg?color=%231DB954" width="20" height="20">&nbsp; The Solution
Resonance treats it as a **decision problem**, not a recommendation problem. The mood of each song is read once from its **lyrics** and cached. Your prompt is interpreted at request time by [Laya](https://huggingface.co/convaiinnovations/laya), a non-autoregressive decision model that answers typed questions with calibrated probabilities. Ranking is plain math over those cached decisions, so a new prompt takes seconds instead of re-reading your whole library.

### How it works

1.  **Library:** you sign in with Spotify. Resonance reads your own and collaborative playlists and merges their tracks without duplicates. Playlists it created itself are always skipped.
2.  **Lyrics:** each track's lyrics are fetched from [LRCLIB](https://lrclib.net) and stored in SQLite, with retries and backoff when the service is busy. This happens once per track.
3.  **Song mood:** Laya reads the full lyrics and answers two questions: which of 7 moods fits (love, happiness, comfort, sadness, loneliness, anger, fear) and whether the song is positive or negative. The full probability distribution is cached per track.
4.  **Your mood:** Laya reads your prompt and decides the signals (feels down, wants a change, wants energy, wants rest), the emotion, the situation and the strategy: *keep me company*, *lift me up*, *energize* or *calm*.
5.  **Playlist:** every song is scored against your target. Only songs at **65% match or higher** make it in. *Lift me up* builds a progression from melancholic to hopeful to positive.

> Laya never picks songs directly. It interprets text into structured decisions, and those decisions choose the songs. Every step shown on screen is a real model output.

<br>

## <img src="https://api.iconify.design/lucide/flask-conical.svg?color=%231DB954" width="20" height="20">&nbsp; What We Measured

Song mood from lyrics is hard, so every design choice was tested on a small labeled set before shipping.

| **Setup**                                         | **Positive vs negative** | **7 moods**  |
| :------------------------------------------------ | :----------------------: | :----------: |
| 4-level emotion tree, 500-char excerpt            | 45%                      | 25% (family) |
| Flat moods, 500-char excerpt                      | 60%                      | 27-33%       |
| **Flat moods, full lyrics, plain text (shipped)** | **83%**                  | **50%**      |

Sending the full lyrics mattered more than any other change. The set is small (30 songs, AI-labeled), so treat these numbers as direction, not a benchmark. `scripts/eval_mood.py` re-runs the evaluation against your own labels.

<br>

## <img src="https://api.iconify.design/lucide/layers.svg?color=%231DB954" width="20" height="20">&nbsp; Tech Stack
A hexagonal Python backend and a small React app. The only model in the loop is Laya, running locally on CPU.

| **Component**      | **Technology**                                                                                             | **Description**                                                                                  |
| :----------------- | :--------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------- |
| **Backend**        | <img src="https://skillicons.dev/icons?i=python,fastapi,sqlite" valign="middle" />                         | FastAPI with ports and adapters. SQLite caches lyrics, mood profiles and sessions.               |
| **Frontend**       | <img src="https://skillicons.dev/icons?i=react,ts,vite" valign="middle" />                                 | React and TypeScript with plain CSS. Animated analysis screen and a custom player.              |
| **Decision model** | <img src="https://api.iconify.design/lucide/brain.svg?color=%231DB954" width="36" valign="middle" />       | [Laya](https://huggingface.co/convaiinnovations/laya) multilingual checkpoint, batched with `predict_batch`. |
| **Lyrics**         | <img src="https://api.iconify.design/lucide/file-music.svg?color=%231DB954" width="36" valign="middle" />  | [LRCLIB](https://lrclib.net), free and open, no API key.                                         |
| **Spotify**        | <img src="https://skillicons.dev/icons?i=spotify" valign="middle" />                                       | OAuth with PKCE, playlist reading and saving, full-track playback via the Web Playback SDK.      |

<br>

## <img src="https://api.iconify.design/lucide/rocket.svg?color=%231DB954" width="20" height="20">&nbsp; Getting Started

### Requirements

- Python 3.11+ with [uv](https://docs.astral.sh/uv/), and Node.js with npm
- A **Spotify Premium** account (the Web Playback SDK requires it)
- A Spotify app from the [developer dashboard](https://developer.spotify.com/dashboard) with this redirect URI:
  ```
  http://127.0.0.1:8000/auth/callback
  ```

### Run it

1. **Backend:** copy the env file, fill in `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`, then start the API.
   ```bash
   cd backend
   cp .env.example .env
   uv sync
   uv run uvicorn mood_dj.api.main:app --host 127.0.0.1 --port 8000
   ```
2. **Frontend:** in a second terminal.
   ```bash
   cd frontend
   cp .env.example .env
   npm install
   npm run dev
   ```
3. **Open** [http://127.0.0.1:5173](http://127.0.0.1:5173) and connect Spotify. Use `127.0.0.1`, not `localhost`: the session cookie and the Spotify redirect are bound to it.

### Try it on a few playlists first

Create `backend/playlists.local.json` to limit the library to some playlists. The file is gitignored, and names match ignoring case and accents.

```json
{
  "playlist_allowlist": ["Road Trip", "Late Night", "Particula"]
}
```

Delete the file to use every playlist you own or collaborate on.

### Tests

```bash
cd backend
uv run pytest            # unit tests, no network, no model
uv run pytest -m laya    # integration tests against the real Laya model
```

<br>

## <img src="https://api.iconify.design/lucide/map.svg?color=%231DB954" width="20" height="20">&nbsp; Roadmap
- [x] **Spotify sign-in** with PKCE, own and collaborative playlists only
- [x] **Lyrics cache** from LRCLIB with retries, backoff and resumable downloads
- [x] **Song mood profiles** with a full 7-mood distribution per track
- [x] **Prompt decisions** for signals, emotion, situation and strategy
- [x] **65% match threshold** and a *lift me up* progression
- [x] **Animated analysis** that reveals chosen and rejected songs
- [x] **Full-track player** and **Save to Spotify**
- [ ] **Human-labeled evaluation set** in Spanish and English
- [ ] **Better situation detection** for short prompts like "I want popcorn"
- [ ] **GPU or distilled model** to profile large libraries faster
- [ ] **Instrumental tracks**, which have no lyrics to read today

<br>

## <img src="https://api.iconify.design/lucide/heart-handshake.svg?color=%231DB954" width="20" height="20">&nbsp; Acknowledgements
- [Laya](https://huggingface.co/convaiinnovations/laya) by Convai Innovations: the decision model behind every choice
- [LRCLIB](https://lrclib.net): open lyrics that make this possible without scraping
- [Spotify Web API and Web Playback SDK](https://developer.spotify.com/documentation): playlists, saving and playback
- [Bridge](https://github.com/Bonevane/Bridge#readme): for the README style

<br>

> [!IMPORTANT]
> The first run can take several minutes. Resonance downloads the lyrics of every song in your library and reads their mood with Laya on your CPU. Everything is cached in SQLite, so later runs only process new songs.

<br>

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/JoseVelazcoH)

</div>

