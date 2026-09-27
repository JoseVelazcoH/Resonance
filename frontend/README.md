# Resonance frontend

React, TypeScript and Vite. Setup, requirements and the full flow live in the [main README](../README.md).

## Quick path

```bash
cp .env.example .env   # VITE_API_BASE_URL=http://127.0.0.1:8000
npm install
npm run dev            # serves http://127.0.0.1:5173
```

> [!NOTE]
> Open `http://127.0.0.1:5173`, not `localhost`. The Spotify redirect, the session cookie and CORS are all bound to `127.0.0.1`.

## Build

```bash
npm run build          # type-check and production build
```
