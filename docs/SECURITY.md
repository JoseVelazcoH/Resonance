# Security Policy

## Supported versions

Resonance is a proof of concept. Only the latest commit on `main` receives security fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems. Email
velazco.joseh@gmail.com with:

1. A description of the issue and its impact.
2. Steps to reproduce it.
3. The commit you tested (`git rev-parse --short HEAD`).

You will get an answer within 7 days. Once a fix is available, the issue can be disclosed
publicly with credit to the reporter, if desired.

## Scope

Resonance is meant to run locally, on `127.0.0.1`. Keep that in mind when reporting:

- **In scope:** leaking Spotify tokens or the client secret, session cookie issues, access
  to another user's data through the API, and unsafe handling of data from Spotify or
  LRCLIB.
- **Out of scope:** attacks that require exposing the API to a public network, which is not
  a supported setup, and vulnerabilities in third-party services (Spotify, LRCLIB, Hugging
  Face). Report those to their owners.

## Handling secrets

- `SPOTIFY_CLIENT_SECRET` lives only in `backend/.env`, which is gitignored.
- Spotify access and refresh tokens are stored in the local SQLite database, which is also
  gitignored. Delete `backend/data/app.db` (path set by `APP_DB_PATH`) to remove them.
- If you commit a secret by mistake, rotate it in the
  [Spotify developer dashboard](https://developer.spotify.com/dashboard) right away.
  Removing it from history is not enough.
