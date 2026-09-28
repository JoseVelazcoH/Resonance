# Contributing

First of all, thank you so much for taking the time to contribute to Resonance!

## Guidelines

The following is a set of guidelines for contributing to this repository. They are
guidelines, not strict rules, so use your best judgment, and feel free to propose changes
to this document in a pull request.

## Development setup

Resonance is a Python backend (FastAPI, managed with [uv](https://docs.astral.sh/uv/)) and a
React frontend (Vite, npm). You also need a Spotify Premium account and a Spotify app, as
described in the [README](../README.md#getting-started).

```sh
git clone https://github.com/JoseVelazcoH/System-one-agent-spotify-mood-dj.git
cd System-one-agent-spotify-mood-dj
make install    # creates the .env files and installs dependencies
make dev        # runs the API and the web app
make            # lists every command
```

The backend follows a hexagonal layout under `backend/mood_dj/` (`domain`, `application`,
`ports`, `adapters`, `api`), with tests in `backend/tests/`. The frontend lives in
`frontend/src/`.

## Issues

Before opening an issue, please search the existing issues (open and closed) to make sure
the feature or bug you want to propose does not already exist.

### Bugs

A bug report must include the following:

1. The commit you are running (`git rev-parse --short HEAD`).
2. Exact steps to reproduce the bug, including the mood prompt if relevant.
3. A proposed fix or a hypothesis about the cause.

### Features

Feature issues are split into two kinds: **UI** and **Code**.

#### UI

UI issues are specific improvements to the user experience. The proposal must include at
least:

1. The screen you want to improve or add.
2. An image or mockup of the result you want to reach.

#### Code

Code issues are improvements to the codebase itself, whether for better mood detection,
selection or structure. The proposal must include at least:

1. The section you want to improve or add.
2. A proposed solution.

Changes to mood detection or selection thresholds must include before and after numbers on
a labeled set, measured on data that was not used to tune them.

## Pull Requests

To contribute, take one of the open issues in the repository. Anything tagged `type:bug`,
`good-first-issue`, or `help-wanted` would be fantastic. To claim an issue, leave a comment
asking for it and a maintainer will assign it to you.

Open your pull request against `main`, and link the issue it resolves.

### Quality bar

Resonance follows test-driven development: add or update tests alongside your change. Before
you submit, make sure the checks below pass locally:

```sh
make test         # unit tests, no network, no model
make test-laya    # integration tests against the real Laya model, when you touch Laya code
```

Never commit credentials, `.env` files, local databases or personal playlists. Local-only
files use the `*.local.*` suffix and are gitignored.

### Review cycle

To speed up the review cycle, you can allow maintainers to push directly to your branch.
This is only done for small fixes.

### Commits

Commits must follow the [commit convention](commits-convention.md). If a pull request
does not follow it, it will be rejected and you will be asked to correct the commit history.

## AI

We are not at odds with the use of AI. On the contrary, we push for more people to use it
to speed up the production process. That said, using AI correctly matters to us: every
change, and every issue and pull request description, must be tested and understood by a
human.
