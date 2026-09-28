# Commit Convention

This project follows [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).

## Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Rules

- The description must be in lowercase and must not end with a period.
- The body wraps at 72 characters.
- Breaking changes are declared with `!` after the type/scope, or with a `BREAKING CHANGE:` footer.
- One logical change per commit. Split unrelated changes into separate commits.

---

## Types

| Type       | When to use                                              |
|------------|----------------------------------------------------------|
| `feat`     | A new feature visible to the user                        |
| `update`   | A change or improvement to an existing feature           |
| `fix`      | A bug fix                                                |
| `docs`     | Documentation only changes                               |
| `style`    | Formatting, whitespace - no logic change                 |
| `refactor` | Code change that is neither a fix nor a feature          |
| `perf`     | Performance improvement                                  |
| `test`     | Adding or updating tests                                 |
| `build`    | Build system or external dependency changes              |
| `ci`       | CI/CD pipeline changes                                   |
| `chore`    | Maintenance tasks that don't touch production code       |
| `revert`   | Reverts a previous commit                                |

---

## Scopes

Scopes are optional and should match the layer or area being changed.

| Scope       | Area                                                    |
|-------------|---------------------------------------------------------|
| `domain`    | Domain logic (`backend/mood_dj/domain/`)                |
| `app`       | Use cases (`backend/mood_dj/application/`)              |
| `adapters`  | Laya, Spotify, LRCLIB and SQLite adapters               |
| `api`       | FastAPI routes and schemas (`backend/mood_dj/api/`)     |
| `data`      | Moods, emotions, situations and policy data files       |
| `frontend`  | React app (`frontend/`)                                 |
| `player`    | Web Playback SDK integration                            |
| `deps`      | Dependency / packaging changes                          |

---

## Examples

```
feat(frontend): add save to spotify button

update(domain): select tracks with per-mood probability thresholds

fix(player): match relinked tracks by name and artist

test(app): cover lift strategy with an empty comfort stage

docs: add commit convention guide

feat(api)!: return null threshold for moods without a fitted cut-off

BREAKING CHANGE: clients must handle `threshold: null`.
```
