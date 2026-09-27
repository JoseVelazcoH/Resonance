"""Print per-mood mood calibration diagnostics from a prepared library's cached data.

Read-only: opens the SQLite lyrics/mood-profile tables and never writes to them
(reuses `SqliteLyricsRepository`/`SqliteMoodProfileRepository`, whose
constructors only `CREATE TABLE IF NOT EXISTS`). Safe to point at the real
`backend/data/app.db`, but tests must always pass a `tmp_path` copy instead --
never run this against the real database from an automated test.

Usage:
    uv run scripts/profile_diagnostics.py [db_path] [version]

If `db_path` is omitted, it defaults to `data/app.db` (relative to `backend/`).
If `version` is omitted, it defaults to the current `LayaTrackProfiler` version
computed from the packaged `moods.json`.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from mood_dj.adapters.sqlite_lyrics_repository import SqliteLyricsRepository  # noqa: E402
from mood_dj.adapters.sqlite_mood_profile_repository import SqliteMoodProfileRepository  # noqa: E402
from mood_dj.domain.mood_diagnostics import MoodDiagnosticsReport, compute_diagnostics  # noqa: E402
from mood_dj.domain.models import LyricsStatus  # noqa: E402

DEFAULT_DB_PATH = str(BACKEND_ROOT / "data" / "app.db")


def build_report(db_path: str, version: str) -> MoodDiagnosticsReport:
    """Read-only: build a `MoodDiagnosticsReport` from the SQLite tables at `db_path`.

    Total tracks is approximated as the count of lyrics rows with status
    `lyrics` (the only rows eligible for a mood profile), since the lyrics
    table (unlike the library, which is session-scoped and in-memory) is the
    only persistent, global record of which tracks were ever seen.
    """

    lyrics_repository = SqliteLyricsRepository(db_path)
    mood_profile_repository = SqliteMoodProfileRepository(db_path)

    import sqlite3

    with sqlite3.connect(db_path) as connection:
        (lyrics_bearing_count,) = connection.execute(
            "SELECT COUNT(*) FROM lyrics WHERE status = ?", (LyricsStatus.LYRICS.value,)
        ).fetchone()

    profiles = mood_profile_repository.get_all(version)
    return compute_diagnostics(profiles, total_tracks=lyrics_bearing_count)


def format_report(report: MoodDiagnosticsReport) -> str:
    lines = ["Mood calibration diagnostics", "=" * 40]
    lines.append(
        f"Profiled: {report.profiled_count} / {report.total_tracks} lyrics-bearing tracks "
        f"({report.unprofiled_share:.1%} unprofiled, mean entropy {report.overall_mean_entropy:.3f})"
    )
    lines.append("")
    lines.append(f"{'mood':<14}{'n':>6}{'mean conf':>11}{'mean pos_prob':>15}{'mean entropy':>14}")
    for mood in report.moods:
        lines.append(
            f"{mood.mood_id:<14}{mood.count:>6}"
            f"{mood.mean_confidence:>11.3f}{mood.mean_positive_probability:>15.3f}{mood.mean_entropy:>14.3f}"
        )
    lines.append("")
    lines.append(f"OVERALL profiled={report.profiled_count} mean_entropy={report.overall_mean_entropy:.3f}")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    db_path = argv[1] if len(argv) > 1 else DEFAULT_DB_PATH
    if len(argv) > 2:
        version = argv[2]
    else:
        from mood_dj.adapters.laya_track_profiler import compute_version

        version = compute_version()

    report = build_report(db_path, version)
    print(format_report(report))


if __name__ == "__main__":
    main(sys.argv)
