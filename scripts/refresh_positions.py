"""Fetch season rosters resumably, audit coverage, and publish listed positions.

Use a new snapshot directory when refreshing an unfinished season; cached responses
are immutable. Team requests must all succeed before publishing anything.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import CommonTeamRoster

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.refresh_data import Snapshot, publish_file
from src.features.positions import build_positions


def refresh_positions(features, teams, directory, output="data/features"):
    directory = Path(directory)
    if set(features.SEASON) != set(teams.SEASON):
        raise ValueError("Team snapshot must cover exactly the feature seasons")
    jobs = teams[["TEAM_ID", "SEASON"]].drop_duplicates().to_dict("records")

    def fetch(job):
        season, team = str(job["SEASON"]), int(job["TEAM_ID"])
        snapshot = Snapshot(directory / f"{season}_{team}")
        roster = snapshot.fetch("roster", CommonTeamRoster, team_id=team, season=season)
        if not roster.SEASON.astype(str).eq(season[:4]).all():
            raise ValueError(f"Roster season mismatch: {season}, {team}")
        if not roster.TeamID.eq(team).all():
            raise ValueError(f"Roster team mismatch: {season}, {team}")
        roster = roster.copy()
        roster["SEASON"] = season
        roster["TeamID"] = team
        return roster

    frames = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for future in as_completed(futures):
            frames.append(future.result())
    positions = build_positions(features, pd.concat(frames, ignore_index=True))
    audit = features.merge(positions, on=["PLAYER_ID", "SEASON"], validate="one_to_one")
    dates = [
        json.loads(p.read_text())["roster"]["fetched_at"] for p in directory.glob("*/sources.json")
    ]
    metadata = {
        "source": "NBA CommonTeamRoster",
        "source_updated_at": min(dates),
        "source_fetch_completed_at": max(dates),
        "team_season_requests": len(jobs),
        "note": "Listed positions from season-requested rosters, not minutes played by role. Hybrid and differing team labels retain all memberships. Unknown positions are not inferred.",
        "season_coverage": {
            season: {
                "players": len(frame),
                "known": int(frame.POSITION.notna().sum()),
                "rotation_players": int(((frame.GP >= 20) & (frame.MIN >= 15)).sum()),
                "rotation_known": int(
                    ((frame.GP >= 20) & (frame.MIN >= 15) & frame.POSITION.notna()).sum()
                ),
            }
            for season, frame in audit.groupby("SEASON")
        },
        "missing": audit.loc[
            audit.POSITION.isna(), ["PLAYER_ID", "PLAYER_NAME", "SEASON", "GP", "MIN"]
        ].to_dict("records"),
        "label_variations": int(positions.POSITION_LABEL_VARIATION.eq(True).sum()),
    }
    directory.mkdir(parents=True, exist_ok=True)
    positions.to_parquet(directory / "player_positions.parquet", index=False)
    (directory / "player_positions.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name in ["player_positions.parquet", "player_positions.json"]:
        publish_file(directory / name, Path(output) / name)
    print(
        f"Published positions: {positions.POSITION.notna().sum()}/{len(positions)} known",
        flush=True,
    )
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--teams", required=True)
    args = parser.parse_args()
    refresh_positions(
        pd.read_parquet("data/features/player_features.parquet"),
        pd.read_parquet(args.teams),
        args.snapshot_dir,
    )


if __name__ == "__main__":
    main()
