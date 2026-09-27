"""Resume a complete, validated NBA snapshot refresh without publishing partial pulls.

Run: python scripts/refresh_data.py --snapshot-dir data/raw/refresh_YYYYMMDD
Then review data/features/player_features.json and commit the published artifacts.
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from nba_api.stats import endpoints

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cache_awards import build_season_awards_lookup
from src.data.comprehensive_stats import ComprehensiveStatsPipeline
from src.data.merge_stats import merge_player_measurements
from src.features.build_features import build_features
from src.features.comparison import attach_measurements
from src.features.schema import publish_features, validate_season_features


class Snapshot:
    """Cache successful endpoint responses and their provenance for safe retries."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "sources.json"
        self.manifest = (
            json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        )

    def fetch(self, key, endpoint, *, allow_empty=False, **parameters):
        path = self.directory / f"{key}.parquet"
        entry = self.manifest.get(key)
        if (
            entry
            and entry["endpoint"] == endpoint.__name__
            and entry["parameters"] == parameters
            and path.exists()
        ):
            return pd.read_parquet(path)
        for attempt in range(3):
            time.sleep(0.65)
            try:
                frame = endpoint(timeout=30, **parameters).get_data_frames()[0]
                if frame.empty and not allow_empty:
                    raise ValueError(f"{key}: empty required response")
                # NBA shot-location headers contain NumPy strings in a MultiIndex;
                # normalize before writing so cached and fresh responses have the same schema.
                if isinstance(frame.columns, pd.MultiIndex):
                    frame = normalize(frame, "shots")
                frame.to_parquet(path, index=False)
                self.manifest[key] = {
                    "endpoint": endpoint.__name__,
                    "parameters": parameters,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "rows": len(frame),
                }
                temporary = self.manifest_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.manifest, indent=2) + "\n")
                temporary.replace(self.manifest_path)
                print(f"{key}: {len(frame)} rows", flush=True)
                return frame
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)


def normalize(frame, kind):
    frame = frame.copy()
    if kind == "shots" and isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            f"{zone}_{stat}".lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            if zone
            else str(stat)
            for zone, stat in frame.columns
        ]
    if kind in ("hustle", "estimated"):
        frame.columns = frame.columns.str.lower()
    if kind == "bio":
        frame["height_inches"] = pd.to_numeric(frame.PLAYER_HEIGHT_INCHES, errors="coerce")
        frame["weight"] = pd.to_numeric(frame.PLAYER_WEIGHT, errors="coerce")
        frame = frame[["PLAYER_ID", "height_inches", "weight"]]
    return frame


def fetch_stats(snapshot, start, end):
    frames, teams = [], []
    info = snapshot.fetch("players", endpoints.CommonAllPlayers, is_only_current_season=0)
    for year in range(start, end + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        basic = snapshot.fetch(
            f"basic_{year}",
            endpoints.LeagueDashPlayerStats,
            season=season,
            per_mode_detailed="PerGame",
        ).copy()
        basic["SEASON"] = season
        totals = snapshot.fetch(
            f"totals_{year}",
            endpoints.LeagueDashPlayerStats,
            season=season,
            per_mode_detailed="Totals",
        )
        rates = snapshot.fetch(
            f"per100_{year}",
            endpoints.LeagueDashPlayerStats,
            season=season,
            per_mode_detailed="Per100Possessions",
        )
        advanced = snapshot.fetch(
            f"advanced_{year}",
            endpoints.LeagueDashPlayerStats,
            season=season,
            measure_type_detailed_defense="Advanced",
        )
        basic = attach_measurements(basic, totals, rates, advanced)
        specs = [
            ("shots", endpoints.LeagueDashPlayerShotLocations, {"per_mode_detailed": "Totals"}),
            ("bio", endpoints.LeagueDashPlayerBioStats, {}),
            ("estimated", endpoints.PlayerEstimatedMetrics, {}),
            (
                "scoring",
                endpoints.LeagueDashPlayerStats,
                {"measure_type_detailed_defense": "Scoring", "per_mode_detailed": "PerGame"},
            ),
        ]
        if year >= 2015:
            specs.append(
                ("hustle", endpoints.LeagueHustleStatsPlayer, {"per_mode_time": "PerGame"})
            )
        for kind, endpoint, parameters in specs:
            cache_kind = "shots_totals" if kind == "shots" else kind
            extra = snapshot.fetch(f"{cache_kind}_{year}", endpoint, season=season, **parameters)
            extra = normalize(extra, kind)
            if kind == "shots":
                # Store exact shot counts normalized by GP so distributions share base units.
                games = extra.PLAYER_ID.map(basic.set_index("PLAYER_ID").GP)
                for column in extra:
                    if column.endswith(("_fga", "_fgm")):
                        extra[column] = extra[column] / games.where(games > 0)
            basic = merge_player_measurements(basic, extra)
        if year >= 2013:
            for kind in ("Drives", "CatchShoot", "PullUpShot", "Passing", "Possessions"):
                extra = snapshot.fetch(
                    f"{kind.lower()}_{year}",
                    endpoints.LeagueDashPtStats,
                    season=season,
                    player_or_team="Player",
                    pt_measure_type=kind,
                    per_mode_simple="Totals",
                )
                basic = merge_player_measurements(basic, extra)
        team = snapshot.fetch(
            f"teams_{year}",
            endpoints.LeagueDashTeamStats,
            season=season,
            per_mode_detailed="Totals",
        ).copy()
        team["SEASON"] = season
        teams.append(team)
        frames.append(basic)
    pipeline = ComprehensiveStatsPipeline()
    stats = pipeline.compute_derived_stats(pd.concat(frames, ignore_index=True))
    info = info.rename(columns={"PERSON_ID": "player_id", "FROM_YEAR": "from_year"})
    stats = pipeline.add_career_year(stats, info)
    stats = backfill_physical(stats, snapshot)
    return stats, pd.concat(teams, ignore_index=True)


def backfill_physical(stats, snapshot):
    """Use explicitly identified profile measurements only where season bio is absent."""
    stats = stats.copy()
    missing = stats.height_inches.isna() | stats.weight.isna()
    for player_id in stats.loc[missing, "PLAYER_ID"].unique():
        info = snapshot.fetch(
            f"profile_{int(player_id)}", endpoints.CommonPlayerInfo, player_id=int(player_id)
        ).iloc[0]
        height = str(info.get("HEIGHT", "")).split("-")
        height_inches = (
            int(height[0]) * 12 + int(height[1])
            if len(height) == 2 and all(x.isdigit() for x in height)
            else float("nan")
        )
        weight = pd.to_numeric(info.get("WEIGHT"), errors="coerce")
        player = stats.PLAYER_ID.eq(player_id)
        for column, value in [("height_inches", height_inches), ("weight", weight)]:
            if pd.isna(value) or value <= 0:
                continue
            fill = player & stats[column].isna()
            stats.loc[fill, column] = value
            stats.loc[fill & stats[column].notna(), column + "_source"] = (
                "NBA player profile fallback"
            )
    return stats


def fetch_awards(snapshot, player_ids):
    frames = []
    for player_id in sorted(set(map(int, player_ids))):
        frame = snapshot.fetch(
            f"awards_{player_id}", endpoints.PlayerAwards, player_id=player_id, allow_empty=True
        ).copy()
        if not frame.empty:
            frame["PLAYER_ID"] = player_id
            frames.append(frame)
    if not frames:
        raise ValueError("No awards returned; refusing to publish")
    return pd.concat(frames, ignore_index=True)


def validate_refresh(features, previous, start, end):
    """Reject missing seasons, lost players, or a major tracking coverage regression."""
    validate_season_features(features)
    expected = {f"{year}-{str(year + 1)[-2:]}" for year in range(start, end + 1)}
    if set(features.SEASON) != expected:
        raise ValueError("Refresh does not contain all requested seasons")
    keys = ["PLAYER_ID", "SEASON"]
    prior = previous
    lost = pd.MultiIndex.from_frame(prior[keys]).difference(
        pd.MultiIndex.from_frame(features[keys])
    )
    if len(lost):
        raise ValueError(f"Refresh lost {len(lost)} existing player-seasons; review source changes")
    for season, frame in features.groupby("SEASON"):
        year = int(season[:4])
        required = ["height_inches", "weight"]
        if year >= 2013:
            required += ["DRIVES", "TOUCHES", "CATCH_SHOOT_FGA", "PULL_UP_FGA", "PASSES_MADE"]
        if year >= 2016:
            required += ["deflections"]
        for column in required:
            if column not in frame or frame[column].notna().mean() < 0.9:
                raise ValueError(
                    f"Insufficient {column} coverage in {season}; refusing partial publish"
                )


def publish_file(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-dir",
        required=True,
        help="Use a NEW directory for each refresh; reuse only to resume",
    )
    parser.add_argument("--start-year", type=int, default=2003)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--awards-only", action="store_true")
    args = parser.parse_args()
    snapshot = Snapshot(args.snapshot_dir)
    if args.awards_only:
        # Retired players' historical awards are retained; refresh everyone in the latest season.
        current = snapshot.fetch(
            f"basic_{args.end_year}",
            endpoints.LeagueDashPlayerStats,
            season=f"{args.end_year}-{str(args.end_year + 1)[-2:]}",
            per_mode_detailed="PerGame",
        )
        awards = fetch_awards(snapshot, current.PLAYER_ID)
        old = pd.read_parquet("data/features/player_awards.parquet")
        previous_awarded = set(old.loc[old.PLAYER_ID.isin(current.PLAYER_ID), "PLAYER_ID"])
        if previous_awarded - set(awards.PLAYER_ID):
            raise ValueError(
                "Awards response lost previously awarded players; refusing partial publish"
            )
        combined = pd.concat(
            [old[~old.PLAYER_ID.isin(current.PLAYER_ID)], awards], ignore_index=True
        )
        combined.to_parquet(snapshot.directory / "player_awards.parquet", index=False)
        build_season_awards_lookup(
            str(snapshot.directory / "player_awards.parquet"),
            str(snapshot.directory / "season_awards.parquet"),
        )
        dates = [
            entry["fetched_at"]
            for key, entry in snapshot.manifest.items()
            if key.startswith("awards_")
        ]
        metadata = {
            "source_updated_at": min(dates),
            "source_fetch_completed_at": max(dates),
            "refreshed_player_count": len(current),
            "scope": f"Players appearing in {args.end_year}-{str(args.end_year + 1)[-2:]}; retired players' historical awards retained",
            "records": len(combined),
        }
        awards_metadata = snapshot.directory / "player_awards.json"
        awards_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        for name in ["player_awards.parquet", "season_awards.parquet", "player_awards.json"]:
            publish_file(snapshot.directory / name, Path("data/features") / name)
        print("Published refreshed awards and rebuilt badges.")
        return
    stats, teams = fetch_stats(snapshot, args.start_year, args.end_year)
    stats_path, teams_path = (
        snapshot.directory / "comprehensive_stats.parquet",
        snapshot.directory / "teams.parquet",
    )
    stats.to_parquet(stats_path, index=False)
    teams.to_parquet(teams_path, index=False)
    features = build_features(
        str(stats_path), str(snapshot.directory / "player_features.parquet"), str(teams_path)
    )
    previous = pd.read_parquet("data/features/player_features.parquet")
    validate_refresh(features, previous, args.start_year, args.end_year)
    metadata_path = snapshot.directory / "player_features.json"
    metadata = json.loads(metadata_path.read_text())
    dates = [entry["fetched_at"] for entry in snapshot.manifest.values()]
    metadata.update(
        {
            "source_updated_at": min(dates),
            "source_fetch_completed_at": max(dates),
            "source_note": "NBA regular-season endpoints refreshed; source dates describe this fetch, not a live feed.",
            "availability": {
                "tracking_first_season": "2013-14",
                "hustle_first_season": "2015-16 (partial coverage)",
                "team_shares": "Unavailable for multi-team player seasons",
            },
            "physical_profile_fallbacks": {
                column: stats.loc[
                    stats.get(column + "_source", pd.Series(index=stats.index, dtype=str)).notna(),
                    ["PLAYER_ID", "SEASON"],
                ].to_dict("records")
                for column in ["height_inches", "weight"]
            },
            "physical_profile_note": "Missing season bio measurements use NBA profile height/weight where available; these are listed profile measurements, not verified season-specific measurements.",
            "season_coverage": {
                season: {
                    "players": len(frame),
                    "max_player_games": int(frame.GP.max()),
                    "team_games_min": int(teams.loc[teams.SEASON.eq(season), "GP"].min()),
                    "team_games_max": int(teams.loc[teams.SEASON.eq(season), "GP"].max()),
                    "available_rows": {
                        column: int(frame[column].notna().sum())
                        for column in [
                            "DRIVES",
                            "TOUCHES",
                            "deflections",
                            "height_inches",
                            "weight",
                            "min_share",
                        ]
                    },
                }
                for season, frame in features.groupby("SEASON")
            },
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    # All requests and validation must succeed before any app artifact is replaced.
    publish_features(features, "data/features/player_features.parquet")
    publish_file(metadata_path, "data/features/player_features.json")
    publish_file(stats_path, "data/processed/comprehensive_stats.parquet")
    for season, frame in teams.groupby("SEASON"):
        frame.to_parquet(f"data/raw/team_totals_{season}.parquet", index=False)
    print("Published validated features and updated local rebuild inputs.")


if __name__ == "__main__":
    main()
