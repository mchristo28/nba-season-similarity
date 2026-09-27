"""Exact shooting, league context, and opportunity-normalized comparison features."""

from src.features.transforms import efficiency_stats
from src.similarity.scoring import MODEL_VERSION  # noqa: F401

TOTAL_STATS = [
    "PTS",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "AST",
    "TOV",
    "OREB",
    "DREB",
    "REB",
    "STL",
    "BLK",
    "MIN",
]
RATE_STATS = [c for c in TOTAL_STATS if c != "MIN"]


def attach_measurements(basic, totals, rates, advanced):
    """Keep endpoint units explicit and require identical player/game coverage."""
    result = basic.copy()
    for frame in (totals, rates, advanced):
        if frame.PLAYER_ID.duplicated().any() or set(frame.PLAYER_ID) != set(basic.PLAYER_ID):
            raise ValueError("Comparison endpoints must cover every base player exactly once")
        games = basic[["PLAYER_ID", "GP"]].merge(frame[["PLAYER_ID", "GP"]], on="PLAYER_ID")
        if not games.GP_x.eq(games.GP_y).all():
            raise ValueError("Comparison endpoint games differ; fetch a new complete snapshot")
    for frame, columns, suffix in [(totals, TOTAL_STATS, "_TOTAL"), (rates, RATE_STATS, "_PER100")]:
        extra = frame[["PLAYER_ID", *columns]].rename(columns={c: c + suffix for c in columns})
        result = result.merge(extra, on="PLAYER_ID", validate="one_to_one")
    # Derive unrounded per-game counts for displays and cross-endpoint ratios.
    for stat in RATE_STATS:
        result[stat] = ratio(result[stat + "_TOTAL"], result.GP)
    return result.merge(
        advanced[["PLAYER_ID", "POSS", "PACE"]], on="PLAYER_ID", validate="one_to_one"
    )


def ratio(numerator, denominator):
    return numerator / denominator.where(denominator > 0)


def comparison_features(frame, teams):
    """Derive features from exact totals. No estimated shooting skill is implied."""
    required = (
        {c + "_TOTAL" for c in TOTAL_STATS} | {c + "_PER100" for c in RATE_STATS} | {"POSS", "PACE"}
    )
    if not required <= set(frame):
        raise ValueError(
            "Exact totals, per-100 rates and possessions are required; run refresh_data.py"
        )
    if teams.empty:
        raise ValueError("Actual team totals are required for league-relative efficiency")
    if teams.duplicated(["TEAM_ID", "SEASON"]).any():
        raise ValueError("Duplicate team-season totals")
    result = efficiency_stats(frame)
    league = teams.groupby("SEASON")[["PTS", "FGA", "FTA"]].sum(min_count=1)
    league_ts = ratio(league.PTS, 2 * (league.FGA + 0.44 * league.FTA))
    result["league_ts_pct"] = result.SEASON.map(league_ts)
    if result.league_ts_pct.isna().any():
        raise ValueError("Missing league shooting baseline")
    result["ts_relative"] = result.ts_pct - result.league_ts_pct
    result["three_attempt_rate"] = ratio(result.FG3A_TOTAL, result.FGA_TOTAL)
    result["free_throw_rate"] = ratio(result.FTA_TOTAL, result.FGA_TOTAL)
    possessions_used = result.FGA_TOTAL + 0.44 * result.FTA_TOTAL + result.TOV_TOTAL
    result["assist_involvement"] = ratio(result.AST_TOTAL, result.AST_TOTAL + possessions_used)
    result["turnover_rate"] = ratio(result.TOV_TOTAL, possessions_used)
    result["offensive_rebound_fraction"] = ratio(result.OREB_TOTAL, result.REB_TOTAL)
    # A player's scoring-action mix is distinct from accuracy; zero made FGs is unknown.
    result["unassisted_fg_share"] = result.PCT_UAST_FGM.where(result.FGM_TOTAL > 0)
    # Tracking was converted once to per game by build_features. These ratios cancel minutes.
    result["passes_per_touch"] = ratio(result.PASSES_MADE, result.TOUCHES)
    result["potential_assists_per_pass"] = ratio(result.POTENTIAL_AST, result.PASSES_MADE)
    result["seconds_per_touch"] = ratio(result.TIME_OF_POSS * 60, result.TOUCHES)
    result["drive_pass_fraction"] = ratio(result.DRIVE_PASSES, result.DRIVES)
    result["drive_shot_fraction"] = ratio(result.DRIVE_FGA, result.DRIVES)
    result["pullup_attempt_share"] = ratio(result.PULL_UP_FGA, result.FGA)
    result["catch_shoot_attempt_share"] = ratio(result.CATCH_SHOOT_FGA, result.FGA)
    return result


def comparison_columns(frame):
    """Columns needed by published profiles and their evidence tables."""
    names = [
        "POSS",
        "PACE",
        "league_ts_pct",
        "ts_relative",
        "three_attempt_rate",
        "free_throw_rate",
        "assist_involvement",
        "turnover_rate",
        "offensive_rebound_fraction",
        "unassisted_fg_share",
        "passes_per_touch",
        "potential_assists_per_pass",
        "seconds_per_touch",
        "drive_pass_fraction",
        "drive_shot_fraction",
        "pullup_attempt_share",
        "catch_shoot_attempt_share",
    ]
    return [c for c in frame if c.endswith(("_TOTAL", "_PER100")) or c in names]
