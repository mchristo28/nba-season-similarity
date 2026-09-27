"""Small, explicit comparison profiles: tendencies and production answer different questions."""

from copy import deepcopy

from src.features.registry import FEATURE_GROUPS


def group(key, features, *, label=None, description=None, weight=None):
    spec = deepcopy(FEATURE_GROUPS[key])
    spec["features"] = features
    if label:
        spec["label"] = label
    if description:
        spec["description"] = spec["desc"] = description
    if weight is not None:
        spec["default_weight"] = weight
    return spec


HISTORICAL_STYLE = {
    "physical": group("physical", ["height_inches", "weight"]),
    "scoring_volume": group(
        "scoring_volume",
        ["free_throw_rate"],
        label="Foul drawing",
        description="Free-throw attempts per field-goal attempt",
    ),
    "shot_profile": group(
        "shot_profile",
        [
            "pct_fga_restricted",
            "pct_fga_paint",
            "pct_fga_midrange",
            "pct_fga_corner3",
            "pct_fga_above_break3",
        ],
    ),
    "shot_creation": group(
        "shot_creation",
        ["unassisted_fg_share"],
        description="Share of made shots created without an assist",
    ),
    "playmaking": group(
        "playmaking",
        ["assist_involvement", "turnover_rate"],
        description="Assist involvement and turnover tendency",
    ),
    "rebounding": group(
        "rebounding",
        ["offensive_rebound_fraction"],
        description="Offensive share of total rebounds",
    ),
    "defense": group(
        "defense",
        ["STL_PER100", "BLK_PER100"],
        label="Defensive activity",
        description="Steals and blocks per 100 possessions; activity, not defensive talent",
    ),
    "usage": group("usage", ["e_usg_pct"]),
}
TRACKING_STYLE = deepcopy(HISTORICAL_STYLE)
TRACKING_STYLE["shot_creation"] = group(
    "shot_creation",
    ["unassisted_fg_share", "pullup_attempt_share", "catch_shoot_attempt_share"],
    description="Unassisted makes, pull-up and catch-and-shoot attempt mix",
)
TRACKING_STYLE["playmaking"] = group(
    "playmaking",
    ["assist_involvement", "turnover_rate", "potential_assists_per_pass"],
    description="Assist involvement, turnovers, potential assists per pass",
)
TRACKING_STYLE["drives"] = group(
    "drives",
    ["drive_pass_fraction", "drive_shot_fraction"],
    description="Pass and shot attempts per drive",
)
TRACKING_STYLE["ball_handling"] = group(
    "ball_handling",
    ["passes_per_touch", "seconds_per_touch", "AVG_DRIB_PER_TOUCH"],
    description="Passes, possession time and dribbles per touch",
)
PRODUCTION = {
    "physical": group("physical", ["height_inches", "weight"], weight=0.25),
    "scoring_volume": group(
        "scoring_volume", ["PTS_PER100"], description="Points per 100 on-court possessions"
    ),
    "scoring_efficiency": group(
        "scoring_efficiency",
        ["ts_relative"],
        description="True shooting percentage points above or below the season league average",
    ),
    "playmaking": group(
        "playmaking",
        ["AST_PER100", "TOV_PER100"],
        description="Assists and turnovers per 100 possessions",
    ),
    "rebounding": group(
        "rebounding",
        ["OREB_PER100", "DREB_PER100"],
        description="Offensive and defensive rebounds per 100 possessions",
    ),
    "defense": group(
        "defense",
        ["STL_PER100", "BLK_PER100"],
        label="Defensive activity",
        description="Steals and blocks per 100 possessions; activity, not defensive talent",
    ),
    "usage": group("usage", ["e_usg_pct"]),
}

PROFILES = {
    "style_historical": {
        "groups": HISTORICAL_STYLE,
        "first_year": 2003,
        "label": "Playing style · historical",
        "description": "Compares shot locations, creation, passing and rebounding tendencies, defensive activity, usage and size. Shooting accuracy and per-game volume do not affect the style score.",
    },
    "style_tracking": {
        "groups": TRACKING_STYLE,
        "first_year": 2013,
        "label": "Playing style · tracking",
        "description": "Adds pull-up and catch-and-shoot mix, choices on drives, passing and handling per touch. Available from 2013–14.",
    },
    "production": {
        "groups": PRODUCTION,
        "first_year": 2003,
        "label": "Production · pace and era adjusted",
        "description": "Compares output per 100 on-court possessions, shooting efficiency relative to that season’s league, usage and size. Minutes do not directly increase production rates.",
    },
}


def get_profile(name):
    if name not in PROFILES:
        raise ValueError(f"Unknown comparison profile: {name}")
    return deepcopy(PROFILES[name])
