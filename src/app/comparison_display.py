"""Readable labels for the actual normalized comparison inputs."""

BANDS = {
    "close": ("Close", "match-strong", "#2a9d5c"),
    "noticeable": ("Noticeable difference", "match-mid", "#c9a227"),
    "large": ("Large difference", "match-weak", "#c44536"),
    "unavailable": ("Unavailable", "", "#888888"),
}

LABELS = {
    "PTS_PER100": ("PTS / 100 poss", False),
    "AST_PER100": ("AST / 100 poss", False),
    "TOV_PER100": ("TOV / 100 poss", False),
    "OREB_PER100": ("OREB / 100 poss", False),
    "DREB_PER100": ("DREB / 100 poss", False),
    "STL_PER100": ("STL / 100 poss", False),
    "BLK_PER100": ("BLK / 100 poss", False),
    "ts_relative": ("TS above league (pp)", True),
    "height_inches": ("Height", False),
    "weight": ("Weight", False),
    "free_throw_rate": ("FTA / FGA", False),
    "pct_fga_restricted": ("Rim attempt share", True),
    "pct_fga_paint": ("Paint attempt share", True),
    "pct_fga_midrange": ("Midrange attempt share", True),
    "pct_fga_corner3": ("Corner 3 attempt share", True),
    "pct_fga_above_break3": ("Above-break 3 share", True),
    "unassisted_fg_share": ("Unassisted makes", True),
    "assist_involvement": ("Assist involvement", True),
    "turnover_rate": ("Turnover tendency", True),
    "offensive_rebound_fraction": ("Offensive rebound share", True),
    "e_usg_pct": ("Usage", True),
    "pullup_attempt_share": ("Pull-up attempt share", True),
    "catch_shoot_attempt_share": ("Catch-and-shoot share", True),
    "potential_assists_per_pass": ("Potential AST per pass", False),
    "drive_pass_fraction": ("Passes per drive", False),
    "drive_shot_fraction": ("FGA per drive", False),
    "passes_per_touch": ("Passes per touch", False),
    "seconds_per_touch": ("Seconds per touch", False),
    "AVG_DRIB_PER_TOUCH": ("Dribbles per touch", False),
}


def profile_categories(groups):
    return [
        {
            "name": spec["label"],
            "stats": [(LABELS[c][0], c, LABELS[c][1]) for c in spec["features"]],
        }
        for spec in groups.values()
    ]


TWO_DECIMAL_FEATURES = {
    "free_throw_rate",
    "potential_assists_per_pass",
    "drive_pass_fraction",
    "drive_shot_fraction",
    "passes_per_touch",
}
