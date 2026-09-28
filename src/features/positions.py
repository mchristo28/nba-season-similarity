"""Season roster labels and overlapping, broad position memberships."""

import pandas as pd

POSITION_GROUPS = {"G": "Guard", "F": "Forward / Wing", "C": "Center / Big"}


def position_groups(label):
    """Keep hybrid memberships; unknown labels are never inferred from measurements."""
    if not isinstance(label, str):
        return ()
    tokens = set(label.replace("/", "-").split("-"))
    return tuple(group for group in POSITION_GROUPS if group in tokens)


def position_mask(frame, groups):
    groups = set(groups)
    if not groups or not groups <= POSITION_GROUPS.keys():
        raise ValueError("Choose a valid position peer group")
    return frame.get("POSITION", pd.Series(index=frame.index, dtype=str)).map(
        lambda label: bool(groups.intersection(position_groups(label)))
    )


def build_positions(features, rosters):
    """Union labels across team stints, then left-join the actual season population."""
    required = {"PLAYER_ID", "SEASON", "POSITION", "TeamID"}
    if not required <= set(rosters):
        raise ValueError("Roster response lacks position identifiers")
    records = []
    for (pid, season), rows in rosters.groupby(["PLAYER_ID", "SEASON"]):
        labels = sorted({str(x).strip() for x in rows.POSITION.dropna() if str(x).strip()})
        groups = {g for label in labels for g in position_groups(label)}
        records.append(
            {
                "PLAYER_ID": int(pid),
                "SEASON": str(season),
                "POSITION": "-".join(g for g in POSITION_GROUPS if g in groups) or None,
                "POSITION_LABELS": "/".join(labels) or None,
                "POSITION_TEAMS": ",".join(str(int(x)) for x in sorted(rows.TeamID.unique())),
                "POSITION_LABEL_VARIATION": len(set(position_groups(x) for x in labels)) > 1,
            }
        )
    result = features[["PLAYER_ID", "SEASON"]].merge(
        pd.DataFrame(records), on=["PLAYER_ID", "SEASON"], how="left", validate="one_to_one"
    )
    result["POSITION_SOURCE"] = result.POSITION.map(
        lambda x: "NBA CommonTeamRoster (season requested)" if pd.notna(x) else None
    )
    return result
