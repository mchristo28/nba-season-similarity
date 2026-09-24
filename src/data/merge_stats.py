"""Merge endpoint measurements without overwriting base stats or multiplying rows."""

import pandas as pd


def merge_player_measurements(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if extra is None or extra.empty:
        return base
    if "PLAYER_ID" not in extra and "player_id" in extra:
        extra = extra.rename(columns={"player_id": "PLAYER_ID"})
    if "PLAYER_ID" not in extra:
        raise ValueError("Endpoint measurements lack PLAYER_ID")
    existing = {column.lower() for column in base.columns}
    columns = ["PLAYER_ID"] + [c for c in extra if c.lower() not in existing]
    return base.merge(extra[columns], on="PLAYER_ID", how="left", validate="one_to_one")
