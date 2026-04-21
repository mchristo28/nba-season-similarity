"""Fetch and cache player awards for all players in the dataset."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.data.nba_api_client import NBAApiClient


def fetch_all_awards(features_path: str = "data/features/player_features.parquet",
                     output_path: str = "data/features/player_awards.parquet"):
    """Fetch awards for all players and save to parquet."""

    print("Loading player features...")
    df = pd.read_parquet(features_path)

    # Get unique player IDs
    player_ids = df["PLAYER_ID"].unique()
    print(f"Found {len(player_ids)} unique players")

    client = NBAApiClient()

    all_awards = []

    for i, player_id in enumerate(player_ids):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(player_ids)}")

        try:
            awards_df = client.get_player_awards(player_id)
            if not awards_df.empty:
                awards_df["PLAYER_ID"] = player_id
                all_awards.append(awards_df)
        except Exception as e:
            print(f"  Error fetching awards for {player_id}: {e}")

    if all_awards:
        combined = pd.concat(all_awards, ignore_index=True)
        print(f"\nFetched {len(combined)} total award records")

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")

        return combined
    else:
        print("No awards found")
        return pd.DataFrame()


def build_season_awards_lookup(awards_path: str = "data/features/player_awards.parquet",
                                output_path: str = "data/features/season_awards.parquet"):
    """Build a lookup table of player_id + season -> award emojis."""

    print("Loading awards...")
    awards_df = pd.read_parquet(awards_path)

    # Map award descriptions to emojis
    def get_emoji(row):
        desc = row.get("DESCRIPTION", "")
        team_num = row.get("ALL_NBA_TEAM_NUMBER", "")

        if "All-Star" in desc and "MVP" not in desc:
            return ("all_star", "⭐")
        elif "All-NBA" in desc:
            if team_num == 1:
                return ("all_nba_1st", "🥇")
            elif team_num == 2:
                return ("all_nba_2nd", "🥈")
            elif team_num == 3:
                return ("all_nba_3rd", "🥉")
            else:
                return ("all_nba", "🏅")
        elif "Champion" in desc:
            return ("champion", "🏆")
        elif "MVP" in desc and "All-Star" not in desc:
            return ("mvp", "👑")
        elif "Rookie of the Year" in desc:
            return ("roy", "🌟")
        elif "Defensive Player" in desc:
            return ("dpoy", "🛡️")
        elif "Most Improved" in desc:
            return ("mip", "📈")
        elif "Sixth Man" in desc:
            return ("6moy", "6️⃣")
        return (None, None)

    # Build lookup
    season_awards = {}

    for _, row in awards_df.iterrows():
        player_id = row["PLAYER_ID"]
        season = row.get("SEASON", "")
        if not season:
            continue

        award_type, emoji = get_emoji(row)
        if not emoji:
            continue

        key = (player_id, season)
        if key not in season_awards:
            season_awards[key] = {"PLAYER_ID": player_id, "SEASON": season, "awards": []}

        if emoji not in season_awards[key]["awards"]:
            season_awards[key]["awards"].append(emoji)

    # Convert to dataframe
    rows = []
    for (player_id, season), data in season_awards.items():
        rows.append({
            "PLAYER_ID": player_id,
            "SEASON": season,
            "AWARDS": "".join(data["awards"])
        })

    result = pd.DataFrame(rows)
    print(f"Built {len(result)} season-award records")

    # Save
    result.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cache player awards")
    parser.add_argument("--fetch", action="store_true", help="Fetch awards from API")
    parser.add_argument("--build-lookup", action="store_true", help="Build season lookup from cached awards")
    args = parser.parse_args()

    if args.fetch:
        fetch_all_awards()

    if args.build_lookup:
        build_season_awards_lookup()

    if not args.fetch and not args.build_lookup:
        print("Usage: python cache_awards.py --fetch    # Fetch from API")
        print("       python cache_awards.py --build-lookup  # Build lookup table")
