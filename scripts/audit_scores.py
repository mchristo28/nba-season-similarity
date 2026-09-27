"""Offline comparison audit: python scripts/audit_scores.py --profile all."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.similarity.profiles import PROFILES  # noqa: E402
from src.similarity.scoring import similarity_score  # noqa: E402
from src.similarity.weighted_matcher import WeightedMatcher  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["all", *PROFILES], default="all")
    parser.add_argument("--output", type=Path, help="Optional JSON audit output")
    args = parser.parse_args()
    df = pd.read_parquet(
        Path(__file__).resolve().parents[1] / "data/features/player_features.parquet"
    )
    records = []
    for profile in PROFILES if args.profile == "all" else [args.profile]:
        matcher = WeightedMatcher(profile=profile).fit(df)
        for name in [
            "Shai Gilgeous-Alexander",
            "Stephen Curry",
            "LeBron James",
            "Nikola Jokić",
            "Rudy Gobert",
            "Draymond Green",
        ]:
            row = df[df.PLAYER_NAME.eq(name)].sort_values("CAREER_YEAR").iloc[-1]
            started = perf_counter()
            matches = matcher.find_similar_season(
                row.PLAYER_ID,
                int(row.CAREER_YEAR),
                n=3,
                min_games=20,
                min_minutes=15,
                exclude_same=True,
                min_coverage=0.8,
            )
            elapsed = perf_counter() - started
            if not matches:
                raise ValueError(f"No matches for audit subject {name} in {profile}")
            for pid, other, year, distance, _ in matches:
                records.append(
                    {
                        "profile": profile,
                        "query": name,
                        "season": row.SEASON,
                        "match": other,
                        "match_season": matcher.get_season_info(pid, year)["season"],
                        "score": round(similarity_score(distance), 1),
                        "coverage": round(
                            matcher.season_coverage(row.PLAYER_ID, int(row.CAREER_YEAR), pid, year)
                            * 100,
                            1,
                        ),
                        "search_ms": round(elapsed * 1000, 1),
                    }
                )
    print(pd.DataFrame(records).to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
