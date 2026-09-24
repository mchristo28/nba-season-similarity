"""Offline score audit: python scripts/audit_scores.py."""

from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.similarity.scoring import similarity_score
from src.similarity.weighted_matcher import WeightedMatcher


def main():
    df = pd.read_parquet(
        Path(__file__).resolve().parents[1] / "data/features/player_features.parquet"
    )
    matcher = WeightedMatcher().fit(df)
    records = []
    for name in ["Shai Gilgeous-Alexander", "Stephen Curry", "LeBron James", "Nikola Jokić"]:
        row = df[df.PLAYER_NAME.eq(name)].sort_values("CAREER_YEAR").iloc[-1]
        started = perf_counter()
        matches = matcher.find_similar_season(
            row.PLAYER_ID,
            int(row.CAREER_YEAR),
            n=5,
            min_games=20,
            min_minutes=10,
            exclude_same=True,
        )
        elapsed = perf_counter() - started
        for pid, other, year, distance, _ in matches:
            records.append(
                {
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


if __name__ == "__main__":
    main()
