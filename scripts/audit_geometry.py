"""Reproducible distribution and aggregate-distance review; does not modify app data."""

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.similarity.profiles import PROFILES  # noqa: E402
from src.similarity.scoring import similarity_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = [
    ("Lauri Markkanen", "2024-25"),
    ("Lauri Markkanen", "2022-23"),
    ("Stephen Curry", "2024-25"),
    ("Shai Gilgeous-Alexander", "2024-25"),
    ("LeBron James", "2024-25"),
    ("Nikola Jokić", "2024-25"),
    ("Rudy Gobert", "2024-25"),
    ("Draymond Green", "2024-25"),
    ("Kevin Durant", "2024-25"),
]


def geometry(reference, columns, groups):
    scaler = StandardScaler().fit(reference[columns])
    z = scaler.transform(reference[columns].dropna())
    covariance = LedoitWolf(store_precision=False).fit(z)
    return scaler, weighted_precision(covariance.covariance_, groups), covariance


def weighted_precision(covariance, groups):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    assert np.all(eigenvalues > 0)
    inverse = np.einsum("ik,k,jk->ij", eigenvectors, 1 / eigenvalues, eigenvectors)
    weights = np.concatenate(
        [np.repeat(s["default_weight"] / len(s["features"]), len(s["features"])) for s in groups]
    )
    precision = inverse * np.sqrt(np.outer(weights, weights))
    # Equal expected squared distance for independent reference draws (approximately 2).
    precision /= np.einsum("ij,ji->", precision, covariance)
    assert np.isfinite(precision).all()
    return precision


def distances(delta, groups, precision):
    squared, weights, offset = [], [], 0
    for spec in groups:
        width = len(spec["features"])
        squared.append(np.mean(delta[:, offset : offset + width] ** 2, axis=1))
        weights.append(spec["default_weight"])
        offset += width
    squared = np.array(squared).T
    return {
        "mean_group_rms": np.average(np.sqrt(squared), weights=weights, axis=1),
        "joint_rms": np.sqrt(np.average(squared, weights=weights, axis=1)),
        "regularized_mahalanobis": np.sqrt(
            np.maximum(0, np.einsum("ni,ij,nj->n", delta, precision, delta))
        ),
    }


def main():
    frame = pd.read_parquet(ROOT / "data/features/player_features.parquet")
    report = {
        "dataset_sha256": hashlib.sha256(
            (ROOT / "data/features/player_features.parquet").read_bytes()
        ).hexdigest(),
        "seed": 42,
        "profiles": {},
    }
    for profile, spec in PROFILES.items():
        groups = list(spec["groups"].values())
        columns = [c for g in groups for c in g["features"]]
        reference = frame[
            (frame.GP >= 20)
            & (frame.MIN >= 15)
            & (frame.SEASON.str[:4].astype(int) >= spec["first_year"])
        ]
        scaler, precision, covariance = geometry(reference, columns, groups)
        earlier = reference[reference.SEASON < "2023-24"]
        earlier_scaler, earlier_precision, _ = geometry(earlier, columns, groups)
        summary = {
            "reference_rows": len(reference),
            "covariance_complete_rows": len(reference[columns].dropna()),
            "covariance_shrinkage": float(covariance.shrinkage_),
            "covariance_condition": float(np.linalg.cond(covariance.covariance_)),
            "distributions": {},
            "correlated_pairs": [],
            "color_pair_fractions": {},
            "cases": [],
        }
        for c, sd in zip(columns, scaler.scale_):
            x = reference[c].dropna()
            summary["distributions"][c] = {
                "n": len(x),
                "missing": int(reference[c].isna().sum()),
                "sd": float(sd),
                "skew": float(x.skew()),
                "zero_fraction": float(x.eq(0).mean()),
                "quantiles": {
                    str(q): float(x.quantile(q)) for q in [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
                },
                "iqr_normal_scale_ratio": float((x.quantile(0.75) - x.quantile(0.25)) / 1.349 / sd),
            }
            rng = np.random.default_rng(42)
            gaps = np.abs(rng.choice(x, 10000) - rng.choice(x, 10000)) / sd
            summary["color_pair_fractions"][c] = {
                "close": float((gaps < 0.5).mean()),
                "noticeable": float(((gaps >= 0.5) & (gaps < 1)).mean()),
                "large": float((gaps >= 1).mean()),
            }
        corr = reference[columns].corr()
        for i, c in enumerate(columns):
            for d in columns[i + 1 :]:
                if abs(corr.loc[c, d]) >= 0.8:
                    summary["correlated_pairs"].append([c, d, float(corr.loc[c, d])])
        # Same candidate evidence for all methods; complete-case audit is explicitly narrower
        # than the app's query-observed policy. No missing values are imputed.
        pool = frame[
            (frame.GP >= 20)
            & (frame.MIN >= 10)
            & (frame.SEASON.str[:4].astype(int) >= spec["first_year"])
        ].dropna(subset=columns)
        for name, season in SUBJECTS:
            query = pool[(pool.PLAYER_NAME == name) & (pool.SEASON == season)]
            if query.empty:
                continue
            candidates = pool[pool.PLAYER_ID != query.PLAYER_ID.iloc[0]]
            delta = scaler.transform(candidates[columns]) - scaler.transform(query[columns])
            ds = distances(delta, groups, precision)
            # Weight sensitivity: independently increase each category's weight by 20%.
            perturbed = []
            for group_index in range(len(groups)):
                changed = deepcopy(groups)
                changed[group_index]["default_weight"] *= 1.2
                perturbed.append(
                    distances(delta, changed, weighted_precision(covariance.covariance_, changed))
                )
            earlier_ds = distances(
                earlier_scaler.transform(candidates[columns])
                - earlier_scaler.transform(query[columns]),
                groups,
                earlier_precision,
            )
            case = {"name": name, "season": season, "methods": {}}
            for method, values in ds.items():
                order = np.argsort(values, kind="stable")
                old_order = np.argsort(earlier_ds[method], kind="stable")
                top = candidates.iloc[order[:5]]
                case["methods"][method] = {
                    "top5": [
                        {
                            "name": r.PLAYER_NAME,
                            "season": r.SEASON,
                            "score": round(similarity_score(values[i]), 2),
                        }
                        for i, (_, r) in zip(order[:5], top.iterrows())
                    ],
                    "top5_overlap_earlier_reference": len(set(order[:5]) & set(old_order[:5])),
                    "mean_top5_overlap_weight_perturbations": float(
                        np.mean(
                            [
                                len(set(order[:5]) & set(np.argsort(p[method], kind="stable")[:5]))
                                for p in perturbed
                            ]
                        )
                    ),
                }
                gg = np.flatnonzero(
                    (
                        (candidates.PLAYER_NAME == "GG Jackson") & (candidates.SEASON == "2023-24")
                    ).to_numpy()
                )
                if name == "Lauri Markkanen" and season == "2024-25" and len(gg):
                    case["methods"][method]["gg_pair"] = {
                        "rank": int(np.flatnonzero(order == gg[0])[0]) + 1,
                        "score": round(similarity_score(values[gg[0]]), 2),
                    }
            summary["cases"].append(case)
        report["profiles"][profile] = summary
    output = ROOT / "docs/comparison-geometry-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    for name, result in report["profiles"].items():
        print(
            name,
            "reference",
            result["reference_rows"],
            "condition",
            round(result["covariance_condition"], 1),
        )
        for case in result["cases"]:
            print(
                case["name"],
                case["season"],
                {
                    k: (v["top5"][0], v["top5_overlap_earlier_reference"], v.get("gg_pair"))
                    for k, v in case["methods"].items()
                },
            )


if __name__ == "__main__":
    main()
