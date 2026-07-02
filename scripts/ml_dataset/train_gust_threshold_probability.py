#!/usr/bin/env python3
"""Train probabilistic strong-gust heads from residual training parquet rows."""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TARGETS = ("labels__target_gust_gt_20kt", "labels__target_gust_gt_25kt")


def import_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc:
        raise SystemExit("Missing ML dependencies. Run inside the CorseWind ML venv.") from exc
    return locals()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    return sorted(dict.fromkeys(paths))


def read_feature_columns(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("numeric") or []), list(payload.get("categorical") or [])


def schema_columns(paths: list[Path], pq: Any) -> set[str]:
    columns: set[str] = set()
    for path in paths:
        columns.update(pq.ParquetFile(path).schema_arrow.names)
    return columns


def read_frame_fast(paths: list[Path], columns: list[str], pq: Any, pd: Any) -> Any:
    frames = []
    for path in paths:
        pf = pq.ParquetFile(path)
        available = [column for column in columns if column in pf.schema_arrow.names]
        frames.append(pf.read(columns=available).to_pandas().reindex(columns=columns))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def add_time_features(frame: Any, np: Any, pd: Any) -> Any:
    issue_time = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    dayofyear = issue_time.dt.dayofyear.fillna(1).astype(float)
    frame["issue_hour_utc"] = issue_time.dt.hour.astype("float64")
    frame["issue_month"] = issue_time.dt.month.astype("float64")
    angle = 2.0 * np.pi * dayofyear / 366.0
    frame["issue_dayofyear_sin"] = np.sin(angle)
    frame["issue_dayofyear_cos"] = np.cos(angle)
    frame["issue_month_number"] = issue_time.dt.month.astype("float64")
    return frame


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str], deps: dict[str, Any]) -> Any:
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                deps["Pipeline"]([
                    ("imputer", deps["SimpleImputer"](strategy="median")),
                ]),
                numeric_columns,
            )
        )
    if categorical_columns:
        try:
            encoder = deps["OneHotEncoder"](handle_unknown="ignore", min_frequency=20, sparse_output=False)
        except TypeError:
            encoder = deps["OneHotEncoder"](handle_unknown="ignore", min_frequency=20, sparse=False)
        transformers.append(
            (
                "categorical",
                deps["Pipeline"]([
                    ("imputer", deps["SimpleImputer"](strategy="most_frequent")),
                    ("onehot", encoder),
                ]),
                categorical_columns,
            )
        )
    return deps["ColumnTransformer"](transformers=transformers, remainder="drop")


def make_classifier(args: argparse.Namespace, deps: dict[str, Any]) -> Any:
    if args.model_family == "extra_trees":
        return deps["ExtraTreesClassifier"](
            n_estimators=args.n_estimators,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
        )
    return deps["HistGradientBoostingClassifier"](
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_seed,
    )


def class_weights(y: Any) -> Any:
    counts = y.value_counts().to_dict()
    total = float(len(y))
    return y.map({klass: total / (2.0 * max(1, count)) for klass, count in counts.items()}).astype(float)


def threshold_metrics(y_true: Any, probability: Any, threshold: float) -> dict[str, Any]:
    pred = probability >= threshold
    actual = y_true.astype(bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "csi": None if tp + fp + fn == 0 else tp / (tp + fp + fn),
    }


def best_csi_threshold(y_true: Any, probability: Any) -> dict[str, Any]:
    best = None
    for threshold in [i / 100.0 for i in range(5, 96, 5)]:
        item = threshold_metrics(y_true, probability, threshold)
        score = -1 if item["csi"] is None else item["csi"]
        if best is None or score > (best["csi"] if best["csi"] is not None else -1):
            best = item
    return best or threshold_metrics(y_true, probability, 0.5)


def classification_metrics(y_true: Any, probability: Any, deps: dict[str, Any]) -> dict[str, Any]:
    np = deps["np"]
    y = y_true.astype(int)
    probability = np.asarray(probability, dtype=float)
    out = {
        "row_count": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else None,
        "brier": float(deps["brier_score_loss"](y, probability)),
        "log_loss": float(deps["log_loss"](y, probability, labels=[0, 1])),
        "threshold_0p20": threshold_metrics(y, probability, 0.20),
        "threshold_0p50": threshold_metrics(y, probability, 0.50),
        "best_csi_threshold": best_csi_threshold(y, probability),
    }
    if len(set(y.tolist())) > 1:
        out["roc_auc"] = float(deps["roc_auc_score"](y, probability))
        out["average_precision"] = float(deps["average_precision_score"](y, probability))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = import_dependencies()
    pd = deps["pd"]
    np = deps["np"]
    pq = deps["pq"]
    paths = expand_paths(args.parquet)
    if not paths:
        raise SystemExit("No parquet inputs matched.")
    numeric_columns, categorical_columns = read_feature_columns(args.feature_columns_json)
    all_columns = schema_columns(paths, pq)
    targets = [target for target in (args.target or DEFAULT_TARGETS) if target in all_columns]
    if not targets:
        raise SystemExit("No requested target columns found in parquet inputs.")
    derived_time = ["issue_hour_utc", "issue_month", "issue_dayofyear_sin", "issue_dayofyear_cos", "issue_month_number"]
    feature_columns = [
        column for column in [*numeric_columns, *categorical_columns]
        if column in all_columns or column in derived_time
    ]
    numeric_columns = [column for column in numeric_columns if column in feature_columns]
    categorical_columns = [column for column in categorical_columns if column in feature_columns]
    required = sorted(set(feature_columns) | set(targets) | {"issue_time_utc", "spot_id", "lead_time_minutes"})
    frame = read_frame_fast(paths, required, pq, pd)
    frame = add_time_features(frame, np, pd)
    issue_time = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    train_mask = issue_time < pd.Timestamp(args.split_time_utc)
    test_mask = issue_time >= pd.Timestamp(args.split_time_utc)
    if args.max_train_rows and int(train_mask.sum()) > args.max_train_rows:
        train_indices = frame[train_mask].sample(n=args.max_train_rows, random_state=args.random_seed).index
        train_mask = frame.index.isin(train_indices)
    if args.max_test_rows and int(test_mask.sum()) > args.max_test_rows:
        test_indices = frame[test_mask].sample(n=args.max_test_rows, random_state=args.random_seed).index
        test_mask = frame.index.isin(test_indices)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "feature_columns.json").write_text(
        json.dumps({"numeric": numeric_columns, "categorical": categorical_columns}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results = {
        "format": "corsewind.gust_threshold_probability_training.v1",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "model_family": args.model_family,
        "source_parquet_count": len(paths),
        "source_parquets": [str(path) for path in paths],
        "split_time_utc": args.split_time_utc,
        "row_count": int(len(frame)),
        "train_row_count": int(train_mask.sum()),
        "test_row_count": int(test_mask.sum()),
        "feature_column_count": len(feature_columns),
        "numeric_column_count": len(numeric_columns),
        "categorical_column_count": len(categorical_columns),
        "models": {},
        "settings": vars(args) | {"output_root": str(args.output_root), "feature_columns_json": str(args.feature_columns_json)},
    }
    preprocessor = make_preprocessor(numeric_columns, categorical_columns, deps)
    for target in targets:
        target_train_mask = train_mask & frame[target].isin([0, 1])
        target_test_mask = test_mask & frame[target].isin([0, 1])
        y_train = frame.loc[target_train_mask, target].astype(int)
        y_test = frame.loc[target_test_mask, target].astype(int)
        if len(y_train) < args.min_train_rows or len(y_test) < args.min_test_rows or y_train.nunique() < 2:
            results["models"][target] = {
                "skipped": True,
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "train_positive_count": int(y_train.sum()) if len(y_train) else 0,
                "test_positive_count": int(y_test.sum()) if len(y_test) else 0,
            }
            continue
        classifier = make_classifier(args, deps)
        model = deps["Pipeline"]([
            ("preprocess", preprocessor),
            ("model", classifier),
        ])
        fit_kwargs = {}
        if args.model_family == "hist_gradient_boosting":
            fit_kwargs["model__sample_weight"] = class_weights(y_train)
        model.fit(frame.loc[target_train_mask, feature_columns], y_train, **fit_kwargs)
        probability = model.predict_proba(frame.loc[target_test_mask, feature_columns])[:, 1]
        model_path = args.output_root / f"{target}.joblib"
        deps["joblib"].dump(model, model_path)
        results["models"][target] = {
            "skipped": False,
            "model_path": str(model_path),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "train_positive_count": int(y_train.sum()),
            "test_positive_count": int(y_test.sum()),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "test": classification_metrics(y_test, probability, deps),
        }
    (args.output_root / "training_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    lines = ["# Gust Threshold Probability Training", ""]
    lines.append(f"Run id: `{args.run_id}`")
    lines.append("")
    lines.append("| Target | Brier | AUC | AP | Pos rate | Best CSI | Threshold | Recall | Precision |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for target, item in sorted(results["models"].items()):
        test = item.get("test", {})
        best = test.get("best_csi_threshold", {})
        lines.append(
            f"| `{target}` | {test.get('brier')} | {test.get('roc_auc')} | {test.get('average_precision')} | "
            f"{test.get('positive_rate')} | {best.get('csi')} | {best.get('threshold')} | "
            f"{best.get('recall')} | {best.get('precision')} |"
        )
    (args.output_root / "training_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", action="append", required=True)
    parser.add_argument("--feature-columns-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--split-time-utc", default="2026-01-01T00:00:00Z")
    parser.add_argument("--model-family", choices=("hist_gradient_boosting", "extra_trees"), default="extra_trees")
    parser.add_argument("--n-estimators", type=int, default=450)
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-train-rows", type=int, default=250000)
    parser.add_argument("--max-test-rows", type=int, default=120000)
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--min-test-rows", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "run_id": result["run_id"],
        "train_row_count": result["train_row_count"],
        "test_row_count": result["test_row_count"],
        "models": {
            target: item.get("test")
            for target, item in result["models"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
