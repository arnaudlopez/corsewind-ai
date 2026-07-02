#!/usr/bin/env python3
"""Neural multi-horizon benchmark for CorseWind SAPHIR dictionary V2 tensors."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_NAMES = ["wind_mean_ms", "gust_ms"]


def import_deps() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("Missing numpy/torch dependencies.") from exc
    return locals()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def seed_all(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric(np: Any, pred: Any, actual: Any) -> dict[str, Any]:
    mask = np.isfinite(pred) & np.isfinite(actual)
    if not mask.any():
        return {"count": 0}
    err = pred[mask] - actual[mask]
    return {
        "count": int(mask.sum()),
        "mae": round(float(np.mean(np.abs(err))), 6),
        "rmse": round(float(np.sqrt(np.mean(err * err))), 6),
        "bias": round(float(np.mean(err)), 6),
    }


def standardize_position(array: Any, train_mask: Any, np: Any, eps: float = 1e-6) -> tuple[Any, dict[str, Any]]:
    train_values = array[train_mask]
    axes = (0,)
    mean = np.nanmean(train_values, axis=axes, keepdims=True)
    std = np.nanstd(train_values, axis=axes, keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > eps), std, 1.0)
    out = np.nan_to_num((array - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return out, {"mean_shape": list(mean.shape), "std_shape": list(std.shape)}


def standardize_2d(array: Any, train_mask: Any, np: Any, eps: float = 1e-6) -> tuple[Any, dict[str, Any]]:
    train_values = array[train_mask]
    mean = np.nanmean(train_values, axis=0, keepdims=True)
    std = np.nanstd(train_values, axis=0, keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > eps), std, 1.0)
    out = np.nan_to_num((array - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return out, {"mean_shape": list(mean.shape), "std_shape": list(std.shape)}


def load_data(root: Path, args: argparse.Namespace, deps: dict[str, Any]) -> dict[str, Any]:
    np = deps["np"]
    path = root / "dictionary_tensors.npz"
    if not path.exists():
        raise SystemExit(f"Missing tensor file: {path}")
    data = np.load(path, allow_pickle=True)
    split = data["split"].astype(str)
    train_mask_full = split == "train"
    if not train_mask_full.any() or (~train_mask_full).sum() == 0:
        raise SystemExit("Need train and eval samples in tensor file.")
    train_indices_full = np.where(train_mask_full)[0]
    val_count = max(args.min_val_samples, int(round(len(train_indices_full) * args.val_fraction)))
    val_count = min(max(val_count, 1), max(len(train_indices_full) - 1, 1))
    val_indices = train_indices_full[-val_count:]
    train_indices = train_indices_full[:-val_count]
    eval_indices = np.where(~train_mask_full)[0]

    station_raw = data["station_tensor"].astype("float32")
    baseline_raw = data["baseline_tensor"].astype("float32")
    static_raw = data["static_tensor"].astype("float32")
    y_actual = data["y_actual"].astype("float32")
    y_residual = data["y_residual"].astype("float32")
    train_mask_samples = np.zeros(len(split), dtype=bool)
    train_mask_samples[train_indices] = True
    station, station_stats = standardize_position(station_raw, train_mask_samples, np)
    baseline, baseline_stats = standardize_position(baseline_raw, train_mask_samples, np)
    if static_raw.shape[1]:
        static, static_stats = standardize_2d(static_raw, train_mask_samples, np)
    else:
        static = np.zeros((len(split), 1), dtype="float32")
        static_stats = {"mean_shape": [1, 1], "std_shape": [1, 1]}
    y_mean = np.nanmean(y_residual[train_indices], axis=(0, 1), keepdims=True)
    y_std = np.nanstd(y_residual[train_indices], axis=(0, 1), keepdims=True)
    y_mean = np.where(np.isfinite(y_mean), y_mean, 0.0)
    y_std = np.where(np.isfinite(y_std) & (y_std > 1e-6), y_std, 1.0)
    y_scaled = np.nan_to_num((y_residual - y_mean) / y_std, nan=0.0).astype("float32")
    manifest = json.loads((root / "tensor_manifest.json").read_text(encoding="utf-8")) if (root / "tensor_manifest.json").exists() else {}
    return {
        "sample_id": data["sample_id"].astype(str),
        "spot_id": data["spot_id"].astype(str),
        "issue_time_utc": data["issue_time_utc"].astype(str),
        "split": split,
        "lead_minutes": data["lead_minutes"].astype(int),
        "station": station,
        "station_raw": station_raw,
        "station_mask": data["station_mask"].astype("float32"),
        "baseline": baseline,
        "baseline_raw": baseline_raw,
        "static": static,
        "y_actual": y_actual,
        "y_residual": y_residual,
        "y_scaled": y_scaled,
        "y_mean": y_mean.astype("float32"),
        "y_std": y_std.astype("float32"),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "eval_indices": eval_indices,
        "normalization": {"station": station_stats, "baseline": baseline_stats, "static": static_stats},
        "manifest": manifest,
    }


def build_dataset_class(deps: dict[str, Any]) -> Any:
    Dataset = deps["Dataset"]
    torch = deps["torch"]

    class TensorDataset(Dataset):
        def __init__(self, data: dict[str, Any], indices: Any):
            self.data = data
            self.indices = indices.astype("int64")

        def __len__(self) -> int:
            return int(len(self.indices))

        def __getitem__(self, item: int) -> dict[str, Any]:
            idx = int(self.indices[item])
            return {
                "station": torch.from_numpy(self.data["station"][idx]),
                "station_mask": torch.from_numpy(self.data["station_mask"][idx]),
                "baseline": torch.from_numpy(self.data["baseline"][idx]),
                "static": torch.from_numpy(self.data["static"][idx]),
                "target": torch.from_numpy(self.data["y_scaled"][idx]),
                "index": torch.tensor(idx, dtype=torch.long),
            }

    return TensorDataset


def build_model_class(deps: dict[str, Any]) -> Any:
    torch = deps["torch"]
    nn = deps["nn"]

    class SaphirV2Net(nn.Module):
        def __init__(self, station_shape: tuple[int, int, int], baseline_shape: tuple[int, int], static_dim: int, hidden_dim: int, dropout: float, output_shape: tuple[int, int]):
            super().__init__()
            time_steps, slots, station_features = station_shape
            leads, baseline_features = baseline_shape
            self.leads = leads
            self.targets = output_shape[1]
            self.station_gru = nn.GRU(slots * station_features, hidden_dim, batch_first=True)
            self.baseline_mlp = nn.Sequential(
                nn.Linear(leads * baseline_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.static_mlp = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, leads * self.targets),
            )

        def forward(self, batch: dict[str, Any]) -> Any:
            station = batch["station"].flatten(start_dim=2)
            _out, state = self.station_gru(station)
            station_emb = state[-1]
            baseline_emb = self.baseline_mlp(batch["baseline"].flatten(start_dim=1))
            static_emb = self.static_mlp(batch["static"])
            output = self.head(torch.cat([station_emb, baseline_emb, static_emb], dim=1))
            return output.reshape(output.shape[0], self.leads, self.targets)

    return SaphirV2Net


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def evaluate(model: Any, loader: Any, criterion: Any, device: Any, torch: Any) -> tuple[float, Any, Any]:
    model.eval()
    losses = []
    rows = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred = model(batch)
            loss = criterion(pred, batch["target"])
            losses.append(float(loss.item()))
            rows.append(batch["index"].detach().cpu())
            preds.append(pred.detach().cpu())
    if not losses:
        return 0.0, torch.empty(0, dtype=torch.long), torch.empty(0)
    return sum(losses) / len(losses), torch.cat(rows), torch.cat(preds)


def train_model(data: dict[str, Any], args: argparse.Namespace, deps: dict[str, Any]) -> tuple[Any, dict[str, Any], Any, Any]:
    torch = deps["torch"]
    nn = deps["nn"]
    DataLoader = deps["DataLoader"]
    seed_all(args.random_state, torch, deps["np"])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_cls = build_dataset_class(deps)
    train_loader = DataLoader(dataset_cls(data, data["train_indices"]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_cls(data, data["val_indices"]), batch_size=args.batch_size, shuffle=False)
    model_cls = build_model_class(deps)
    model = model_cls(
        station_shape=tuple(data["station"].shape[1:]),
        baseline_shape=tuple(data["baseline"].shape[1:]),
        static_dim=data["static"].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        output_shape=tuple(data["y_scaled"].shape[1:]),
    ).to(device)
    criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    patience = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = criterion(pred, batch["target"])
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
        val_loss, _rows, _preds = evaluate(model, val_loader, criterion, device, torch)
        train_loss = sum(losses) / max(1, len(losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"epochs_ran": len(history), "best_val_loss": best_val, "history": history, "device": str(device)}, criterion, device


def predictions_and_metrics(model: Any, data: dict[str, Any], criterion: Any, device: Any, args: argparse.Namespace, deps: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    np = deps["np"]
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    dataset_cls = build_dataset_class(deps)
    all_indices = np.arange(len(data["sample_id"]))
    loader = DataLoader(dataset_cls(data, all_indices), batch_size=args.batch_size, shuffle=False)
    _loss, rows, pred_scaled = evaluate(model, loader, criterion, device, torch)
    pred_resid = np.full_like(data["y_residual"], np.nan, dtype="float32")
    pred_resid[rows.numpy().astype(int)] = pred_scaled.numpy().astype("float32") * data["y_std"] + data["y_mean"]
    baseline_features = data["manifest"].get("baseline_features", [])
    wind_base_idx = baseline_features.index("baseline_wind_mean_ms") if "baseline_wind_mean_ms" in baseline_features else 0
    gust_base_idx = baseline_features.index("baseline_gust_ms") if "baseline_gust_ms" in baseline_features else 1
    raw_pred = np.stack([data["baseline_raw"][:, :, wind_base_idx], data["baseline_raw"][:, :, gust_base_idx]], axis=2)
    nn_pred = raw_pred + pred_resid
    persist_pred = np.full_like(raw_pred, np.nan, dtype="float32")
    station_features = data["manifest"].get("station_features", [])
    for target_idx, name in enumerate(("wind_mean_ms", "gust_ms")):
        if name in station_features:
            fidx = station_features.index(name)
            values = data["station_raw"][:, :, 0, fidx]
            last = values[:, -1]
            persist_pred[:, :, target_idx] = last[:, None]
    eval_mask = np.zeros(len(data["sample_id"]), dtype=bool)
    eval_mask[data["eval_indices"]] = True
    train_mask = np.zeros(len(data["sample_id"]), dtype=bool)
    train_mask[data["train_indices"]] = True
    metrics: dict[str, Any] = {}
    for tidx, target in enumerate(TARGET_NAMES):
        metrics[target] = {
            "train": {
                "raw_nwp": metric(np, raw_pred[train_mask, :, tidx], data["y_actual"][train_mask, :, tidx]),
                "persistence": metric(np, persist_pred[train_mask, :, tidx], data["y_actual"][train_mask, :, tidx]),
                "saphir_v2_nn": metric(np, nn_pred[train_mask, :, tidx], data["y_actual"][train_mask, :, tidx]),
            },
            "eval": {
                "raw_nwp": metric(np, raw_pred[eval_mask, :, tidx], data["y_actual"][eval_mask, :, tidx]),
                "persistence": metric(np, persist_pred[eval_mask, :, tidx], data["y_actual"][eval_mask, :, tidx]),
                "saphir_v2_nn": metric(np, nn_pred[eval_mask, :, tidx], data["y_actual"][eval_mask, :, tidx]),
            },
            "eval_by_lead": {},
        }
        for lead_idx, lead in enumerate(data["lead_minutes"]):
            metrics[target]["eval_by_lead"][str(int(lead))] = {
                "raw_nwp": metric(np, raw_pred[eval_mask, lead_idx, tidx], data["y_actual"][eval_mask, lead_idx, tidx]),
                "persistence": metric(np, persist_pred[eval_mask, lead_idx, tidx], data["y_actual"][eval_mask, lead_idx, tidx]),
                "saphir_v2_nn": metric(np, nn_pred[eval_mask, lead_idx, tidx], data["y_actual"][eval_mask, lead_idx, tidx]),
            }
    prediction_payload = {
        "sample_id": data["sample_id"],
        "spot_id": data["spot_id"],
        "issue_time_utc": data["issue_time_utc"],
        "lead_minutes": data["lead_minutes"],
        "raw_pred": raw_pred,
        "persistence_pred": persist_pred,
        "nn_pred": nn_pred,
        "actual": data["y_actual"],
    }
    return metrics, prediction_payload


def write_predictions_npz(path: Path, payload: dict[str, Any], np: Any) -> None:
    np.savez_compressed(path, **payload)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# CorseWind SAPHIR Dictionary V2 Neural Benchmark",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Dataset root: `{result['dataset_root']}`",
        f"Device: `{result['training']['device']}`",
        f"Epochs: `{result['training']['epochs_ran']}`",
        "",
        "## Eval Metrics",
        "",
        "| Target | Model | RMSE | MAE | Bias | Count |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for target, target_metrics in result["metrics"].items():
        for model_name, item in target_metrics["eval"].items():
            lines.append(f"| `{target}` | `{model_name}` | {item.get('rmse')} | {item.get('mae')} | {item.get('bias')} | {item.get('count')} |")
    lines.extend(["", "## Eval RMSE By Lead", ""])
    for target, target_metrics in result["metrics"].items():
        lines.append(f"### {target}")
        lines.append("")
        lines.append("| Lead | Raw NWP | Persistence | SAPHIR V2 NN |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for lead, items in target_metrics["eval_by_lead"].items():
            lines.append(f"| {lead} | {items['raw_nwp'].get('rmse')} | {items['persistence'].get('rmse')} | {items['saphir_v2_nn'].get('rmse')} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-val-samples", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = import_deps()
    output_root = (args.output_root or args.dataset_root / "benchmark_v2_neural").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data = load_data(args.dataset_root, args, deps)
    model, training, criterion, device = train_model(data, args, deps)
    metrics, predictions = predictions_and_metrics(model, data, criterion, device, args, deps)
    write_predictions_npz(output_root / "predictions_neural.npz", predictions, deps["np"])
    result = {
        "format": "corsewind.saphir_dictionary_v2_neural_benchmark.v1",
        "generated_at_utc": utc_now(),
        "dataset_root": str(args.dataset_root.resolve()),
        "output_root": str(output_root),
        "sample_count": int(len(data["sample_id"])),
        "train_samples": int(len(data["train_indices"])),
        "val_samples": int(len(data["val_indices"])),
        "eval_samples": int(len(data["eval_indices"])),
        "training": training,
        "metrics": metrics,
        "normalization": data["normalization"],
        "args": vars(args),
    }
    (output_root / "benchmark_results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(output_root / "benchmark_results.md", result)
    print(json.dumps({"output_root": str(output_root), "metrics": {k: v["eval"] for k, v in metrics.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
