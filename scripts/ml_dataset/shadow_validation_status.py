#!/usr/bin/env python3
"""Summarize CorseWind shadow validation watcher and suite status."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        return {"_error": str(exc), "_path": str(path)}


def pid_status(pid_file: Path) -> dict[str, Any]:
    raw = read_text(pid_file)
    out: dict[str, Any] = {"pid_file": str(pid_file), "pid": raw, "running": False}
    if not raw or not raw.isdigit():
        return out
    completed = subprocess.run(
        ["ps", "-o", "pid=,ppid=,sid=,stat=,etime=,cmd=", "-p", raw],
        text=True,
        capture_output=True,
        check=False,
    )
    out["running"] = completed.returncode == 0 and bool(completed.stdout.strip())
    out["ps"] = completed.stdout.strip()
    return out


def latest_coverage(log_file: Path) -> dict[str, Any] | None:
    text = read_text(log_file)
    if not text:
        return None
    latest = None
    for line in text.splitlines():
        marker = " coverage "
        if marker not in line:
            continue
        prefix, payload = line.split(marker, 1)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        parsed["log_time_utc"] = prefix.strip().split(" ", 1)[0]
        latest = parsed
    return latest


def suite_status(output_root: Path) -> dict[str, Any]:
    summary_path = output_root / "suite_summary.json"
    summary = read_json(summary_path)
    out: dict[str, Any] = {
        "output_root": str(output_root),
        "suite_summary": str(summary_path),
        "exists": summary is not None,
        "complete": False,
        "case_count": 0,
        "scored_cases": 0,
        "shadow_cases": 0,
    }
    if not summary:
        return out
    if "_error" in summary:
        out["error"] = summary["_error"]
        return out
    cases = summary.get("cases") or []
    out["generated_at_utc"] = summary.get("generated_at_utc")
    out["failures"] = summary.get("failures") or []
    out["case_count"] = len(cases)
    out["scored_cases"] = sum(1 for case in cases if case.get("score_json") or case.get("joined_rows"))
    out["shadow_cases"] = sum(1 for case in cases if case.get("shadow_router_v1"))
    out["complete"] = bool(cases) and out["scored_cases"] == len(cases) and out["shadow_cases"] == len(cases)
    out["cases"] = [
        {
            "run_id": case.get("run_id"),
            "score": bool(case.get("score_json") or case.get("joined_rows")),
            "shadow": bool(case.get("shadow_router_v1")),
            "rows": case.get("joined_rows"),
            "target_end_scored_utc": case.get("target_end_scored_utc"),
        }
        for case in cases
    ]
    return out


def artifact_status(output_root: Path) -> dict[str, Any]:
    items = {
        "aggregate_json": output_root / "shadow_aggregate_v1.json",
        "aggregate_markdown": output_root / "shadow_aggregate_v1.md",
        "wind_gate_json": output_root / "wind_router_promotion_gate_v1.json",
        "gust_gate_json": output_root / "gust_guarded_stacker_promotion_gate_v1.json",
        "promotion_review_json": output_root / "promotion_candidate_review_v1.json",
    }
    out: dict[str, Any] = {}
    for name, path in items.items():
        payload = read_json(path) if path.suffix == ".json" else None
        out[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        if payload:
            out[name]["format"] = payload.get("format")
            out[name]["decision"] = payload.get("decision")
            out[name]["passed"] = payload.get("passed")
            out[name]["case_count"] = payload.get("case_count")
            out[name]["joined_rows"] = payload.get("joined_rows")
            if payload.get("best"):
                out[name]["best"] = {
                    target: None
                    if item is None
                    else {
                        "candidate": item.get("candidate"),
                        "decision": item.get("decision"),
                        "failed_check_count": item.get("failed_check_count"),
                    }
                    for target, item in (payload.get("best") or {}).items()
                }
    return out


def gate_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }
    if not payload:
        return out
    out["format"] = payload.get("format")
    out["decision"] = payload.get("decision")
    out["passed"] = payload.get("passed")
    out["candidate"] = payload.get("candidate")
    out["metric_target"] = payload.get("metric_target")
    out["case_count"] = payload.get("case_count")
    out["shadow_case_count"] = payload.get("shadow_case_count")
    out["joined_rows"] = payload.get("joined_rows")
    out["reasons"] = payload.get("reasons") or []
    return out


def rollup_status(rollup_root: Path | None) -> dict[str, Any] | None:
    if rollup_root is None:
        return None

    discovered_path = rollup_root / "discovered_suite_summaries.txt"
    discovered = [line for line in (read_text(discovered_path) or "").splitlines() if line.strip()]
    aggregate_path = rollup_root / "shadow_multi_day_aggregate.json"
    aggregate = read_json(aggregate_path)

    out: dict[str, Any] = {
        "rollup_root": str(rollup_root),
        "exists": rollup_root.exists(),
        "discovered_suite_summaries": str(discovered_path),
        "complete_suite_count": len(discovered),
        "complete_suites": discovered,
        "aggregate_json": {
            "path": str(aggregate_path),
            "exists": aggregate_path.exists(),
            "size_bytes": aggregate_path.stat().st_size if aggregate_path.exists() else 0,
        },
        "wind_router_gate": gate_summary(rollup_root / "wind_router_promotion_gate.json"),
        "gust_guarded_stacker_gate": gate_summary(rollup_root / "gust_guarded_stacker_promotion_gate.json"),
        "promotion_review": {
            "path": str(rollup_root / "promotion_candidate_review.json"),
            "exists": (rollup_root / "promotion_candidate_review.json").exists(),
            "size_bytes": (rollup_root / "promotion_candidate_review.json").stat().st_size
            if (rollup_root / "promotion_candidate_review.json").exists()
            else 0,
        },
        "promotion_decision": {
            "path": str(rollup_root / "promotion_decision.json"),
            "exists": (rollup_root / "promotion_decision.json").exists(),
            "size_bytes": (rollup_root / "promotion_decision.json").stat().st_size
            if (rollup_root / "promotion_decision.json").exists()
            else 0,
        },
        "threshold_guard_audit": {
            "path": str(rollup_root / "threshold_guard_impact_audit.json"),
            "exists": (rollup_root / "threshold_guard_impact_audit.json").exists(),
            "size_bytes": (rollup_root / "threshold_guard_impact_audit.json").stat().st_size
            if (rollup_root / "threshold_guard_impact_audit.json").exists()
            else 0,
        },
        "rollup_index": {
            "path": str(rollup_root / "rollup_index.md"),
            "exists": (rollup_root / "rollup_index.md").exists(),
        },
    }
    if aggregate:
        out["aggregate_json"]["format"] = aggregate.get("format")
        out["aggregate_json"]["generated_at_utc"] = aggregate.get("generated_at_utc")
        out["aggregate_json"]["case_count"] = aggregate.get("case_count")
        out["aggregate_json"]["shadow_case_count"] = aggregate.get("shadow_case_count")
        out["aggregate_json"]["joined_rows"] = aggregate.get("joined_rows")
    review = read_json(rollup_root / "promotion_candidate_review.json")
    if review:
        out["promotion_review"]["format"] = review.get("format")
        out["promotion_review"]["evidence_progress"] = review.get("evidence_progress") or {}
        out["promotion_review"]["best"] = {
            target: None
            if item is None
            else {
                "candidate": item.get("candidate"),
                "decision": item.get("decision"),
                "failed_check_count": item.get("failed_check_count"),
                "global_failed_check_count": item.get("global_failed_check_count"),
                "performance_failed_check_count": item.get("performance_failed_check_count"),
                "performance_gap_summary": item.get("performance_gap_summary") or {},
            }
            for target, item in (review.get("best") or {}).items()
        }
        out["promotion_review"]["best_by_rmse"] = {
            target: None
            if item is None
            else {
                "candidate": item.get("candidate"),
                "decision": item.get("decision"),
                "rmse_ms": (item.get("overall_ms") or {}).get("rmse_ms"),
                "global_failed_check_count": item.get("global_failed_check_count"),
                "performance_failed_check_count": item.get("performance_failed_check_count"),
                "performance_gap_summary": item.get("performance_gap_summary") or {},
            }
            for target, item in (review.get("best_by_rmse") or {}).items()
        }
    decision = read_json(rollup_root / "promotion_decision.json")
    if decision:
        out["promotion_decision"]["format"] = decision.get("format")
        out["promotion_decision"]["decision"] = decision.get("decision")
        out["promotion_decision"]["evidence_progress"] = decision.get("evidence_progress") or {}
        out["promotion_decision"]["targets"] = {
            target: {
                "decision": item.get("decision"),
                "blocker_type": item.get("blocker_type"),
                "candidate": ((item.get("candidate") or {}).get("candidate")),
                "best_candidate": ((item.get("best") or {}).get("candidate")),
                "best_rmse_candidate": ((item.get("best_by_rmse") or {}).get("candidate")),
            }
            for target, item in (decision.get("targets") or {}).items()
        }
    threshold_audit = read_json(rollup_root / "threshold_guard_impact_audit.json")
    if threshold_audit:
        out["threshold_guard_audit"]["format"] = threshold_audit.get("format")
        out["threshold_guard_audit"]["rows"] = threshold_audit.get("rows")
        out["threshold_guard_audit"]["wind_rmse_gain_vs_baseline"] = (
            ((threshold_audit.get("overall") or {}).get("wind") or {}).get("rmse_gain_vs_baseline") or {}
        )
        out["threshold_guard_audit"]["gust_rmse_gain_vs_baseline"] = (
            ((threshold_audit.get("overall") or {}).get("gust") or {}).get("rmse_gain_vs_baseline") or {}
        )
        out["threshold_guard_audit"]["risk_flags"] = threshold_audit.get("risk_flags") or {}
    return out


def latest_nonempty_line(path: Path) -> str | None:
    text = read_text(path)
    if not text:
        return None
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return None


def campaign_status(campaign_id: str | None, log_root: Path) -> dict[str, Any] | None:
    if not campaign_id:
        return None
    pid_file = log_root / f"{campaign_id}.pid"
    log_file = log_root / f"{campaign_id}.log"
    log_tail = (read_text(log_file) or "").splitlines()[-8:]
    latest_line = latest_nonempty_line(log_file)
    return {
        "campaign_id": campaign_id,
        "pid_file": str(pid_file),
        "log_file": str(log_file),
        "watcher": pid_status(pid_file),
        "latest_line": latest_line,
        "log_tail": log_tail,
    }


def disk_status(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gb": round(usage.total / 1_000_000_000, 3),
        "used_gb": round(usage.used / 1_000_000_000, 3),
        "free_gb": round(usage.free / 1_000_000_000, 3),
        "used_percent": round(usage.used / usage.total * 100.0, 2),
    }


def health_status(
    *,
    generated_at_utc: str,
    main_watcher: dict[str, Any],
    postprocess_watcher: dict[str, Any],
    coverage: dict[str, Any] | None,
    suite: dict[str, Any],
    artifacts: dict[str, Any],
    max_coverage_age_minutes: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    now = parse_utc(generated_at_utc) or datetime.now(timezone.utc)
    coverage_time = parse_utc((coverage or {}).get("log_time_utc"))
    coverage_age_minutes = None
    if coverage_time is not None:
        coverage_age_minutes = round((now - coverage_time).total_seconds() / 60.0, 3)
        if coverage_age_minutes > max_coverage_age_minutes:
            reasons.append(f"coverage stale: {coverage_age_minutes:.1f} min > {max_coverage_age_minutes:.1f} min")
    else:
        reasons.append("no coverage log entry")

    main_running = bool(main_watcher.get("running"))
    post_running = bool(postprocess_watcher.get("running"))
    suite_complete = bool(suite.get("complete"))
    postprocessed = bool(
        artifacts.get("aggregate_json", {}).get("exists")
        and artifacts.get("wind_gate_json", {}).get("exists")
        and artifacts.get("gust_gate_json", {}).get("exists")
    )

    if postprocessed:
        status = "postprocessed"
        if main_running:
            reasons.append("main watcher still running after postprocess artifacts exist")
        if post_running:
            reasons.append("postprocess watcher still running after postprocess artifacts exist")
    elif suite_complete:
        status = "awaiting_postprocess"
        if not post_running:
            reasons.append("suite complete but postprocess watcher is not running")
    elif main_running and post_running:
        status = "waiting_for_observations"
    elif main_running and not post_running:
        status = "attention"
        reasons.append("main watcher running but postprocess watcher is not running")
    elif not main_running and suite.get("scored_cases", 0) > 0:
        status = "running_or_partial_suite"
        if not post_running:
            reasons.append("partial suite and postprocess watcher is not running")
    else:
        status = "attention"
        reasons.append("main watcher is not running and suite is incomplete")
        if not post_running:
            reasons.append("postprocess watcher is not running")

    ok = status in {"waiting_for_observations", "awaiting_postprocess", "postprocessed", "running_or_partial_suite"} and not reasons
    if status == "waiting_for_observations" and reasons and all(reason.startswith("coverage stale") for reason in reasons):
        ok = False
    return {
        "status": status,
        "ok": ok,
        "coverage_age_minutes": coverage_age_minutes,
        "max_coverage_age_minutes": max_coverage_age_minutes,
        "reasons": reasons,
    }


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    compact = args.target_date.replace("-", "")
    if args.output_root is None:
        args.output_root = args.ml_root / "live_inference" / f"collector_hindcast_shadow_unseen_{compact}_full_day_v1"
    log_root = args.ml_root / "live_inference" / "watch_logs"
    if args.main_pid_file is None:
        args.main_pid_file = log_root / f"{compact}_full_day_shadow.pid"
    if args.postprocess_pid_file is None:
        args.postprocess_pid_file = log_root / f"{compact}_full_day_postprocess.pid"
    if args.main_log_file is None:
        args.main_log_file = log_root / f"{compact}_full_day_shadow.log"
    if args.postprocess_log_file is None:
        args.postprocess_log_file = log_root / f"{compact}_full_day_postprocess.log"
    if args.rollup_root is None and not args.no_rollup:
        args.rollup_root = args.ml_root / "live_inference" / "shadow_rollups" / "shadow_rollup_latest"
    if args.campaign_log_root is None:
        args.campaign_log_root = log_root
    return args


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    args = resolve_defaults(args)
    output_root = args.output_root
    generated_at = utc_now()
    main_watcher = pid_status(args.main_pid_file)
    postprocess_watcher = pid_status(args.postprocess_pid_file)
    coverage = latest_coverage(args.main_log_file)
    suite = suite_status(output_root)
    artifacts = artifact_status(output_root)
    rollup = rollup_status(args.rollup_root) if not args.no_rollup else None
    campaign = campaign_status(args.campaign_id, args.campaign_log_root)
    return {
        "format": "corsewind.shadow_validation_status.v1",
        "generated_at_utc": generated_at,
        "target_date": args.target_date,
        "output_root": str(output_root),
        "main_watcher": main_watcher,
        "postprocess_watcher": postprocess_watcher,
        "latest_coverage": coverage,
        "postprocess_log_tail": (read_text(args.postprocess_log_file) or "").splitlines()[-8:],
        "suite": suite,
        "artifacts": artifacts,
        "rollup": rollup,
        "campaign": campaign,
        "disk": disk_status(args.disk_path),
        "health": health_status(
            generated_at_utc=generated_at,
            main_watcher=main_watcher,
            postprocess_watcher=postprocess_watcher,
            coverage=coverage,
            suite=suite,
            artifacts=artifacts,
            max_coverage_age_minutes=args.max_coverage_age_minutes,
        ),
    }


def render_markdown(status: dict[str, Any]) -> str:
    suite = status["suite"]
    coverage = status.get("latest_coverage") or {}
    disk = status["disk"]
    health = status["health"]
    lines = [
        "# Shadow Validation Status",
        "",
        f"- generated: `{status['generated_at_utc']}`",
        f"- target date: `{status['target_date']}`",
        f"- output: `{status['output_root']}`",
        f"- health: `{health['status']}` ok `{health['ok']}`",
        f"- main watcher running: `{status['main_watcher']['running']}` pid `{status['main_watcher'].get('pid')}`",
        f"- postprocess watcher running: `{status['postprocess_watcher']['running']}` pid `{status['postprocess_watcher'].get('pid')}`",
        f"- suite complete: `{suite['complete']}`",
        f"- cases: `{suite['scored_cases']}/{suite['case_count']}` scored, `{suite['shadow_cases']}/{suite['case_count']}` shadow",
        f"- disk free: `{disk['free_gb']} GB`",
        "",
    ]
    if health.get("reasons"):
        lines.extend(["## Health Reasons", ""])
        for reason in health["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
    if coverage:
        lines.extend(
            [
                "## Latest Coverage",
                "",
                f"- log time: `{coverage.get('log_time_utc')}`",
                f"- complete: `{coverage.get('complete')}`",
                f"- age minutes: `{health.get('coverage_age_minutes')}`",
                f"- target end: `{coverage.get('target_end_utc')}`",
                f"- missing: `{', '.join(coverage.get('missing') or [])}`",
                "",
                "| Spot | Latest observation UTC |",
                "| --- | --- |",
            ]
        )
        for spot, value in sorted((coverage.get("latest_by_spot") or {}).items()):
            lines.append(f"| `{spot}` | `{value}` |")
        lines.append("")
    if suite.get("cases"):
        lines.extend(["## Suite Cases", "", "| Case | Score | Shadow | Rows | Scored end |", "| --- | --- | --- | ---: | --- |"])
        for case in suite["cases"]:
            lines.append(
                f"| `{case.get('run_id')}` | `{case.get('score')}` | `{case.get('shadow')}` | "
                f"{case.get('rows') or ''} | `{case.get('target_end_scored_utc')}` |"
            )
        lines.append("")
    artifacts = status.get("artifacts") or {}
    lines.extend(["## Artifacts", "", "| Artifact | Exists | Decision | Cases | Rows |", "| --- | ---: | --- | ---: | ---: |"])
    for name, item in artifacts.items():
        lines.append(
            f"| `{name}` | `{item.get('exists')}` | `{item.get('decision') or ''}` | "
            f"{item.get('case_count') or ''} | {item.get('joined_rows') or ''} |"
        )
    lines.append("")

    rollup = status.get("rollup")
    if rollup:
        aggregate = rollup.get("aggregate_json") or {}
        wind_gate = rollup.get("wind_router_gate") or {}
        gust_gate = rollup.get("gust_guarded_stacker_gate") or {}
        promotion_review = rollup.get("promotion_review") or {}
        promotion_decision = rollup.get("promotion_decision") or {}
        threshold_audit = rollup.get("threshold_guard_audit") or {}
        evidence = promotion_review.get("evidence_progress") or {}
        best = promotion_review.get("best") or {}
        best_by_rmse = promotion_review.get("best_by_rmse") or {}
        lines.extend(
            [
                "## Multi-Day Rollup",
                "",
                f"- rollup: `{rollup.get('rollup_root')}`",
                f"- complete suites: `{rollup.get('complete_suite_count')}`",
                f"- aggregate exists: `{aggregate.get('exists')}`",
                f"- aggregate cases: `{aggregate.get('case_count')}`",
                f"- aggregate shadow cases: `{aggregate.get('shadow_case_count')}`",
                f"- aggregate rows: `{aggregate.get('joined_rows')}`",
                f"- wind router gate: `{wind_gate.get('decision')}` passed `{wind_gate.get('passed')}`",
                f"- gust guarded stacker gate: `{gust_gate.get('decision')}` passed `{gust_gate.get('passed')}`",
                f"- evidence ready: `{evidence.get('ready')}`",
                f"- evidence days/cases/rows: `{evidence.get('actual_days')}/{evidence.get('required_days')}` "
                f"`{evidence.get('case_count')}/{evidence.get('required_cases')}` "
                f"`{evidence.get('joined_rows')}/{evidence.get('required_rows')}`",
                f"- best wind candidate: `{(best.get('wind') or {}).get('candidate')}` decision `{(best.get('wind') or {}).get('decision')}` "
                f"performance fails `{(best.get('wind') or {}).get('performance_failed_check_count')}`",
                f"- best gust candidate: `{(best.get('gust') or {}).get('candidate')}` decision `{(best.get('gust') or {}).get('decision')}` "
                f"performance fails `{(best.get('gust') or {}).get('performance_failed_check_count')}`",
                f"- best wind gap summary: `{(best.get('wind') or {}).get('performance_gap_summary')}`",
                f"- best gust gap summary: `{(best.get('gust') or {}).get('performance_gap_summary')}`",
                f"- best wind by RMSE: `{(best_by_rmse.get('wind') or {}).get('candidate')}` rmse `{(best_by_rmse.get('wind') or {}).get('rmse_ms')}`",
                f"- best gust by RMSE: `{(best_by_rmse.get('gust') or {}).get('candidate')}` rmse `{(best_by_rmse.get('gust') or {}).get('rmse_ms')}`",
                f"- final promotion decision: `{promotion_decision.get('decision')}`",
                f"- threshold guard audit rows: `{threshold_audit.get('rows')}`",
                f"- threshold guard wind gains: `{threshold_audit.get('wind_rmse_gain_vs_baseline')}`",
                f"- threshold guard gust gains: `{threshold_audit.get('gust_rmse_gain_vs_baseline')}`",
                f"- threshold guard local risk flags: `{((threshold_audit.get('risk_flags') or {}).get('flag_count'))}`",
                "",
            ]
        )
        if promotion_decision.get("targets"):
            lines.extend(
                [
                    "### Promotion Decision",
                    "",
                    "| Target | Decision | Blocker | Candidate | Best Gate | Best RMSE |",
                    "| --- | --- | --- | --- | --- | --- |",
                ]
            )
            for target, item in sorted((promotion_decision.get("targets") or {}).items()):
                lines.append(
                    f"| `{target}` | `{item.get('decision')}` | `{item.get('blocker_type')}` | "
                    f"`{item.get('candidate')}` | `{item.get('best_candidate')}` | `{item.get('best_rmse_candidate')}` |"
                )
            lines.append("")
        if wind_gate.get("reasons") or gust_gate.get("reasons"):
            lines.extend(["### Gate Reasons", ""])
            for reason in wind_gate.get("reasons") or []:
                lines.append(f"- wind: {reason}")
            for reason in gust_gate.get("reasons") or []:
                lines.append(f"- gust: {reason}")
        lines.append("")

    campaign = status.get("campaign")
    if campaign:
        watcher = campaign.get("watcher") or {}
        lines.extend(
            [
                "## Shadow Campaign",
                "",
                f"- campaign id: `{campaign.get('campaign_id')}`",
                f"- running: `{watcher.get('running')}` pid `{watcher.get('pid')}`",
                f"- log: `{campaign.get('log_file')}`",
                f"- latest: `{campaign.get('latest_line')}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    status = build_status(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(status, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(status), encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", default="2026-07-02")
    parser.add_argument("--ml-root", type=Path, default=Path("/srv/data/corsewind/ml_dataset"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--main-pid-file", type=Path)
    parser.add_argument("--postprocess-pid-file", type=Path)
    parser.add_argument("--main-log-file", type=Path)
    parser.add_argument("--postprocess-log-file", type=Path)
    parser.add_argument("--rollup-root", type=Path)
    parser.add_argument("--no-rollup", action="store_true")
    parser.add_argument("--campaign-id")
    parser.add_argument("--campaign-log-root", type=Path)
    parser.add_argument("--disk-path", type=Path, default=Path("/srv/data"))
    parser.add_argument("--max-coverage-age-minutes", type=float, default=45.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
