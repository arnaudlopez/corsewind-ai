#!/usr/bin/env python3
"""Check z2 background RMSE/rebuild jobs and important result artifacts."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=False)


def remote_script(args: argparse.Namespace) -> str:
    log_root = Path(args.remote_log_root)
    sweep_root = Path(args.ml_root) / f"benchmarks/calibrator_2025_to_2026_sweep_{args.sweep_suffix}"
    training_root = Path(args.ml_root) / "training_tables"
    if args.compact:
        bench_root = Path(args.ml_root).parent / "ml_dataset_z2_rebuild/benchmarks"
        selection_json = bench_root / "tabular_regime_v1_selection/tabular_regime_v1_selection.json"
        return "\n".join([
            "set +e",
            "echo '## host/date'; hostname; date",
            "echo '## resources'; free -h | sed -n '1,3p'; df -h /srv/data 2>/dev/null | tail -n 1 || true",
            "echo '## statuses'",
            f"for name in rebuild_regime_v1_2024_2026 regime_v1_after_rebuild_audit_train tabular_regime_v1_rmse09_audit regime_v1_grouped_by_lead_after_global regime_v1_grouped_by_spot_lead_after_lead tabular_regime_v1_selection tabular_regime_v1_assertion; do "
            f"echo -n \"$name=\"; cat {shlex.quote(str(log_root))}/$name.status 2>/dev/null || echo missing; done",
            "echo '## active processes'",
            "pgrep -af 'rebuild_regime_v1|regime_v1_after_rebuild|tabular_regime_audit|tabular_regime_v1_selection|tabular_regime_v1_assertion|regime_v1_grouped_by_lead|regime_v1_grouped_by_spot_lead|run_monthly_training_shards|run_training_backfill_pipeline|export_training_table_parquet|train_residual_correction_parquet' || true",
            "echo '## regime_v1 shards'",
            f"paths=$(find {shlex.quote(str(training_root))} -maxdepth 2 -path '*/residual_windsup_sst_prev_regime_v1_*/training_rows.parquet' -print | sort); "
            "printf '%s\\n' \"$paths\" | sed '/^$/d' | wc -l; printf '%s\\n' \"$paths\" | tail -12",
            "echo '## jsonl leftovers'",
            f"echo -n 'combined='; find {shlex.quote(str(training_root))} -maxdepth 2 -path '*/residual_windsup_sst_prev_regime_v1_*/training_rows.jsonl' -print | wc -l",
            f"echo -n 'chunks='; find {shlex.quote(str(Path(args.ml_root) / 'training_runs'))} -path '*/residual_windsup_sst_prev_regime_v1_*/chunks/*/training_rows.jsonl' -print | wc -l",
            "echo '## tabular audits'",
            f"find {shlex.quote(str(bench_root))} -maxdepth 3 -name tabular_regime_v1_rmse09_audit.json -path '*regime_v1*' -print 2>/dev/null | sort | while read f; do "
            "echo \"-- $f\"; python3 -c "
            + shlex.quote(
                "import json,sys; "
                "d=json.load(open(sys.argv[1])); "
                "print(json.dumps({k:d.get(k) for k in ['run_id','verdict','corrected_rmse','corrected_mae','raw_rmse','rmse_gain_pct_vs_raw','rmse_gain_pct_vs_previous_best','rmse_gap_to_threshold','metric_source','metric_count','source_parquet_count','train_row_count','test_row_count'] if k in d}, indent=2, sort_keys=True))"
            )
            + " \"$f\"; done",
            "echo '## tabular selection'",
            f"if [ -f {shlex.quote(str(selection_json))} ]; then python3 -c "
            + shlex.quote(
                "import json,sys; "
                "d=json.load(open(sys.argv[1])); best=d.get('best') or {}; "
                "print(json.dumps({'decision':d.get('decision'),'audit_count':d.get('audit_count'),'valid_audit_count':d.get('valid_audit_count'),'best_run_id':best.get('run_id'),'best_rmse':best.get('corrected_rmse'),'best_gap':best.get('rmse_gap_to_threshold'),'best_audit':best.get('path')}, indent=2, sort_keys=True))"
            )
            + f" {shlex.quote(str(selection_json))}; else echo missing; fi",
            "echo '## tabular assertion'",
            f"assertion={shlex.quote(str(bench_root / 'tabular_regime_v1_selection/tabular_regime_v1_assertion.json'))}; "
            "if [ -f \"$assertion\" ]; then python3 -c "
            + shlex.quote(
                "import json,sys; d=json.load(open(sys.argv[1])); "
                "print(json.dumps({'status':d.get('status'),'threshold_rmse':d.get('threshold_rmse'),'evidence':d.get('evidence'),'reasons':d.get('reasons')}, indent=2, sort_keys=True))"
            )
            + " \"$assertion\"; else echo missing; fi",
            "echo '## watcher tails'",
            f"for name in regime_v1_after_rebuild_audit_train tabular_regime_v1_rmse09_audit regime_v1_grouped_by_lead_after_global regime_v1_grouped_by_spot_lead_after_lead tabular_regime_v1_selection tabular_regime_v1_assertion; do "
            f"echo --$name; tail -n {args.tail_lines} {shlex.quote(str(log_root))}/$name.log 2>/dev/null || true; done",
        ])
    lines = [
        "set +e",
        f"echo '## host/date'; hostname; date",
        "echo '## resources'; free -h; df -h /srv/data 2>/dev/null || true",
        f"echo '## log root'; ls -lah {shlex.quote(str(log_root))} 2>/dev/null || true",
        "echo '## active rmse09/regime processes'",
        "pgrep -af 'rebuild_regime_v1|regime_v1_after_rebuild|tabular_regime_audit|tabular_regime_v1_selection|regime_v1_grouped_by_lead|regime_v1_grouped_by_spot_lead|run_monthly_training_shards|run_training_backfill_pipeline|export_training_table_parquet|train_residual_correction_parquet|rmse09' || true",
    ]
    for name in (
        "rebuild_training_shards",
        "rebuild_regime_v1_2024_2026",
        "rmse09_sequence_experiment",
        "rmse09_fresh_lowmem",
        "rmse09_fresh_lowmem_watcher",
        "regime_v1_after_rebuild_audit_train",
        "tabular_regime_v1_rmse09_audit",
        "regime_v1_grouped_by_lead_after_global",
        "regime_v1_grouped_by_spot_lead_after_lead",
        "tabular_regime_v1_selection",
        "tabular_regime_v1_assertion",
    ):
        pid = log_root / f"{name}.pid"
        status = log_root / f"{name}.status"
        log = log_root / f"{name}.log"
        lines.extend([
            f"echo '## {name}'",
            f"echo -n 'pid: '; cat {shlex.quote(str(pid))} 2>/dev/null || echo missing",
            f"pid=$(cat {shlex.quote(str(pid))} 2>/dev/null); if [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null; then echo running=true; else echo running=false; fi",
            f"echo -n 'status: '; cat {shlex.quote(str(status))} 2>/dev/null || echo missing",
            f"echo '-- log tail --'; tail -n {args.tail_lines} {shlex.quote(str(log))} 2>/dev/null || true",
        ])
    lines.extend([
        "echo '## rmse09 benchmark roots'",
        f"for d in {shlex.quote(str(Path(args.ml_root) / 'benchmarks/sequence_2025_windsurf_1h_rmse09_v1'))} "
        f"{shlex.quote(str(Path(args.ml_root) / 'benchmarks/sequence_2026_windsurf_1h_rmse09_v1'))}; do "
        "echo \"-- $d\"; "
        "if [ -d \"$d\" ]; then ls -lh \"$d\"/predictions*.parquet 2>/dev/null || true; else echo missing; fi; done",
        "echo '## audits/results'",
        f"for f in {shlex.quote(str(sweep_root / 'training_table_feature_audit.json'))} "
        f"{shlex.quote(str(sweep_root / 'calibrator_feature_selection_audit.json'))} "
        f"{shlex.quote(str(sweep_root / 'sweep_results.json'))} "
        f"{shlex.quote(str(sweep_root / 'rmse09_audit.json'))} "
        f"{shlex.quote(str(sweep_root / 'rmse09_error_analysis.json'))} "
        f"{shlex.quote(str(sweep_root / 'rmse09_decision.json'))} "
        f"{shlex.quote(str(sweep_root / 'rmse09_run_manifest.json'))} "
        f"{shlex.quote(str(training_root / 'fresh_full_feature_audit.json'))} "
        f"{shlex.quote(str(training_root / 'calibrator_feature_selection_audit.json'))} "
        f"{shlex.quote(str(Path(args.ml_root) / 'benchmarks/rmse09_environment_preflight.json'))}; do "
        "echo \"-- $f\"; "
        "if [ -f \"$f\" ]; then python3 -c "
        + shlex.quote(
            "import json,sys; "
            "d=json.load(open(sys.argv[1])); "
            "keys=['verdict','reasons','decision','summary','recommended_next_actions','best_model_family','best_run_name','best_fit_group','best_metric','effective_rmse','effective_rmse_source','best_split_coverage','model_family','run_name','fit_group','overall','threshold_gap','oracle_summary','audit_verdict','audit_effective_rmse','format','generated_at_utc','benchmark_train_root','benchmark_eval_root','sweep_root','predictions_file','final_assert_command','stale_shard_count','missing_shard_count','existing_shard_count','failure_count','max_training_features','required_patterns']; "
        "print(json.dumps({k:d.get(k) for k in keys if k in d}, indent=2, sort_keys=True))"
        )
        + " \"$f\"; else echo missing; fi; done",
        "echo '## regime_v1 rebuild'",
        f"echo 'parquet count:'; find {shlex.quote(str(training_root))} -maxdepth 2 "
        "-path '*/residual_windsup_sst_prev_regime_v1_*/training_rows.parquet' -print | sort | tee /tmp/corsewind_regime_v1_parquets.txt | wc -l",
        "echo 'latest regime_v1 parquets:'; tail -20 /tmp/corsewind_regime_v1_parquets.txt 2>/dev/null || true",
        "echo 'regime_v1 JSONL leftovers:'",
        f"echo -n 'combined='; find {shlex.quote(str(training_root))} -maxdepth 2 "
        "-path '*/residual_windsup_sst_prev_regime_v1_*/training_rows.jsonl' -print | wc -l",
        f"echo -n 'chunks='; find {shlex.quote(str(Path(args.ml_root) / 'training_runs'))} "
        "-path '*/residual_windsup_sst_prev_regime_v1_*/chunks/*/training_rows.jsonl' -print | wc -l",
        "echo '## regime_v1 tabular benchmark/audit'",
        f"for d in {shlex.quote(str(Path(args.ml_root).parent / 'ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_150000_regime_v1_2024_2025_to_2026_v1'))} "
        f"{shlex.quote(str(Path(args.ml_root).parent / 'ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_100000_regime_v1_2024_2025_to_2026_v1'))} "
        f"{shlex.quote(str(Path(args.ml_root).parent / 'ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_150000_regime_v1_grouped_by_lead_2024_2025_to_2026_v1'))} "
        f"{shlex.quote(str(Path(args.ml_root).parent / 'ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_100000_regime_v1_grouped_by_lead_2024_2025_to_2026_v1'))} "
        f"{shlex.quote(str(Path(args.ml_root).parent / 'ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_150000_regime_v1_grouped_by_spot_lead_2024_2025_to_2026_v1'))} "
        f"{shlex.quote(str(Path(args.ml_root).parent / 'ml_dataset_z2_rebuild/benchmarks/tabular_lgbm_100000_regime_v1_grouped_by_spot_lead_2024_2025_to_2026_v1'))}; do "
        "echo \"-- $d\"; "
        "for f in training_results.json tabular_regime_v1_rmse09_audit.json; do "
        "p=\"$d/$f\"; echo \"file: $p\"; "
        "if [ -f \"$p\" ]; then python3 -c "
        + shlex.quote(
            "import json,sys; "
            "d=json.load(open(sys.argv[1])); "
            "target=(d.get('models') or {}).get('labels__residual_wind_mean_ms',{}); "
            "short=target.get('corrected_nwp_eval_leads') or {}; "
            "raw=target.get('raw_nwp_eval_leads') or {}; "
            "keys=['format','run_id','verdict','threshold_rmse','corrected_rmse','raw_rmse','rmse_gain_pct_vs_raw','rmse_gain_pct_vs_previous_best','rmse_gap_to_threshold','metric_source','metric_count','source_parquet_count','train_row_count','test_row_count','feature_column_count','temporal_split_issue_time_utc','reasons','warnings']; "
            "summary={k:d.get(k) for k in keys if k in d}; "
            "if short: summary['model_corrected_eval_leads']=short; "
            "if raw: summary['model_raw_eval_leads']=raw; "
            "print(json.dumps(summary, indent=2, sort_keys=True))"
        )
        + " \"$p\"; else echo missing; fi; done; done",
        "echo '## regime_v1 tabular selection'",
        f"for f in {shlex.quote(str(Path(args.ml_root).parent / 'ml_dataset_z2_rebuild/benchmarks/tabular_regime_v1_selection/tabular_regime_v1_selection.json'))} "
        f"{shlex.quote(str(Path(args.ml_root).parent / 'ml_dataset_z2_rebuild/benchmarks/tabular_regime_v1_selection/tabular_regime_v1_selection.md'))}; do "
        "echo \"-- $f\"; "
        "if [ -f \"$f\" ] && echo \"$f\" | grep -q '\\.json$'; then python3 -c "
        + shlex.quote(
            "import json,sys; "
            "d=json.load(open(sys.argv[1])); "
            "best=d.get('best') or {}; "
            "print(json.dumps({'decision': d.get('decision'), 'audit_count': d.get('audit_count'), 'valid_audit_count': d.get('valid_audit_count'), 'best_run_id': best.get('run_id'), 'best_rmse': best.get('corrected_rmse'), 'best_gap': best.get('rmse_gap_to_threshold')}, indent=2, sort_keys=True))"
        )
        + " \"$f\"; elif [ -f \"$f\" ]; then tail -n 80 \"$f\"; else echo missing; fi; done",
        "echo '## fresh shard tail'",
        f"find {shlex.quote(str(training_root))} -maxdepth 2 -type f -name training_rows.parquet "
        "-printf '%TY-%Tm-%Td %TH:%TM %s %p\\n' 2>/dev/null | sort | tail -40",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="z2")
    parser.add_argument("--ml-root", default="/srv/data/corsewind/ml_dataset")
    parser.add_argument("--sweep-suffix", default="context_v1")
    parser.add_argument("--remote-log-root", default="/srv/data/corsewind/ml_dataset/run_logs")
    parser.add_argument("--ssh-connect-timeout", type=int, default=12)
    parser.add_argument("--tail-lines", type=int, default=80)
    parser.add_argument("--compact", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ssh = ["ssh", "-o", f"ConnectTimeout={args.ssh_connect_timeout}", args.host]
    run([*ssh, remote_script(args)], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
