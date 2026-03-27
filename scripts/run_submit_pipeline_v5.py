import os
import shlex
import sys
import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_SCRIPT = REPO_ROOT / "scripts" / "backtest_v5_external_ml.py"
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "submit_predictions.py"
DEFAULT_PRED_CSV = os.path.expanduser("~/statiz/data/backtest_pred_v5_best.csv")
DEFAULT_PYTHON = str(REPO_ROOT / ".venv311" / "bin" / "python")


def run_cmd(cmd, env=None):
    print("+", shlex.join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser(
        description="Run v5 model prediction generation then submit to /prediction/savePrediction"
    )
    ap.add_argument("--skip-backtest", action="store_true", help="submit only, skip prediction generation")
    ap.add_argument("--python-bin", default="", help="python interpreter for child scripts (default: .venv311/bin/python if exists)")
    ap.add_argument("--home-dir", default=os.path.expanduser("~"), help="HOME for child scripts")

    ap.add_argument("--pred-csv", default=DEFAULT_PRED_CSV, help="prediction csv path for submit step")
    ap.add_argument("--train-start", default="20230101")
    ap.add_argument("--train-end", default="20261231")
    ap.add_argument("--test-start", default="20260101")
    ap.add_argument("--test-end", default="20261231")
    ap.add_argument("--feature-filter", choices=["strict", "balanced", "none"], default="strict")
    ap.add_argument("--select-objective", choices=["composite", "logloss", "accuracy"], default="composite")
    ap.add_argument("--selection-pool", choices=["stable_tree", "all"], default="stable_tree")
    ap.add_argument("--model-pool", choices=["full", "stable_tree_only"], default="stable_tree_only")
    ap.add_argument("--season-mode", choices=["regular", "exhibition", "all"], default="regular")
    ap.add_argument("--pipeline-test", action="store_true", help="simple smoke test mode (preseason-focused)")
    ap.add_argument("--include-non-regular", action="store_true", help="include exhibition/postseason rows")

    ap.add_argument("--date", default="", help="submit only exact date YYYYMMDD")
    ap.add_argument("--date-from", default="", help="submit date >= YYYYMMDD")
    ap.add_argument("--date-to", default="", help="submit date <= YYYYMMDD")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--sleep-sec", type=float, default=0.15)

    args = ap.parse_args()

    if args.python_bin:
        py = args.python_bin
    elif os.path.exists(DEFAULT_PYTHON):
        py = DEFAULT_PYTHON
    else:
        py = sys.executable

    child_env = os.environ.copy()
    child_env["HOME"] = args.home_dir
    child_env.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    if not args.skip_backtest:
        bt_cmd = [
            py,
            str(BACKTEST_SCRIPT),
            "--train-start",
            args.train_start,
            "--train-end",
            args.train_end,
            "--test-start",
            args.test_start,
            "--test-end",
            args.test_end,
            "--feature-filter",
            args.feature_filter,
            "--select-objective",
            args.select_objective,
            "--selection-pool",
            args.selection_pool,
            "--model-pool",
            args.model_pool,
            "--season-mode",
            args.season_mode,
        ]
        if args.include_non_regular:
            bt_cmd.append("--include-non-regular")
        if args.pipeline_test:
            bt_cmd.append("--pipeline-test")
        run_cmd(bt_cmd, env=child_env)

    sub_cmd = [
        py,
        str(SUBMIT_SCRIPT),
        "--in-csv",
        args.pred_csv,
        "--retries",
        str(args.retries),
        "--timeout",
        str(args.timeout),
        "--sleep-sec",
        str(args.sleep_sec),
    ]
    if args.limit > 0:
        sub_cmd += ["--limit", str(args.limit)]
    if args.resume:
        sub_cmd.append("--resume")
    if args.dry_run:
        sub_cmd.append("--dry-run")
    if args.date:
        sub_cmd += ["--date", args.date]
    if args.date_from:
        sub_cmd += ["--date-from", args.date_from]
    if args.date_to:
        sub_cmd += ["--date-to", args.date_to]

    run_cmd(sub_cmd, env=child_env)
    print("DONE pipeline")


if __name__ == "__main__":
    main()
