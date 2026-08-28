"""End-to-end orchestrator for the full analytics pipeline.

Runs, in order:

    ingest -> cleanse -> transform -> sql_analytics -> feature_engineering
    -> train_model -> evaluate

Each stage is an already-independent, already-tested CLI script
(`src/ingestion/ingest.py`, `src/transformation/cleanse.py`, ...); this
module doesn't reimplement any of their logic, and doesn't import them
either. It runs each one as its own subprocess -- the exact command you'd
type by hand to run that stage alone -- for two reasons:

- **Isolation.** Every stage builds and tears down its own SparkSession.
  Doing that seven times inside one long-lived Python process (rather than
  seven separate ones) is exactly the kind of thing that occasionally goes
  sideways in subtle, hard-to-reproduce ways (lingering JVM/gateway state,
  a stray `sys.exit()` inside argparse taking down the whole orchestrator
  instead of just one stage). A fresh process per stage means a stage's
  failure mode is contained to that stage.
- **An unambiguous success signal.** A subprocess's exit code is a plain,
  unavoidable fact -- there's no way for a stage to succeed at the Python
  level but still have quietly failed (e.g. a caught-and-swallowed
  exception). If a stage's process exits non-zero, the pipeline treats it
  as failed, full stop.

Usage:
    python src/pipeline.py --stage all
    python src/pipeline.py --stage ingest
    python src/pipeline.py --stage cleanse transform
    python src/pipeline.py --stage all --env databricks
    python src/pipeline.py --stage all --continue-on-failure
    python src/pipeline.py --stage all --dry-run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import load_yaml_config  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402

DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Stage definitions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    name: str
    script: Path
    description: str


STAGES: dict[str, StageSpec] = {
    "ingest": StageSpec(
        "ingest",
        ROOT_DIR / "src" / "ingestion" / "ingest.py",
        "Ingest the raw transactions CSV into the processed data layer",
    ),
    "cleanse": StageSpec(
        "cleanse",
        ROOT_DIR / "src" / "transformation" / "cleanse.py",
        "Deduplicate, standardize, and impute the ingested transactions",
    ),
    "transform": StageSpec(
        "transform",
        ROOT_DIR / "src" / "transformation" / "transform.py",
        "Build the 5 analytics-ready aggregated tables (revenue, RFM, top products, payment mix)",
    ),
    "sql_analytics": StageSpec(
        "sql_analytics",
        ROOT_DIR / "src" / "transformation" / "sql_runner.py",
        "Run the Spark SQL analytics query catalog against the processed data",
    ),
    "feature_engineering": StageSpec(
        "feature_engineering",
        ROOT_DIR / "src" / "ml" / "feature_engineering.py",
        "Build the clustering and churn ML feature tables",
    ),
    "train_model": StageSpec(
        "train_model",
        ROOT_DIR / "src" / "ml" / "train_model.py",
        "Train and persist the customer-segmentation and churn models",
    ),
    "evaluate": StageSpec(
        "evaluate",
        ROOT_DIR / "src" / "ml" / "evaluate.py",
        "Evaluate both models and write metrics + a confusion matrix plot to /reports",
    ),
}

# The pipeline's one true order. `--stage` accepts names in any order, but
# execution always follows this sequence -- running, say, cleanse before
# ingest isn't a valid alternate path, it's just guaranteed to fail against
# missing input, so there's no reason to honor an out-of-order request.
STAGE_ORDER: list[str] = [
    "ingest",
    "cleanse",
    "transform",
    "sql_analytics",
    "feature_engineering",
    "train_model",
    "evaluate",
]

STATUS_LABELS = {"success": "OK", "failed": "FAILED", "errored": "ERROR", "skipped": "SKIPPED"}


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str  # "success" | "failed" | "errored" | "skipped"
    elapsed_seconds: float
    return_code: Optional[int]


def resolve_stages(requested: list[str]) -> list[StageSpec]:
    names = STAGE_ORDER if "all" in requested else [name for name in STAGE_ORDER if name in requested]
    return [STAGES[name] for name in names]


# --------------------------------------------------------------------------
# Running a single stage
# --------------------------------------------------------------------------


def run_stage(stage: StageSpec, config_path: Path, env: Optional[str], python_executable: str) -> StageResult:
    """Run one stage as a subprocess, timing it and logging its outcome.

    Stdout/stderr are left un-captured (inherited from this process) on
    purpose: each stage already emits its own structured JSON logs via its
    own `setup_logging()` call, plus Spark's own console progress output.
    Streaming that straight through -- rather than buffering it and
    replaying it after the fact -- means you see a long-running stage's
    progress live, the same as if you'd run it by hand.
    """
    command = [python_executable, str(stage.script), "--config", str(config_path)]
    if env:
        command += ["--env", env]

    logger.info(
        "stage_started",
        extra={"stage": stage.name, "description": stage.description, "command": " ".join(command)},
    )
    start = time.perf_counter()

    try:
        subprocess.run(command, cwd=str(ROOT_DIR), check=True)
    except subprocess.CalledProcessError as exc:
        elapsed_seconds = round(time.perf_counter() - start, 2)
        logger.error(
            "stage_failed",
            extra={"stage": stage.name, "elapsed_seconds": elapsed_seconds, "return_code": exc.returncode},
        )
        return StageResult(stage=stage.name, status="failed", elapsed_seconds=elapsed_seconds, return_code=exc.returncode)
    except OSError:
        # The process itself never started (e.g. a bad --python path) --
        # there's no return code to report, only the launch failure.
        elapsed_seconds = round(time.perf_counter() - start, 2)
        logger.exception("stage_errored", extra={"stage": stage.name, "elapsed_seconds": elapsed_seconds})
        return StageResult(stage=stage.name, status="errored", elapsed_seconds=elapsed_seconds, return_code=None)

    elapsed_seconds = round(time.perf_counter() - start, 2)
    logger.info("stage_completed", extra={"stage": stage.name, "elapsed_seconds": elapsed_seconds})
    return StageResult(stage=stage.name, status="success", elapsed_seconds=elapsed_seconds, return_code=0)


# --------------------------------------------------------------------------
# Running the full (or partial) pipeline
# --------------------------------------------------------------------------


def run_pipeline(
    stages: list[StageSpec],
    config_path: Path,
    env: Optional[str],
    python_executable: str,
    continue_on_failure: bool,
) -> list[StageResult]:
    results: list[StageResult] = []
    pipeline_start = time.perf_counter()

    for index, stage in enumerate(stages):
        result = run_stage(stage, config_path, env, python_executable)
        results.append(result)

        if result.status != "success" and not continue_on_failure:
            remaining = stages[index + 1 :]
            if remaining:
                logger.warning(
                    "pipeline_aborted_early",
                    extra={"failed_stage": stage.name, "skipped_stages": [s.name for s in remaining]},
                )
                # Recorded explicitly (not just omitted) so a downstream
                # stage's absence from a partial run is visible in the
                # summary rather than looking like it was never planned.
                results.extend(
                    StageResult(stage=s.name, status="skipped", elapsed_seconds=0.0, return_code=None)
                    for s in remaining
                )
            break

    total_elapsed_seconds = round(time.perf_counter() - pipeline_start, 2)
    print_summary(results, total_elapsed_seconds)
    return results


# --------------------------------------------------------------------------
# Human-facing output (dry-run plan, final summary)
# --------------------------------------------------------------------------


def print_plan(stages: list[StageSpec], config_path: Path, env: Optional[str], python_executable: str) -> None:
    print(f"Dry run -- {len(stages)} stage(s) would run, in this order:\n")
    for position, stage in enumerate(stages, start=1):
        command = [python_executable, str(stage.script), "--config", str(config_path)]
        if env:
            command += ["--env", env]
        print(f"  {position}. {stage.name} -- {stage.description}")
        print(f"     $ {' '.join(command)}\n")


def print_summary(results: list[StageResult], total_elapsed_seconds: float) -> None:
    width = 64
    print("\n" + "=" * width)
    print("PIPELINE SUMMARY")
    print("=" * width)
    for result in results:
        label = STATUS_LABELS[result.status]
        print(f"  [{label:^7}] {result.stage:<22} {result.elapsed_seconds:>8.2f}s")
    print("-" * width)
    print(f"  Total elapsed: {total_elapsed_seconds:.2f}s")
    print("=" * width + "\n")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["all"],
        choices=["all"] + STAGE_ORDER,
        metavar="STAGE",
        help=(
            "Stage(s) to run: 'all' or one or more of "
            f"{{{', '.join(STAGE_ORDER)}}}. Always executed in pipeline "
            "order regardless of the order given here."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml, forwarded to every stage (default: config/config.yaml)",
    )
    parser.add_argument(
        "--env",
        choices=["local", "databricks"],
        default=None,
        help="Override the 'environment' value from config.yaml, forwarded to every stage",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python executable to run each stage with (default: the interpreter running this script)",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Run every remaining stage even after one fails (default: stop at the first failure)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned stage order and commands without running anything",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    raw_config = load_yaml_config(args.config)
    setup_logging(raw_config.get("logging", {}).get("level", "INFO"))

    stages = resolve_stages(args.stage)
    config_path = args.config.resolve()
    python_executable = args.python or sys.executable

    if args.dry_run:
        print_plan(stages, config_path, args.env, python_executable)
        return

    logger.info(
        "pipeline_started",
        extra={"stages": [s.name for s in stages], "config": str(config_path), "env": args.env},
    )

    results = run_pipeline(stages, config_path, args.env, python_executable, args.continue_on_failure)

    failed_stages = [r.stage for r in results if r.status in ("failed", "errored")]
    if failed_stages:
        logger.error("pipeline_failed", extra={"failed_stages": failed_stages})
        sys.exit(1)

    logger.info("pipeline_succeeded", extra={"stages": [r.stage for r in results]})


if __name__ == "__main__":
    main()
