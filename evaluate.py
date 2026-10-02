"""Publish the untouched chronological holdout metrics from the last training run.

Training and evaluation intentionally share one split and metric source. This
prevents a second evaluator from quietly changing labels, preprocessing, or
test-window boundaries after a model has been selected.
"""

import json
import os

from train.train_models import write_evaluation_csv


ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(ROOT, "train", "models", "training_summary.json")


def main() -> None:
    if not os.path.exists(SUMMARY_PATH):
        raise SystemExit("No training summary found. Run python train/train_models.py first.")
    with open(SUMMARY_PATH, encoding="utf-8") as file:
        summary = json.load(file)
    results = summary.get("results", [])
    if not results:
        raise SystemExit("The training summary contains no successful reservoir runs.")

    output_path = write_evaluation_csv(summary)
    print(f"Training run: {summary.get('run_id', 'legacy summary')}")
    expected_dams = set(summary.get("expected_dams", []))
    trained_dams = set(summary.get("trained_dams", [result["dam_name"] for result in results]))
    failures = summary.get("failures", [])
    complete = summary.get("run_status") == "complete" if "run_status" in summary else (
        not failures and (not expected_dams or expected_dams == trained_dams)
    )
    print(f"Training status: {'complete' if complete else 'partial'}")
    print(summary.get("test_policy", "Metrics copied from the saved chronological holdout."))
    print(f"Published {len(results)} dams x 4 horizons to {output_path}")
    for result in results:
        skills = result.get("test_skill_vs_persistence", {})
        selected = result.get("horizon_strategy", {})
        details = []
        for key in ("t+1", "t+7", "t+14", "t+30"):
            score = skills.get(key)
            if score is None:
                model_mae = result["test_storage_metrics"][key]["mae_tmc"]
                baseline_mae = result["persistence_baseline"][key]["mae_tmc"]
                score = {"mae_skill_pct": (1 - model_mae / baseline_mae) * 100 if baseline_mae > 1e-12 else None}
            skill = score.get("mae_skill_pct")
            skill_label = f"{skill:.1f}%" if skill is not None else "n/a"
            details.append(f"{key} {selected.get(key, {}).get('selected', '?')}, MAE skill {skill_label}")
        print(f"{result['dam_name']}: " + " | ".join(details))
    if not complete:
        failed_names = ", ".join(item.get("file", "unknown") for item in failures)
        suffix = f" Failures: {failed_names}." if failed_names else ""
        raise SystemExit(
            f"Partial evaluation report published; only {len(trained_dams)} of "
            f"{len(expected_dams) or len(trained_dams)} expected reservoirs are present.{suffix}"
        )


if __name__ == "__main__":
    main()
