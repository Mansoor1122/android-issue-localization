import argparse
import json
import shlex
import subprocess

from retrieve_tfidf import (
    DATA_DIR,
    REPOSITORIES_DIR,
    RESULTS_DIR,
    is_candidate,
)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def parse_patch_path(text):
    text = text.split("\t", 1)[0]

    if text.startswith('"'):
        parts = shlex.split(text)
        if len(parts) != 1:
            raise ValueError("Unsupported patch path: " + text)
        text = parts[0]

    if text.startswith(("a/", "b/")):
        return text[2:]

    return text


def extract_changed_paths(patch):
    """Use pre-fix paths for edits/deletions, new paths for additions."""
    paths = set()
    old_path = None

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            old_path = None

        elif line.startswith("--- "):
            old_path = parse_patch_path(line[4:])

        elif line.startswith("+++ "):
            new_path = parse_patch_path(line[4:])

            if old_path and old_path != "/dev/null":
                paths.add(old_path)
            elif new_path and new_path != "/dev/null":
                paths.add(new_path)

    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="commons-app__apps-android-commons-6041",
    )
    parser.add_argument(
        "--method",
        choices=["tfidf", "dependency","dependency_v2","dependency_v3"],
        default="tfidf",
    )
    
    parser.add_argument(
        "--split",
        choices=["dev", "eval","validation"],
        default="dev",
    )
    args = parser.parse_args()

    split_tasks = read_jsonl(DATA_DIR / (args.split + "_tasks.jsonl"))
    task_matches = [
        task for task in split_tasks
        if task["instance_id"] == args.task
    ]

    if len(task_matches) != 1:
        raise ValueError(
            "Expected exactly one matching task in selected split."
        )

    task = task_matches[0]

    records = read_jsonl(DATA_DIR / "benchmark.jsonl")
    matches = [
        record for record in records
        if record["instance_id"] == args.task
    ]

    if len(matches) != 1:
        raise ValueError("Expected exactly one benchmark task.")

    record = matches[0]

    prediction_path = RESULTS_DIR / (
        args.task + "_" + args.method + ".json"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            "Run retrieval first. Missing: " + str(prediction_path)
        )

    with open(prediction_path, "r", encoding="utf-8") as source:
        prediction_data = json.load(source)

    if prediction_data["instance_id"] != args.task:
        raise ValueError("Prediction task ID does not match.")

    if prediction_data["method"] != args.method:
        raise ValueError("Prediction method does not match.")

    if prediction_data["base_sha"] != task["base_sha"]:
        raise ValueError("Prediction commit does not match task.")

    repository = REPOSITORIES_DIR / args.task

    actual_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repository),
        text=True,
    ).strip()

    if actual_sha != prediction_data["base_sha"]:
        raise RuntimeError(
            "Repository commit differs from prediction commit."
        )

    tracked_files = set(
        subprocess.check_output(
            ["git", "ls-files", "-z"],
            cwd=str(repository),
        ).decode("utf-8").split("\0")
    )
    tracked_files.discard("")

    patch = record["fix_patch"]
    if not isinstance(patch, str):
        raise ValueError("Expected fix_patch to contain a diff string.")

    changed_paths = extract_changed_paths(patch)

    if not changed_paths:
        raise ValueError(
            "No changed paths extracted. Inspect the patch format."
        )

    all_targets = {
        path for path in changed_paths
        if is_candidate(path)
    }

    existing_targets = all_targets & tracked_files
    unavailable_targets = all_targets - tracked_files
    excluded_paths = changed_paths - all_targets

    if not all_targets:
        raise ValueError("No eligible target files for this task.")

    # Account for files explicitly skipped by the retrieval indexer.
    skipped_files = set(prediction_data.get("skipped_files", []))
    indexable_targets = existing_targets - skipped_files

    ranked_paths = [
        prediction["path"]
        for prediction in prediction_data["predictions"]
    ]

    if len(ranked_paths) != len(set(ranked_paths)):
        raise ValueError("Predictions contain duplicate paths.")

    invalid_predictions = [
        path for path in ranked_paths
        if path not in tracked_files
        or not is_candidate(path)
        or path in skipped_files
    ]
    if invalid_predictions:
        raise ValueError(
            "Predictions contain ineligible files: "
            + str(invalid_predictions)
        )

    print("\nTask:", args.task)
    print("Method:", args.method)

    print("\nEligible developer-modified files:")
    for path in sorted(all_targets):
        suffix = ""
        if path in unavailable_targets:
            suffix = " [absent at base commit]"
        elif path in skipped_files:
            suffix = " [skipped by indexer]"
        print(path + suffix)

    print("\nExcluded files:", len(excluded_paths))
    for path in sorted(excluded_paths):
        print("  " + path)

    print("\nRetrieval evaluation:")
    metrics = []

    for k in (5, 10):
        if len(ranked_paths) < k:
            print(
                "Skipping @{}: insufficient predictions.".format(k)
            )
            continue

        retrieved = set(ranked_paths[:k])
        hits = retrieved & all_targets

        precision = len(hits) / k
        all_recall = len(hits) / len(all_targets)

        existing_recall = (
            len(hits & existing_targets) / len(existing_targets)
            if existing_targets else None
        )

        indexable_recall = (
            len(hits & indexable_targets) / len(indexable_targets)
            if indexable_targets else None
        )

        complete_existing = (
            existing_targets.issubset(retrieved)
            if existing_targets else None
        )

        metrics.append({
            "k": k,
            "hits": sorted(hits),
            "precision": precision,
            "all_target_recall": all_recall,
            "existing_target_recall": existing_recall,
            "indexable_target_recall": indexable_recall,
            "complete_existing_target_coverage": complete_existing,
        })

        print("\nTop {}:".format(k))
        print("  Hits:", len(hits))
        print("  Precision: {:.3f}".format(precision))
        print("  All-target recall: {:.3f}".format(all_recall))

        if existing_recall is not None:
            print(
                "  Existing-target recall: {:.3f}".format(
                    existing_recall
                )
            )
            print(
                "  All existing targets retrieved:",
                complete_existing,
            )
        else:
            print("  Existing-target recall: N/A")

        if indexable_recall is not None:
            print(
                "  Indexable-target recall: {:.3f}".format(
                    indexable_recall
                )
            )
        else:
            print("  Indexable-target recall: N/A")

        for path in sorted(hits):
            print("  HIT: " + path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / (
        args.task + "_" + args.method + "_evaluation.json"
    )

    with open(output_path, "w", encoding="utf-8") as destination:
        json.dump(
            {
                "instance_id": args.task,
                "method": args.method,
                "base_sha": actual_sha,
                "all_targets": sorted(all_targets),
                "existing_targets": sorted(existing_targets),
                "indexable_targets": sorted(indexable_targets),
                "unavailable_targets": sorted(unavailable_targets),
                "excluded_paths": sorted(excluded_paths),
                "metrics": metrics,
            },
            destination,
            indent=2,
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
