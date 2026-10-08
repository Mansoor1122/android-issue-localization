import json
import random
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

SELECTED_REPOS = {
    ("tuskyapp", "Tusky"),
    ("commons-app", "apps-android-commons"),
}

RANDOM_SEED = 42
DEV_TASKS_PER_REPO = 2


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as destination:
        for record in records:
            destination.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def extract_base_sha(base, instance_id):
    """Handle a base object or a JSON-encoded base object."""
    if isinstance(base, str):
        try:
            base = json.loads(base)
        except json.JSONDecodeError:
            # Also accept a raw commit SHA if encountered.
            base = {"sha": base.strip()}

    if not isinstance(base, dict):
        raise ValueError(
            "Unexpected base format for {}: {}".format(
                instance_id, type(base).__name__
            )
        )

    sha = base.get("sha")

    if not isinstance(sha, str):
        raise ValueError(
            "Missing base SHA for {}".format(instance_id)
        )

    sha = sha.strip()

    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise ValueError(
            "Invalid full commit SHA for {}: {!r}".format(
                instance_id, sha
            )
        )

    return sha.lower()


def main():
    source_path = DATA_DIR / "android_inventory.jsonl"

    if not source_path.exists():
        raise FileNotFoundError(
            "Run inspect_dataset.py before this script."
        )

    records = read_jsonl(source_path)
    selected = []

    for record in records:
        repo_key = (record["org"], record["repo"])

        if repo_key not in SELECTED_REPOS:
            continue

        # Explicitly allow only inputs needed by our experiment.
        selected.append({
            "instance_id": record["instance_id"],
            "org": record["org"],
            "repo": record["repo"],
            "lang": record["lang"],
            "problem_statement": record["problem_statement"],
            "base_sha": extract_base_sha(
                record["base"], record["instance_id"]
            ),
        })

    if not selected:
        raise ValueError("No tasks matched the selected repositories.")

    ids = [task["instance_id"] for task in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs found.")

    development = []
    evaluation = []

    for repo_key in sorted(SELECTED_REPOS):
        repo_tasks = sorted(
            [
                task
                for task in selected
                if (task["org"], task["repo"]) == repo_key
            ],
            key=lambda task: task["instance_id"],
        )

        if len(repo_tasks) <= DEV_TASKS_PER_REPO:
            raise ValueError(
                "Not enough tasks for {}".format(repo_key)
            )

        rng = random.Random(
            "{}:{}/{}".format(RANDOM_SEED, *repo_key)
        )
        rng.shuffle(repo_tasks)

        development.extend(repo_tasks[:DEV_TASKS_PER_REPO])
        evaluation.extend(repo_tasks[DEV_TASKS_PER_REPO:])

        print(
            "{}/{}: {} tasks ({} development, {} evaluation)".format(
                repo_key[0],
                repo_key[1],
                len(repo_tasks),
                DEV_TASKS_PER_REPO,
                len(repo_tasks) - DEV_TASKS_PER_REPO,
            )
        )

    development.sort(key=lambda task: task["instance_id"])
    evaluation.sort(key=lambda task: task["instance_id"])

    # Prevent the same repository commit appearing in both splits.
    dev_commits = {
        (task["org"], task["repo"], task["base_sha"])
        for task in development
    }
    overlap = [
        task["instance_id"]
        for task in evaluation
        if (task["org"], task["repo"], task["base_sha"])
        in dev_commits
    ]

    if overlap:
        raise ValueError(
            "Shared base commits across splits. "
            "We need a commit-grouped split before continuing: {}".format(
                overlap
            )
        )

    write_jsonl(DATA_DIR / "dev_tasks.jsonl", development)
    write_jsonl(DATA_DIR / "eval_tasks.jsonl", evaluation)

    print("\nDevelopment tasks:")
    for task in development:
        print(
            "{}  {}".format(
                task["instance_id"], task["base_sha"]
            )
        )

    print("\nSaved:")
    print(DATA_DIR / "dev_tasks.jsonl")
    print(DATA_DIR / "eval_tasks.jsonl")


if __name__ == "__main__":
    main()
