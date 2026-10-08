
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# New repositories, selected without inspecting developer fixes.
REPOSITORIES = [
    ("AntennaPod", "AntennaPod"),
    ("Futsch1", "medTimer"),
    ("openhab", "openhab-android"),
]

SEED = 42
TASKS_PER_REPOSITORY = 5


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def extract_sha(base):
    if isinstance(base, str):
        try:
            base = json.loads(base)
        except json.JSONDecodeError:
            base = {"sha": base.strip()}

    if not isinstance(base, dict):
        raise ValueError("Unexpected base format.")

    sha = base.get("sha")
    if not isinstance(sha, str):
        raise ValueError("Missing commit SHA.")

    sha = sha.strip().lower()
    if len(sha) != 40 or any(
        character not in "0123456789abcdef" for character in sha
    ):
        raise ValueError("Invalid commit SHA: " + sha)

    return sha


def main():
    output = DATA_DIR / "validation_tasks.jsonl"

    if output.exists():
        raise FileExistsError(
            "Validation sample already exists. Preserve it."
        )

    inventory = read_jsonl(DATA_DIR / "android_inventory.jsonl")
    previous = (
        read_jsonl(DATA_DIR / "dev_tasks.jsonl")
        + read_jsonl(DATA_DIR / "eval_tasks.jsonl")
    )
    previous_ids = {task["instance_id"] for task in previous}
    previous_commits = {
        (task["org"], task["repo"], task["base_sha"])
        for task in previous
    }

    selected = []

    for org, repo in REPOSITORIES:
        candidates = sorted(
            [
                task for task in inventory
                if task["org"] == org
                and task["repo"] == repo
                and task["instance_id"] not in previous_ids
            ],
            key=lambda task: task["instance_id"],
        )

        if len(candidates) < TASKS_PER_REPOSITORY:
            raise ValueError("Insufficient tasks for " + repo)

        rng = random.Random("{}:{}/{}".format(SEED, org, repo))
        chosen = rng.sample(candidates, TASKS_PER_REPOSITORY)

        for task in chosen:
            sha = extract_sha(task["base"])
            if (org, repo, sha) in previous_commits:
                raise ValueError("Commit overlaps a previous split.")

            selected.append({
                "instance_id": task["instance_id"],
                "org": org,
                "repo": repo,
                "lang": task["lang"],
                "problem_statement": task["problem_statement"],
                "base_sha": sha,
            })

        print("{}/{}: {} selected".format(
            org, repo, len(chosen)
        ))

    selected.sort(key=lambda task: task["instance_id"])

    ids = [task["instance_id"] for task in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs.")

    with output.open("w", encoding="utf-8") as destination:
        for task in selected:
            destination.write(
                json.dumps(task, ensure_ascii=False) + "\n"
            )

    print("\nSaved {} fresh tasks: {}".format(
        len(selected), output
    ))


if __name__ == "__main__":
    main()
