import argparse
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPOSITORIES_DIR = PROJECT_ROOT / "repositories"


def run_git(arguments, cwd=None):
    command = ["git"] + arguments
    print("\nRunning:", " ".join(command), flush=True)

    subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
    )


def read_tasks(path):
    with open(path, "r", encoding="utf-8") as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        default="commons-app__apps-android-commons-6041",
        help="Task ID to prepare.",
    )

    parser.add_argument(
        "--split",
        choices=["dev", "eval","validation"],
        default="dev",
    )

    args = parser.parse_args()
    

    tasks = read_tasks(DATA_DIR / (args.split + "_tasks.jsonl"))
    matches = [
        task for task in tasks
        if task["instance_id"] == args.task
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one development task matching: "
            + args.task
        )

    task = matches[0]

    REPOSITORIES_DIR.mkdir(parents=True, exist_ok=True)

    # Each task gets its own directory and historical code version.
    repository_path = REPOSITORIES_DIR / task["instance_id"]
    repository_url = "https://github.com/{}/{}.git".format(
        task["org"], task["repo"]
    )
    sha = task["base_sha"]

    if not repository_path.exists():
        repository_path.mkdir()
        run_git(["init"], cwd=repository_path)
        run_git(
            ["remote", "add", "origin", repository_url],
            cwd=repository_path,
        )
    elif not (repository_path / ".git").exists():
        raise RuntimeError(
            "Directory exists but is not a Git repository: "
            + str(repository_path)
        )

    origin = subprocess.check_output(
        ["git", "remote", "get-url", "origin"],
        cwd=str(repository_path),
        text=True,
    ).strip()

    if origin != repository_url:
        raise RuntimeError(
            "Unexpected repository origin: " + origin
        )

    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=str(repository_path),
        text=True,
    ).strip()

    if status:
        raise RuntimeError(
            "Repository has local changes. "
            "Preserve them before changing commits."
        )

    run_git(
        ["fetch", "--depth", "1", "origin", sha],
        cwd=repository_path,
    )

    run_git(
        ["checkout", "--detach", sha],
        cwd=repository_path,
    )

    actual_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repository_path),
        text=True,
    ).strip()

    if actual_sha != sha:
        raise RuntimeError(
            "Commit mismatch: expected {}, found {}".format(
                sha, actual_sha
            )
        )

    tracked_files = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=str(repository_path),
    ).decode("utf-8").split("\0")

    tracked_files = [path for path in tracked_files if path]

    source_files = [
        path for path in tracked_files
        if path.endswith((".kt", ".java"))
    ]

    xml_files = [
        path for path in tracked_files
        if path.endswith(".xml")
    ]

    print("\nRepository prepared successfully")
    print("Task:", task["instance_id"])
    print("Commit:", actual_sha)
    print("Directory:", repository_path)
    print("Tracked files:", len(tracked_files))
    print("Java/Kotlin files:", len(source_files))
    print("XML files:", len(xml_files))

    print("\nIssue description:")
    print(task["problem_statement"])


if __name__ == "__main__":
    main()
