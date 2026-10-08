
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPOSITORIES_DIR = PROJECT_ROOT / "repositories"
RESULTS_DIR = PROJECT_ROOT / "results"


def tokenize(text):
    """Split identifiers such as SwipeRefreshLayout into words."""
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)

    return re.findall(r"[a-z][a-z0-9]*", text.lower())


def is_candidate(path):
    parts = Path(path).parts
    excluded = {
        "test",
        "tests",
        "androidTest",
        "testFixtures",
        "build",
        ".gradle",
    }

    if any(part in excluded for part in parts):
        return False

    return path.endswith((
        ".java",
        ".kt",
        ".xml",
        ".gradle",
        ".gradle.kts",
        ".toml",
        ".properties",
    ))


def git_output(repository, arguments):
    return subprocess.check_output(
        ["git"] + arguments,
        cwd=str(repository),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="commons-app__apps-android-commons-6041",
    )
    parser.add_argument("--top-k", type=int, default=10)
    
    
    parser.add_argument(
        "--split",
        choices=["dev", "eval","validation"],
        default="dev",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")

    with open(
        DATA_DIR / (args.split + "_tasks.jsonl"),
        "r",
        encoding="utf-8",
    ) as source:
        tasks = [
            json.loads(line)
            for line in source
            if line.strip()
        ]

    matches = [
        task for task in tasks
        if task["instance_id"] == args.task
    ]

    if len(matches) != 1:
        raise ValueError("Task not found in selected split: " + args.task)

    task = matches[0]
    repository = REPOSITORIES_DIR / args.task

    if not (repository / ".git").exists():
        raise FileNotFoundError(
            "Prepare this repository first with "
            "prepare_repository.py --task " + args.task
        )

    actual_sha = git_output(
        repository, ["rev-parse", "HEAD"]
    ).decode("utf-8").strip()

    if actual_sha != task["base_sha"]:
        raise RuntimeError("Repository is at the wrong commit.")

    tracked_files = git_output(
        repository, ["ls-files", "-z"]
    ).decode("utf-8").split("\0")

    paths = []
    documents = []
    skipped = []

    for relative_path in sorted(tracked_files):
        if not relative_path or not is_candidate(relative_path):
            continue

        # Read committed content, so local edits cannot affect results.
        raw = git_output(
            repository,
            ["show", "HEAD:" + relative_path],
        )

        if b"\0" in raw:
            skipped.append(relative_path)
            continue

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append(relative_path)
            continue

        paths.append(relative_path)
        documents.append(relative_path + "\n" + content)

    if not documents:
        raise ValueError("No eligible text files found.")

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        stop_words=None,
        sublinear_tf=True,
        norm="l2",
    )

    file_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform(
        [task["problem_statement"]]
    )

    if query_vector.nnz == 0:
        raise ValueError(
            "No issue tokens appear in the indexed vocabulary."
        )

    # With normalized vectors, this is cosine similarity.
    scores = (file_vectors @ query_vector.T).toarray().ravel()

    # Break tied scores deterministically using file paths.
    ranking = sorted(
        range(len(paths)),
        key=lambda index: (-float(scores[index]), paths[index]),
    )

    predictions = [
        {
            "rank": rank,
            "path": paths[index],
            "score": float(scores[index]),
        }
        for rank, index in enumerate(
            ranking[:args.top_k], start=1
        )
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / (
        args.task + "_tfidf.json"
    )

    result = {
        "instance_id": args.task,
        "base_sha": actual_sha,
        "method": "tfidf",
        "indexed_files": len(paths),
        "skipped_files": skipped,
        "query": task["problem_statement"],
        "predictions": predictions,
    }

    with open(output_path, "w", encoding="utf-8") as destination:
        json.dump(
            result,
            destination,
            ensure_ascii=False,
            indent=2,
        )

    print("\nIndexed files:", len(paths))
    print("Skipped binary/non-UTF-8 files:", len(skipped))
    print("\nIssue:")
    print(task["problem_statement"])

    print("\nTop retrieved files:")
    for prediction in predictions:
        print(
            "{:2d}. {:.4f}  {}".format(
                prediction["rank"],
                prediction["score"],
                prediction["path"],
            )
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
