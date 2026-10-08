
import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieve_dependency import build_graph
from retrieve_tfidf import (
    DATA_DIR,
    REPOSITORIES_DIR,
    RESULTS_DIR,
    git_output,
    is_candidate,
    tokenize,
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
        choices=["dev", "eval"],
        default="dev",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")

    with open(
        DATA_DIR / (args.split + "_tasks.jsonl"), encoding="utf-8"
    ) as source:
        tasks = [
            json.loads(line) for line in source if line.strip()
        ]

    matches = [
        task for task in tasks
        if task["instance_id"] == args.task
    ]
    if len(matches) != 1:
        raise ValueError("Task not found in selected split.")

    task = matches[0]
    repository = REPOSITORIES_DIR / args.task

    actual_sha = git_output(
        repository, ["rev-parse", "HEAD"]
    ).decode("utf-8").strip()

    if actual_sha != task["base_sha"]:
        raise RuntimeError("Repository is at the wrong commit.")

    tracked = git_output(
        repository, ["ls-files", "-z"]
    ).decode("utf-8").split("\0")

    paths = []
    contents = []
    skipped = []

    for path in sorted(tracked):
        if not path or not is_candidate(path):
            continue

        raw = git_output(repository, ["show", "HEAD:" + path])

        if b"\0" in raw:
            skipped.append(path)
            continue

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append(path)
            continue

        paths.append(path)
        contents.append(content)

    if not paths:
        raise ValueError("No eligible files found.")

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        stop_words=None,
        sublinear_tf=True,
        norm="l2",
    )

    matrix = vectorizer.fit_transform([
        path + "\n" + content
        for path, content in zip(paths, contents)
    ])
    query = vectorizer.transform([task["problem_statement"]])

    if query.nnz == 0:
        raise ValueError("Issue has no indexed vocabulary.")

    similarity = (matrix @ query.T).toarray().ravel()

    if similarity.sum() <= 0:
        raise ValueError("No positive retrieval scores.")

    graph = build_graph(paths, contents)

    baseline_order = sorted(
        range(len(paths)),
        key=lambda index: (-float(similarity[index]), paths[index]),
    )

    # Expand only the top 10 positively scored baseline files.
    seeds = [
        index for index in baseline_order
        if similarity[index] > 0
    ][:10]

    seed_weights = np.zeros(len(paths))
    for index in seeds:
        seed_weights[index] = similarity[index]
    seed_weights /= seed_weights.sum()

    support = np.zeros(len(paths))

    for index in seeds:
        neighbors = graph[index]

        if neighbors:
            contribution = seed_weights[index] / len(neighbors)
            for neighbor in neighbors:
                support[neighbor] += contribution
        else:
            # Preserve mass when a seed has no graph connections.
            support[index] += seed_weights[index]

    # Rank-based fusion limits the effect of hub score magnitudes.
    baseline_rank_score = np.zeros(len(paths))
    for rank, index in enumerate(baseline_order, start=1):
        if similarity[index] > 0:
            baseline_rank_score[index] = 1.0 / (60 + rank)

    support_order = sorted(
        [
            index for index in range(len(paths))
            if support[index] > 0
        ],
        key=lambda index: (-float(support[index]), paths[index]),
    )

    support_rank_score = np.zeros(len(paths))
    for rank, index in enumerate(support_order, start=1):
        support_rank_score[index] = 1.0 / (60 + rank)

    scores = (
        0.8 * baseline_rank_score
        + 0.2 * support_rank_score
    )

    ranking = sorted(
        range(len(paths)),
        key=lambda index: (-float(scores[index]), paths[index]),
    )

    predictions = [
        {
            "rank": rank,
            "path": paths[index],
            "score": float(scores[index]),
            "tfidf_score": float(similarity[index]),
            "dependency_support": float(support[index]),
            "graph_degree": len(graph[index]),
        }
        for rank, index in enumerate(
            ranking[:args.top_k], start=1
        )
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / (
        args.task + "_dependency_v2.json"
    )

    with open(output_path, "w", encoding="utf-8") as destination:
        json.dump(
            {
                "instance_id": args.task,
                "base_sha": actual_sha,
                "method": "dependency_v2",
                "settings": {
                    "seed_count": 10,
                    "baseline_weight": 0.8,
                    "dependency_weight": 0.2,
                    "rank_constant": 60,
                    "expansion_steps": 1,
                },
                "indexed_files": len(paths),
                "skipped_files": skipped,
                "predictions": predictions,
            },
            destination,
            indent=2,
        )

    print("\nTop retrieved files:")
    for item in predictions:
        print(
            "{:2d}. {:.6f}  {}".format(
                item["rank"], item["score"], item["path"]
            )
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
