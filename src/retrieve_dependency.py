
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieve_tfidf import (
    DATA_DIR,
    REPOSITORIES_DIR,
    RESULTS_DIR,
    git_output,
    is_candidate,
    tokenize,
)


def build_graph(paths, contents):
    """Build an approximate, undirected file-reference graph."""
    owners = defaultdict(set)

    for index, (path, content) in enumerate(zip(paths, contents)):
        if path.endswith((".java", ".kt")):
            # Class/interface/object declarations, including nested types.
            names = re.findall(
                r"\b(?:class|interface|object|enum)\s+"
                r"([A-Za-z_][A-Za-z0-9_]*)",
                content,
            )
            for name in names:
                owners[("symbol", name)].add(index)

        # Index resource files by Android resource type and name.
        parts = Path(path).parts
        if "res" in parts and path.endswith(".xml"):
            position = parts.index("res")
            if position + 2 < len(parts):
                resource_type = parts[position + 1].split("-")[0]
                if resource_type in {
                    "layout", "drawable", "menu",
                    "navigation", "xml", "anim", "animator",
                }:
                    owners[
                        ("resource", resource_type, Path(path).stem)
                    ].add(index)

    graph = [set() for _ in paths]

    for index, content in enumerate(contents):
        references = {
            ("symbol", name)
            for name in re.findall(
                r"\b[A-Za-z_][A-Za-z0-9_]*\b", content
            )
        }

        for resource_type, name in re.findall(
            r"\bR\.(\w+)\.(\w+)", content
        ):
            references.add(("resource", resource_type, name))

        for resource_type, name in re.findall(
            r"@\+?(\w+)/([A-Za-z0-9_]+)", content
        ):
            references.add(("resource", resource_type, name))

        for reference in references:
            targets = owners.get(reference, set())

            # Ignore ambiguous names with multiple defining files.
            if len(targets) != 1:
                continue

            target = next(iter(targets))
            if target == index:
                continue

            graph[index].add(target)
            graph[target].add(index)

    return graph


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

    # Identical TF-IDF configuration and candidate files to baseline.
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

    initial = similarity / similarity.sum()
    graph = build_graph(paths, contents)

    # Personalized graph diffusion: retain query relevance while
    # sharing weight with connected files.
    restart_weight = 0.5
    scores = initial.copy()

    for _ in range(30):
        propagated = np.zeros(len(paths))

        for index, neighbors in enumerate(graph):
            if neighbors:
                contribution = scores[index] / len(neighbors)
                for neighbor in neighbors:
                    propagated[neighbor] += contribution
            else:
                propagated[index] += scores[index]

        updated = (
            restart_weight * initial
            + (1 - restart_weight) * propagated
        )

        difference = np.abs(updated - scores).sum()
        scores = updated
        if difference < 1e-10:
            break

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
            "graph_degree": len(graph[index]),
        }
        for rank, index in enumerate(
            ranking[:args.top_k], start=1
        )
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / (
        args.task + "_dependency.json"
    )

    with open(output_path, "w", encoding="utf-8") as destination:
        json.dump(
            {
                "instance_id": args.task,
                "base_sha": actual_sha,
                "method": "dependency",
                "restart_weight": restart_weight,
                "indexed_files": len(paths),
                "skipped_files": skipped,
                "graph_edges": sum(map(len, graph)) // 2,
                "predictions": predictions,
            },
            destination,
            indent=2,
        )

    print("\nIndexed files:", len(paths))
    print("Graph edges:", sum(map(len, graph)) // 2)
    print("\nTop retrieved files:")

    for item in predictions:
        print(
            "{:2d}. {:.6f}  {}  (connections: {})".format(
                item["rank"],
                item["score"],
                item["path"],
                item["graph_degree"],
            )
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
