
import argparse
import json

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


import re
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET


def mask_noncode(text):
    """Mask comments/literals; preserve offsets and line breaks.

    Kotlin string interpolation is masked too in this conservative pilot.
    """
    out = list(text)
    i = 0
    size = len(text)

    def erase(start, end):
        for position in range(start, end):
            if text[position] not in '\r\n':
                out[position] = ' '

    while i < size:
        start = i
        if text.startswith('//', i):
            end = text.find('\n', i)
            i = size if end < 0 else end
            erase(start, i)
        elif text.startswith('/*', i):
            i += 2
            depth = 1
            while i < size and depth:
                if text.startswith('/*', i):
                    depth += 1
                    i += 2
                elif text.startswith('*/', i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            erase(start, i)
        elif text.startswith('"""', i):
            i += 3
            while i < size:
                if text.startswith('"""', i):
                    preceding = 0
                    j = i - 1
                    while j >= start and text[j] == '\\':
                        preceding += 1
                        j -= 1
                    if preceding % 2 == 0:
                        i += 3
                        break
                i += 1
            erase(start, i)
        elif text[i] in ('"', "'"):
            quote = text[i]
            i += 1
            while i < size:
                if text[i] == '\\':
                    i = min(i + 2, size)
                elif text[i] == quote:
                    i += 1
                    break
                else:
                    i += 1
            erase(start, i)
        else:
            i += 1
    return ''.join(out)


def build_graph(paths, contents):
    """Conservative lexical graph, not resolved compiler dependencies."""
    owners = defaultdict(set)
    cleaned = {}
    xml_roots = {}

    for index, (path, content) in enumerate(zip(paths, contents)):
        if path.endswith(('.java', '.kt')):
            code = mask_noncode(content)
            cleaned[index] = code
            for name in re.findall(
                r'\b(?:class|interface|object|enum)\s+'
                r'([A-Za-z_][A-Za-z0-9_]*)', code
            ):
                owners[('symbol', name)].add(index)

        if path.endswith('.xml'):
            try:
                xml_roots[index] = ET.fromstring(content)
            except ET.ParseError:
                # No raw-text fallback that would reintroduce prose edges.
                continue
            parts = Path(path).parts
            if 'res' in parts:
                pos = parts.index('res')
                if pos + 2 < len(parts):
                    resource_type = parts[pos + 1].split('-')[0]
                    if resource_type in {
                        'layout', 'drawable', 'menu', 'navigation',
                        'xml', 'anim', 'animator',
                    }:
                        owners[('resource', resource_type,
                                Path(path).stem)].add(index)

    graph = [set() for _ in paths]
    for index in range(len(paths)):
        references = set()
        if index in cleaned:
            code = cleaned[index]
            references.update(
                ('symbol', name)
                for name in re.findall(
                    r'\b[A-Za-z_][A-Za-z0-9_]*\b', code
                )
            )
            references.update(
                ('resource', kind, name)
                for kind, name in re.findall(
                    r'\bR\.(\w+)\.(\w+)', code
                )
            )
        elif index in xml_roots:
            for element in xml_roots[index].iter():
                for value in element.attrib.values():
                    references.update(
                        ('resource', kind, name)
                        for kind, name in re.findall(
                            r'@\+?(\w+)/([A-Za-z0-9_]+)', value
                        )
                    )

        for reference in references:
            targets = owners.get(reference, set())
            if len(targets) != 1:
                continue
            target = next(iter(targets))
            if target != index:
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
        choices=["dev", "eval","validation"],
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
        args.task + "_dependency_v3.json"
    )

    with open(output_path, "w", encoding="utf-8") as destination:
        json.dump(
            {
                "instance_id": args.task,
                "base_sha": actual_sha,
                "method": "dependency_v3",
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
