import json
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

REPO_ID = "MobileDev-Bench/mobiledev-bench"
DATASET_FILENAME = "data/test.jsonl"


def read_jsonl(path):
    """Read records without requiring consistent column types."""
    records = []

    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON on line {}: {}".format(
                        line_number, error
                    )
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    "Expected a JSON object on line {}".format(
                        line_number
                    )
                )

            records.append(record)

    if not records:
        raise ValueError("The dataset contains no records.")

    return records


def write_jsonl(path, records):
    """Preserve original values, including nested objects."""
    with open(path, "w", encoding="utf-8") as destination:
        for record in records:
            destination.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\nDownloading or reusing cached dataset...")

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=DATASET_FILENAME,
        repo_type="dataset",
    )

    benchmark_path = DATA_DIR / "benchmark.jsonl"

    # Preserve the original downloaded file unchanged.
    shutil.copyfile(local_path, benchmark_path)

    print("\nReading JSONL directly...")
    records = read_jsonl(benchmark_path)

    df = pd.DataFrame(records)

    print("\nTotal tasks: {}".format(len(records)))
    print("\nAvailable fields:")
    print("\n".join(df.columns))

    required = {
        "instance_id",
        "org",
        "repo",
        "lang",
        "problem_statement",
        "base",
        "fix_patch",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing fields: {}".format(
                sorted(missing)
            )
        )

    # Check required keys in every record, not just the combined columns.
    for index, record in enumerate(records, start=1):
        missing_keys = required - set(record)

        if missing_keys:
            raise ValueError(
                "Record {} is missing fields: {}".format(
                    index, sorted(missing_keys)
                )
            )

    android_records = [
        record
        for record in records
        if str(record.get("lang") or "").strip().lower()
        in {"java", "kotlin"}
    ]

    print("\nAndroid tasks: {}".format(len(android_records)))

    if not android_records:
        print("\nLanguage values found:")
        print(df["lang"].value_counts(dropna=False).to_string())
        raise ValueError("No Java or Kotlin tasks were found.")

    android_df = pd.DataFrame(android_records)

    counts = (
        android_df.groupby(["org", "repo"], dropna=False)
        .size()
        .reset_index(name="task_count")
        .sort_values(
            ["task_count", "org", "repo"],
            ascending=[False, True, True],
        )
    )

    print("\nAndroid repositories:")
    print(counts.to_string(index=False))

    inventory_fields = [
        "instance_id",
        "org",
        "repo",
        "lang",
        "problem_statement",
        "base",
    ]

    # Exclude developer fixes, tests, PR descriptions, and comments
    # from the inventory intended for retrieval.
    inventory = [
        {
            field: record[field]
            for field in inventory_fields
        }
        for record in android_records
    ]

    inventory_path = DATA_DIR / "android_inventory.jsonl"
    counts_path = DATA_DIR / "repository_counts.csv"

    write_jsonl(inventory_path, inventory)
    counts.to_csv(counts_path, index=False)

    print("\nBase field types:")
    base_types = pd.Series(
        [type(record["base"]).__name__ for record in android_records]
    )
    print(base_types.value_counts().to_string())

    print("\nExample Android issue:")
    example = android_records[0]
    print("Task: {}".format(example["instance_id"]))
    print(str(example["problem_statement"])[:2000])

    print("\nSaved files:")
    print(benchmark_path)
    print(inventory_path)
    print(counts_path)


if __name__ == "__main__":
    main()
