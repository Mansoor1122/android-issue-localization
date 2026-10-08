# Android Issue File Localization

A small research pilot comparing TF-IDF with conservative reference-based reranking for Android issue localization. This measures retrieval of developer-modified files, not automated repair success.

## Fresh validation
15 tasks, five each from AntennaPod, medTimer and openHAB; all completed, including one preparation retry.

| Metric | TF-IDF | Frozen V3 |
|---|---:|---:|
| Mean indexable recall@5 | 13.50% | 13.79% |
| Mean indexable recall@10 | 17.56% | 20.43% |
| Mean precision@10 | 20.67% | 24.67% |

Top-ten recall: five wins, zero losses, ten ties. Gains mainly come from medTimer. The earlier 18-task set is a post-hoc diagnostic after a V2 parsing defect was found; keep its results separate.

## Method
V3 masks Java/Kotlin comments and literals for graph extraction, parses XML attributes, and builds undirected approximate type/resource connections. Raw text remains in TF-IDF. Ten positive lexical seeds provide one-hop support; weighted reciprocal ranks combine lexical and support ranks (0.8/0.2, constant 60). Isolated seeds retain mass. Values-resource entries and Kotlin interpolation are not resolved.

## Run one task
Keep the supplied scripts in src and the frozen validation sample in data. Install numpy and scikit-learn in your Python environment; record exact versions. Git must be available.

```bash
TASK_ID=Futsch1__medTimer-584
python src/prepare_repository.py --task "$TASK_ID" --split validation
python src/retrieve_tfidf.py --task "$TASK_ID" --split validation
python src/retrieve_dependency_v3.py --task "$TASK_ID" --split validation
python src/evaluate_retrieval.py --task "$TASK_ID" --split validation --method tfidf
python src/evaluate_retrieval.py --task "$TASK_ID" --split validation --method dependency_v3
```

Both CLI scripts and the evaluator must accept validation. V3 imports retrieve_tfidf.py, so retain that dependency. Preserve all results and failures without replacing tasks. Record source hashes, environment versions and task base commits before future runs.

## Evidence and limitations
See [Research report](reports/Research_Report_Validated.md) and [Validation audit](reports/validation_audit.json). for verified scores and the five improved cases. Saved predictions and evaluation target sets were checked for internal consistency. Source-level edges for fresh cases were not traced. Small sample, approximate graph, patch-based relevance and uneven repository gains limit conclusions. No repair, performance or statistical-significance claim is made.
