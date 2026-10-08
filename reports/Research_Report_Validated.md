# Android Issue File Retrieval: Validated Pilot

Mansoor Ahmad · 8 October 2026

## Research question
Does lightweight reference-based reranking improve retrieval of developer-modified files compared with TF-IDF? This study evaluates file localization, not generated patches or successful repairs.

## Experimental history
Four development tasks informed the original method. An initial 18-task evaluation across Commons and Tusky exposed a graph-extraction defect during subsequent source inspection: prose such as “interface should” created a false type declaration. V2 results remain historical results of that defective implementation. V3 masks comments and literals and parses XML separately. Its rerun on those 18 tasks is a post-hoc diagnostic, not fresh validation.

V3 was then kept fixed for a new 15-task sample, five tasks each from AntennaPod, medTimer, and openHAB. The reported selection procedure used seed 42 without inspecting fixes. Selection provenance and absence of all prior-split overlap are not independently rechecked by the three latest attachments alone. All 15 tasks ultimately completed; medTimer-584 preparation initially failed and succeeded on retry, without replacing the task.

## Implemented method
TF-IDF indexes file paths and raw contents using the shared baseline tokenizer, sublinear term frequency and normalized vectors. V3 changes graph extraction, not the TF-IDF document text: comments and literals remain in lexical retrieval. Candidate eligibility comes from the imported baseline module.

The graph connects unique simple-name type owners to matching identifiers in masked Java/Kotlin code. Connections are undirected and approximate, without import or scope resolution. XML attribute references and code R.type.name references connect to supported XML resource files. Named entries in values XML, image resources, XML class names, build dependency semantics and Kotlin interpolation are not resolved. Ambiguous owners are skipped; resource variants can therefore prevent edges. The declaration regex also remains imperfect for forms such as enum class.

The top ten positively scored TF-IDF files seed one-hop propagation. Each seed distributes normalized similarity evenly among its neighbors; isolated seeds retain their weight. Final score is 0.8/(60 + lexical rank) + 0.2/(60 + support rank), with absent signals contributing zero. Positive support on an isolated file therefore does not prove an incoming reference edge.

All 15 saved V3 settings agree: ten seeds, one expansion step, weights 0.8/0.2 and rank constant 60. The uploaded source SHA-256 is `db2955d81e9ad07ce4dbe3acb0bd3874c4d108bb34386b41e6e27a812eb0ffbe`. This fingerprints the submitted file; saved results do not themselves prove which exact source bytes produced them.

## Fresh validation results
All averages are per-task macro averages. Indexable targets are eligible developer-modified files available to the index at the base commit. All-target recall also counts eligible unavailable files. Developer-modified files are a proxy for relevance, not an exhaustive set of valid repair locations.

| Metric | TF-IDF | V3 | Difference, percentage points |
|---|---:|---:|---:|
| Indexable recall@5 | 13.50% | 13.79% | +0.29 |
| Precision@5 | 29.33% | 30.67% | +1.33 |
| All-target recall@5 | 12.00% | 11.94% | −0.06 |
| Indexable recall@10 | 17.56% | 20.43% | +2.87 |
| Precision@10 | 20.67% | 24.67% | +4.00 |
| All-target recall@10 | 15.39% | 17.71% | +2.32 |

Differences are calculated before rounding. At top five V3 has four wins, three losses and eight ties. At top ten it has five wins, zero losses and ten ties. Zero task-level recall losses does not imply no individual target was displaced.

| Repository (five tasks each) | TF-IDF recall@10 | V3 recall@10 |
|---|---:|---:|
| AntennaPod | 14.34% | 14.34% |
| medTimer | 20.63% | 28.62% |
| openHAB | 17.71% | 18.34% |

## Five improved cases
| Task | Recall@10, TF-IDF → V3 | Newly retrieved targets (V3 rank) |
|---|---:|---|
| Futsch1__medTimer-29 | 36.36% → 45.45% | Reminder.java (10) |
| Futsch1__medTimer-54 | 11.76% → 17.65% | PathHelper.java (10) |
| Futsch1__medTimer-584 | 15.00% → 20.00% | MedicineHelper.kt (8) |
| Futsch1__medTimer-70 | 20.00% → 40.00% | RescheduleWork.java (3), EditMedicine.java (6), TimeHelper.java (7) |
| openhab__openhab-android-3021 | 0.00% → 3.12% | ContentController.kt (6) |

Seven new top-ten target hits are Java/Kotlin files. medTimer-70 loses one previous target hit while gaining three, yielding a net gain of two. The other four cases each gain one hit. All promoted files have positive saved support and nonzero degree, consistent with graph-assisted promotion. Exact supporting seed edges cannot be reconstructed from these attachments, which contain no validation repository source or edge provenance. Baseline ranks beyond ten are not saved, so they must not be invented.

medTimer-584 retrieves four of twenty indexable targets, including two strings XML files, versus four of twenty-eight all targets. openHAB-3021 is a broad refactoring task with 32 indexable targets: one hit yields 3.125% recall. These denominators affect macro averages. openHAB-3770 misses Widget.kt with both methods: all ten candidates are string-resource files. This illustrates lexical candidate crowding, without establishing its exact cause.

## Earlier diagnostic, kept separate
| Method | Indexable recall@5 | Indexable recall@10 |
|---|---:|---:|
| TF-IDF | 23.35% | 25.35% |
| V2 | 22.54% | 31.21% |
| V3 | 22.23% | 28.43% |

These 18-task figures are the earlier diagnostic comparison, not pooled with fresh validation. Historical source tracing found legitimate request-code support for Commons-5851, false prose edges for Tusky-4599 and ranking displacement for Tusky-2757. The latest audit focuses on the 15 fresh pairs.

## Verification and limits
The audit checked task/commit/method metadata for predictions and evaluations; unique prediction paths; agreement between paired target sets; and recomputed hits, precision and all/existing/indexable recall from saved lists. All 15 pairs passed. This verifies internal consistency, not correctness of target extraction against original patches, reproduction of rankings from source, or statistical significance.

The sample is small, repository gains are uneven and related tasks may not be independent. Translation changes, refactoring and incidental edits can influence patch-based targets. Recall remains low. Graph extraction is lexical and omits important Android relationships. No timing, memory, patch generation, compilation or app tests were evaluated. A larger preregistered sample and source-level edge traces would strengthen a follow-up study; changing retrieval now would require a separately versioned experiment.

## Reproducibility
Preserve validation_tasks.jsonl, all predictions/evaluations, validation_run_status.json, the frozen source and its imported baseline module, evaluator, preparation script and environment versions. At each selected base commit run prepare_repository.py, retrieve_tfidf.py, retrieve_dependency_v3.py and evaluate_retrieval.py for both methods using --task ID --split validation. Do not substitute failed tasks. The audit JSON accompanies this report.

## Conclusion
On fifteen new tasks across three additional Android repositories, V3 increased mean indexable-target recall@10 by 2.87 percentage points. Gains were concentrated in medTimer. The evidence supports a modest candidate-coverage improvement in this pilot, with mixed top-five behavior and unresolved graph limitations.
