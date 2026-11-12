# How Far Does Lexical Retrieval Get a Small Code Agent? An Empirical Study of Issue Localization on Pre-built Code Graphs

**Subtitle:** Reproducible file- and function-level localization baselines (BM25, graph expansion) on the Gemma 4 Developer Agent dev set, with a label-generation tool for patch-to-graph-node mapping.

## Abstract

Before a small language model can repair a bug in an unfamiliar repository it must find where to look. The Gemma 4 Developer Agent competition ships every task with a pre-built code graph (functions and classes as nodes, call/import relations as edges) and exposes graph tools to the agent, but it offers no measurement of how well simple retrieval over that graph locates the code a gold patch actually changes. We contribute (1) an open, deterministic procedure that maps unified-diff patches to graph nodes, producing file-level and function-level localization labels for the development tasks; (2) a zero-model baseline study of identifier-aware BM25 over graph node text, with and without one- and two-hop graph expansion and with test-code down-weighting; and (3) a script-sized localization routine that an agent can call as a skill. On the 28 development tasks (fastapi, requests, rich) for which a graph and a non-trivial gold label were available, BM25 alone places a gold file in the top 5 for 71.4% of tasks and a gold function in the top 5 for 35.7%. One-hop graph expansion raises these to 75.0% and 39.3%, a difference of one task each, and is not statistically distinguishable from BM25 at this sample size. Function-level localization remains the bottleneck: the correct function is in the top 10 only half of the time. We discuss why (long issue text, multi-hunk patches, patches that add new code) and release the labels and code so that other participants can evaluate retrieval strategies without running a model.

## 1. Introduction

Repository-level issue resolution benchmarks such as SWE-bench (Jimenez et al., 2024) require an agent to edit the right files before any test can pass. Pipeline-style systems such as Agentless (Xia et al., 2024) make this explicit with a separate localization stage, and agentic systems such as SWE-agent (Yang et al., 2024) spend a large share of their turns exploring the repository. For small models with limited context and a fixed 12-hour budget, wasted exploration turns are costly, so the quality of the first retrieval step matters.

The Gemma 4 Developer Agent competition provides, for every task, a pre-computed graph of the repository and embeddings of its nodes, and three tools (`get_code_neighbors`, `search_similar_code`, `get_code_subgraph`) over them. Participants therefore have a ready-made substrate for localization, but two questions are open to anyone designing an agent: how good is plain lexical retrieval on this graph, and does the graph structure help it?

We answer both with a model-free experiment. Because no LLM is involved, the study is cheap, deterministic, and exactly reproducible; it measures an upper-level component of any agent rather than an end-to-end solve rate. Our contributions are:

1. **A labeling procedure** that converts each task's gold patch into the set of graph nodes it modifies (function/class granularity) and the set of modules it touches, handling plain unified diffs without `diff --git` headers, new-file patches, and non-Python files.
2. **A baseline study** of four retrieval variants on the development set, reported at file and function level, per repository.
3. **An analysis of failure modes** and honest limits of the evaluation, including its small sample.
4. **Released artifacts:** `loc_labels.csv`, `loc_results.csv`, `loc_results_by_repo.csv`, and a single-file script (`study2.py`) that reproduces everything from the competition data.

## 2. Related Work

**Issue resolution benchmarks and agents.** SWE-bench defines the task of producing a patch from an issue and repository snapshot, scored by tests (Jimenez et al., 2024). SWE-agent introduced an agent-computer interface with file viewing and editing commands (Yang et al., 2024). Agentless showed that a fixed pipeline of localization, repair, and patch validation can be competitive, and it localizes hierarchically from files to functions to lines (Xia et al., 2024). Our study isolates the first two levels of that hierarchy.

**Lexical and graph-based retrieval for code.** BM25 (Robertson and Zaragoza, 2009) remains a strong baseline for matching issue text against source code because issues usually mention identifiers, error messages, and API names. Graph-augmented methods expand retrieved nodes along call or dependency edges on the premise that code relevant to a bug is structurally near code that is textually relevant. We test that premise directly with a fixed expansion rule rather than a learned one.

**Positioning.** Unlike end-to-end leaderboard scores, which conflate localization with repair quality, our measurements are independent of any model and can be run in seconds per task.

## 3. Method

### 3.1 Data

We use the competition's development tasks (129 tasks listed in `tasks.jsonl`: `instance_id`, `repo`, `base_commit`, `patch`, `test_patch`, `problem_statement`, `hints_text`) and the per-commit graph files (`{directed, multigraph, nodes:[{id,name,text}], edges:[{source,target,type}]}`). A node id is a dotted symbol path (for example `requests.utils.get_netrc_auth`); `text` holds the source of the function or class. In our run, graph files could be retrieved for the first 30 tasks (16 fastapi, 8 requests, 6 rich). Tasks without a retrievable graph are excluded; we make no claims about them.

### 3.2 Gold labels from patches

Patches in this dataset are plain unified diffs: file headers are `--- a/<path>` / `+++ b/<path>` with no `diff --git` line. Our parser reads these headers and the `@@` hunks. For each modified Python file outside tests, docs, and scripts we:

1. Resolve the file to a module prefix by stripping directories (e.g. `src/`) until the prefix matches node ids in that task's graph (`src/requests/utils.py` becomes `requests.utils`). The module prefixes form the **gold file set**.
2. For each hunk, collect probe lines (removed lines; if a hunk only adds lines, the surrounding context lines). Each probe is normalized to a lowercase word sequence. Among nodes under the module prefix, the node whose text contains the most probes is the gold node (ties broken by the shortest text). If no probe matches, we fall back to the function or class named in the hunk header.
3. New-file patches and non-Python files yield no gold label and are skipped.

The **gold node set** is the union over hunks. 28 of the 30 tasks with graphs received a gold file label; the two exceptions (both fastapi) do not modify an existing Python module that matches the graph. We did not hand-verify all labels; we spot-checked tasks across all three repositories (for example `requests_7205` maps to `requests.utils.get_netrc_auth` and `rich_4006` to `rich.cells.split_graphemes`, matching the edited functions).

### 3.3 Retrievers

All retrievers rank graph nodes; the query is the problem statement concatenated with the hints text.

- **bm25.** Okapi BM25 (k1 = 1.4, b = 0.75) over each node's id plus source text, tokenized with identifier splitting: `get_netrc_auth` and `getNetrcAuth` both yield `get`, `netrc`, `auth` plus the full lowercase identifier. Query term frequency enters as 1 + ln(tf).
- **bm25+graph.** Take the 8 highest-scoring nodes and add 0.35 times each seed's score to every call/import neighbour (in or out edges).
- **bm25+graph(2hop).** The same rule applied for two hops.
- **Test down-weighting.** For all methods, nodes in test modules (`tests.*`, `conftest`, names beginning `test_`) are multiplied by 0.3, because issues often quote test code that should not be edited.

Embedding-based neighbour expansion is implemented in the released script (`--emb`) but not reported here, because the competition's query embedder is not available outside the harness and we did not run it in time for this submission.

### 3.4 Metrics

For k in {1, 3, 5, 10}: **file@k** is the fraction of tasks where some top-k node lies in a gold module; **func@k** is the fraction where some top-k node is a gold node. Both are hit rates ("any gold item found"), not recall of all gold items, which is lenient for multi-node patches. func@k is computed over the 28 labeled tasks.

## 4. Results

### 4.1 Main table (28 tasks, any-hit rate)

| Method | file@1 | file@3 | file@5 | file@10 | func@1 | func@3 | func@5 | func@10 |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.393 | 0.679 | 0.714 | 0.821 | 0.143 | 0.321 | 0.357 | 0.500 |
| bm25+graph | 0.464 | 0.643 | 0.750 | 0.821 | 0.107 | 0.286 | 0.393 | 0.464 |
| bm25+graph(2hop) | 0.393 | 0.714 | 0.714 | 0.786 | 0.071 | 0.286 | 0.357 | 0.464 |

Each task is worth 3.6 percentage points, so every difference between methods in this table corresponds to at most one or two tasks.

### 4.2 Per repository, k = 5

| Repository | n | bm25 file / func | +graph file / func | +graph(2hop) file / func |
|---|---|---|---|---|
| fastapi | 14 | 0.643 / 0.286 | 0.643 / 0.286 | 0.643 / 0.286 |
| requests | 8 | 0.875 / 0.500 | 0.875 / 0.625 | 0.875 / 0.500 |
| rich | 6 | 0.667 / 0.333 | 0.833 / 0.333 | 0.667 / 0.333 |

### 4.3 Findings

**Lexical retrieval is a solid file-level baseline.** With no model, BM25 places a gold file in the top 5 for about 71% of tasks and in the top 10 for 82%. An agent that starts from the top-5 file list begins with the right file in most cases.

**Function level is the bottleneck.** The gold function appears in the top 5 only 36% of the time (top 10: 50%). Half of the tasks therefore still require the agent to search inside the right file or run further retrieval.

**Graph expansion helps little and not reliably.** One-hop expansion raises file@1 by 7 points and func@5 by 3.6 points (one task) but lowers func@1, func@3, and func@10. Two-hop expansion is no better than BM25 and slightly worse at k = 10, consistent with expansion diluting a good ranking with loosely related neighbours. With n = 28 we cannot claim any of these differences is real. The per-repository view shows the single-task gains: requests func@5 (one task) and rich file@5 (one task).

**Repository matters.** requests, a small and cohesive library, is easiest (file@5 = 0.875); fastapi, where bugs often span `routing` and `dependencies` modules with long, similar functions, is hardest (func@5 = 0.286).

### 4.4 Failure analysis

We inspected the labels for the cases that BM25 misses and observed three recurring causes. (a) *Multi-hunk patches:* some patches touch many functions across modules (one rich task changes 20 nodes in 9 modules); an any-hit metric is lenient here but top-k lists rarely cover such breadth. (b) *Class-versus-method granularity:* class nodes appear with stubbed bodies and a method listing, so a class can outrank the method that actually contains the bug. (c) *Issues describing symptoms, not code:* where the issue names no identifier, lexical overlap with the right function is weak, and this is where embedding-based or learned retrieval should help most. We have not quantified the shares of these causes.

## 5. Application: a localization skill for the agent

The competition submission format allows a skill containing scripts. We package the retrieval routine as a script that takes the issue text and the task's graph and prints the top-k files and functions with their neighbours. We do not report end-to-end solve rates for it in this paper; its value to an agent is that it costs no model tokens and, on our measurements, supplies the right file in the first five suggestions in about three of four tasks. Whether this reduces exploration turns enough to raise solve rate with Gemma 4 E4B/E2B is the natural next experiment and is not claimed here.

## 6. Limitations

- **Small sample.** 28 labeled tasks from three repositories; we could retrieve graphs for only 30 of 129 tasks in our run. Confidence intervals would be wide (a hit rate near 0.7 with n = 28 has a 95% interval of roughly ±17 points). Differences between retrievers are not significant.
- **Label noise.** Gold nodes are inferred heuristically from patch text, not annotated by hand, and the any-hit metric does not require covering all changed nodes.
- **Dev set only.** The hidden test set uses different (private) repositories; we cannot say how the numbers transfer.
- **No end-to-end evaluation.** We measure retrieval, not whether the model then writes a passing patch.
- **Fixed hyper-parameters.** We did not tune expansion decay, seed count, or BM25 parameters, to avoid overfitting a 28-task sample.

## 7. Conclusion

Identifier-aware BM25 over the competition's graph nodes is a strong, free file-level localizer and a weak function-level one; a simple fixed graph expansion adds at most a small, statistically unsupported gain. The labels and script we release let other participants test better retrievers (embedding rerankers, learned expansion) against the same targets in seconds, and our results suggest that function-level localization is where such effort would pay off.

## Reproducibility

`python study2.py` (requires a Kaggle API token and the `kaggle` and `numpy` packages) downloads `tasks.jsonl` and graph files, builds labels, and writes `loc_labels.csv`, `loc_results.csv`, and `loc_results_by_repo.csv`. Version used: VERSION 5. All retrievers are deterministic.

## References

- Jimenez, C. E., et al. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024.
- Xia, C. S., et al. (2024). Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489.
- Yang, J., et al. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. NeurIPS 2024.
- Robertson, S., and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4).
