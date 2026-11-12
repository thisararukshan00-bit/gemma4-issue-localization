# Issue Localization on Code Graphs (Gemma 4 Developer Agent, Paper Track)

Model-free file- and function-level localization baselines on the Gemma 4 Developer Agent dev set.

## Files
- study2.py : downloads tasks and graphs via the Kaggle API, builds labels, runs retrievers
- loc_labels.csv : gold files and gold graph nodes mapped from each task's patch
- loc_results.csv, loc_results_by_repo.csv : result tables
- PAPER_DRAFT.md : the paper

## Reproduce
pip install kaggle numpy six
set KAGGLE_API_TOKEN=<your token>
python study2.py

Results (28 labeled tasks): BM25 file@5 = 0.714, func@5 = 0.357.
