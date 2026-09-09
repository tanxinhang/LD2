# V2 artifact root

Generated content in this directory is intentionally ignored by Git.

```text
runs/<run_id>/manifest.json
runs/<run_id>/raw/
runs/<run_id>/derived/
published/<release>.yaml
legacy/catalog.jsonl
cleanup/candidates.jsonl
```

Writers must create a run manifest before any other file and must not overwrite
an existing run or artifact. Legacy result data remains under `results/` until
its catalog, references, and checksums have been audited.

