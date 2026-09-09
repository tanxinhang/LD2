# V2 artifact root

Generated content in this directory is intentionally ignored by Git.

```text
runs/<run_id>/manifest.json
runs/<run_id>/completion.json
runs/<run_id>/raw/
runs/<run_id>/derived/
published/<release>.yaml
legacy/catalog.jsonl
cleanup/candidates.jsonl
```

Writers must create a run manifest in `created` state before any other file and
must not overwrite an existing run or artifact. A terminal `completion.json`
is written only after hashing and verifying every declared output. Legacy result data remains under `results/` until
its catalog, references, and checksums have been audited.

Generated runs remain ignored by default. A completed formal release may be
versioned only through an explicit path allowlist in `.gitignore`; its registry
entry must bind the result envelope and paired CSV by SHA-256. This keeps a
clean clone scientifically auditable without committing the historical result
tree.
