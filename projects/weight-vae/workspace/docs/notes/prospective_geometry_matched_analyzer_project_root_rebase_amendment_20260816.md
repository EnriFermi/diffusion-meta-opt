# Prospective replication: exact-analyzer project-root rebase amendment

Status: frozen before any successful execution of the formal CPU analyzer.

## Trigger and root cause

The first metadata compatibility view is immutable with manifest SHA-256
`b265015ba57592fe2125f4cbe2d999e3d7de6270a2a01091fb97964d9a19a9b7`.
It repaired only the already documented ten-key versus nine-key metadata schema
mismatch. The unmodified frozen analyzer, SHA-256
`94cfc4012f1e64e272d43ff1045a86b2b8d5870777f9c4f14bcdd0e14c25cb8d`,
then advanced one validation stage and stopped before loading either
sufficient-statistics CSV with:

`RuntimeError: contract local file resources.ae_checkpoint lies outside project/runtime roots`

The repository layout explains the failure:

- `/home/coder/project/experiments` is a symlink into
  `/home/coder/project/projects/weight-vae/workspace/experiments`;
- the analyzer defines `PROJECT = Path(__file__).resolve().parents[1]`, so its
  project root becomes the nested `workspace` directory;
- `workspace/artifacts` is a symlink to sibling
  `/home/coder/project/projects/shared/storage/artifacts`;
- the contract correctly records the canonical shared-storage paths, but the
  analyzer's local-file allowlist admits only the nested `workspace` root and
  the Python runtime root.

Thus the analyzer rejects its own formal artifact root after symlink
resolution. The stop occurred in preexecution-contract validation, before the
logged `sufficient_statistic_grid_validation` stage.

## Narrow repair

Do not modify the analyzer source. Create an independent byte-for-byte copy at

`/home/coder/project/_analyzer_compat_20260816/analyze_prospective_geometry_matched_replication.py`

and require its SHA-256 to remain
`94cfc4012f1e64e272d43ff1045a86b2b8d5870777f9c4f14bcdd0e14c25cb8d`.
At that one-directory-deep location, the unchanged expression
`Path(__file__).resolve().parents[1]` evaluates to `/home/coder/project`.
Both the weight-VAE workspace and canonical shared artifact storage then lie
inside the analyzer's existing project-root allowlist. Its resolved design,
source-gain, source-cache, formal-input, and formal-output files remain the
same canonical files as before.

Build a second, fresh compatibility input from the first compatibility view.
The first view remains untouched. Only these administrative bindings may
change:

1. `preexecution_contract.json:/scripts/analyzer/path` becomes the exact-copy
   path above. Its analyzer SHA is unchanged. The resulting contract digest is
   recomputed.
2. `runner_metadata.json:/contract_sha256` becomes that recomputed contract
   digest. No other runner-metadata value changes.
3. `preexecution_binding.json` is updated only as required to bind the amended
   byte-identical-analyzer contract:
   - `scripts/analyzer/path` becomes the exact-copy path;
   - `external_contract_path` becomes a byte-identical amended-contract copy
     inside `/home/coder/project/_analyzer_compat_20260816`;
   - `copied_contract_path` becomes the second compatibility input's own
     `preexecution_contract.json`;
   - `external_contract_sha256` and `copied_contract_sha256` become the amended
     contract digest.
4. Regenerate only the second view's self-excluding artifact manifest.

The analyzer top-level SHA in contract, metadata, and binding stays unchanged.
All three copies of the amended contract must be byte-identical where
applicable: the external copy and input copy are files; the binding embeds the
same `scripts`, resources, dependencies, and panel-cache structures.

Relative to the first compatibility view, exactly 17 of the 20 payload files
must remain byte-identical. The only changed payload files are
`preexecution_contract.json`, `runner_metadata.json`, and
`preexecution_binding.json`. Every scientific input and audit payload remains
byte-identical, including both sufficient-statistics CSVs, latent tensors,
tilings, permutations, prediction manifest, source preflights, model contract,
schemas, and runner log.

## Acceptance checks

- Both earlier inputs remain manifest-exact and unchanged:
  original formal raw `b30b8064...5b80e` and first compatibility view
  `b265015b...a9b7`.
- The exact analyzer copy is a regular independent file, not a symlink or
  hardlink, and is byte-identical to the frozen analyzer.
- Static path simulation and a metadata-only independent reviewer confirm that
  the copied analyzer computes `PROJECT=/home/coder/project`, while all design,
  source-cache, input, output, and contract resources resolve inside that root
  or the Python runtime root.
- Structural diffs contain only the administrative fields enumerated above.
- Exactly 17/20 payloads are byte-identical to the first compatibility view;
  there are no shared inodes or symlinks.
- The amended external and copied contracts are byte-identical and share the
  recomputed digest recorded in metadata and binding.
- The second compatibility manifest is complete and verifies all hashes and
  sizes.
- The exact analyzer completes its own audits, reads the predeclared 864/1728
  grids, computes the frozen criteria without threshold changes, rechecks its
  input manifest at the end, writes seven fixed plots, and records a clean
  source-only seal.
- An independent reviewer recomputes the final decision from the unchanged
  original T/P/D files and audits every real plot before any claim is made.

## Disclosure

This is a second procedural compatibility amendment, so the run is not called
pristine preregistration. It is scientifically lossless only because analyzer
bytes and every scientific input, formula, bootstrap seed, threshold, and
outcome rule remain fixed. Raw point aggregates were already inspected before
this amendment; the exact administrative repair is therefore justified by its
unique path-layout semantics, not by outcome blindness.

No target-domain data are unsealed by this amendment.
