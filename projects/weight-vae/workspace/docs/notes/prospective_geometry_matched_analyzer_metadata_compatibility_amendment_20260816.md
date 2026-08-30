# Prospective geometry-matched replication: analyzer metadata compatibility amendment

Status: frozen before any successful execution of the formal CPU analyzer.

## Trigger

The formal Weight-AE runner completed successfully and wrote the immutable raw
artifact directory
`artifacts/crossmodal_united_structure/prospective_geometry_matched_replication_20260816`.
Its self-excluding artifact-manifest SHA-256 is
`b30b8064ccd98e6d83b0e557e1267be5cd4abfc0d6e9b5430b2f4c1f5f95b80e`.

The first execution of the exact frozen CPU analyzer
`experiments/analyze_prospective_geometry_matched_replication.py`, SHA-256
`94cfc4012f1e64e272d43ff1045a86b2b8d5870777f9c4f14bcdd0e14c25cb8d`,
stopped before loading either sufficient-statistics CSV with:

`RuntimeError: runner metadata hard_checks exact key set mismatch`

The runner wrote the nine hard-check keys expected by the analyzer plus the
additional successful key
`loaded_local_module_closure_pass: true`. The analyzer's exact-set validator
was not updated when that runner hard check was added. The analyzer self-test
did not contain a positive fixture with the runner's final ten-key metadata
schema and therefore missed this integration mismatch.

## Narrow compatibility transformation

The original raw directory remains untouched. A separate compatibility input
view is allowed with exactly these transformations:

1. Copy every manifest-covered raw artifact byte-for-byte.
2. In the copied `runner_metadata.json`, require that
   `loaded_local_module_closure_pass` exists and is exactly `true`, then remove
   only that key from the copied `hard_checks` mapping. All other JSON values
   must remain structurally equal.
3. In the copied `preexecution_binding.json`, change only
   `copied_contract_path` from the original raw directory's copied contract to
   the compatibility view's byte-identical copied contract. This is necessary
   because the frozen analyzer requires the bound copied-contract path to equal
   `<input_dir>/preexecution_contract.json`.
4. Regenerate only the compatibility view's self-excluding
   `artifact_manifest.json` so it covers the transformed metadata and all
   unchanged files.

The exact frozen analyzer is not modified. The preexecution contract, design,
checkpoint, panels, source gains, scientific formulas, bootstrap seeds and
draw count, pass thresholds, outcome precedence, geometry protocol, and all
scientific raw inputs remain unchanged.

In particular, these files must be byte-identical between the original raw
directory and the compatibility view:

- `weight_sufficient_stats.csv`
- `operator_sufficient_stats.csv`
- `correct_latent_codes.pt`
- `correct_latent_code_manifest.csv`
- `tiling_indices.pt`
- `tiling_manifest.csv`
- `permutation_manifest.csv`
- `prediction_manifest.csv`
- `preexecution_contract.json`
- `known_source_numeric_preflight.json`
- `decoder_entry_parity.json`
- `loaded_local_module_audit.json`
- `source_template_audit.json`
- `model_contract.json`
- `resolved_config.json`
- `schema.json`
- `sufficient_statistics_schema.json`
- `run.log`

## Required checks before accepting the amended analysis

- The original raw manifest still verifies exactly and its directory hash set
  is unchanged after analysis.
- The compatibility transformation is deterministic, fail-closed, and records
  old/new SHA-256 and byte counts for both changed JSON files.
- A structural diff proves that the runner metadata change is the deletion of
  exactly one true key and that the binding change is exactly one path value.
- All scientific and audit files listed above are byte-identical.
- The compatibility manifest is complete, contains no symlinks, and verifies
  all hashes and sizes.
- The exact frozen analyzer SHA remains
  `94cfc4012f1e64e272d43ff1045a86b2b8d5870777f9c4f14bcdd0e14c25cb8d`.
- The analyzer completes its own runtime, contract, manifest, sufficient-stat,
  permutation, algebra, geometry, source-only-seal, and end-of-run immutability
  checks.
- An independent reviewer audits the compatibility transformation and final
  outputs.

## Disclosure and scope

Before this amendment was frozen, an independent raw audit had already
computed non-gating point aggregates and flagged likely per-role failures.
Therefore the amendment is not described as outcome-blind. Its admissibility
rests on being a uniquely determined schema projection that cannot alter any
scientific statistic, threshold, bootstrap, or decision rule.

This amendment does not unseal target-domain data and does not convert the
two-panel source-replication gate into evidence of cross-domain transfer.
