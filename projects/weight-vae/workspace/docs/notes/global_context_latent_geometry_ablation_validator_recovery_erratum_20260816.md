# Global-context latent-geometry ablation: validator-only recovery erratum

Date: 2026-08-16 UTC

## Immutable pre-failure provenance

- Original pre-forward contract:
  `docs/notes/global_context_latent_geometry_ablation_preexecution_contract_20260816.json`
- Original contract SHA-256:
  `0dbe0748743a79c0398b9d0d3c41a0e604ac75d02495a904cbf6312675ff8fac`
- Frozen design SHA-256:
  `6cacb7bb269fd22a7652b07d37e026c8e8e24d6e8023e437656ddad794f1d56f`
- Original runner SHA-256:
  `e15e8efb2a291585ffec1db6ea650ac05fd0ee4ac1149853b54d12637702c6a3`
- Original analyzer SHA-256:
  `05f6f48ebe3ed7711c0769a41becef112359d64583c4072a1f5af45191a214c7`
- Failed private run quarantine:
  `artifacts/crossmodal_united_structure/global_context_latent_geometry_ablation_run_20260816_FAILED_20260816T133703573135Z_pid-3516929`
- Quarantined `failure_record.json` SHA-256:
  `f0b03290033df7c381ebe7e95d12f925f3c2eba4055cc2dd10a3dae0c74db742`
- Quarantined `run.log` SHA-256:
  `bd61f3c5c15616d578ecb5ad9df83018fff34195a7a58f0098d1c71abba8a27b`

The original contract was created and independently audited before the first
Weight-AE forward. The failed run passed the bit-exact known-source numeric
preflight, completed all 2,592 declared representation/cell encode calls and
all 432 CountSketch cells, and then failed at `stage=raw_grid_validation`.
Neither the formal runner output nor the formal analyzer output was published.

## Causal mechanism

The validator constructed `observed["latent_code_width"]` with the Python
expression:

```python
len({int(value.shape[1]) for value in latent_codes.values()}) == 1 and 512
```

Python's `and` returns an operand, not a coerced Boolean. For a uniform
512-wide code grid, the expression therefore returned the integer `512`.
The next predicate used identity comparison, `is not True`, so a correct grid
was guaranteed to be rejected. This is why the terminal diagnostic reported
the superficially contradictory pair `observed=512, expected=512`.

This is a validator type/logic defect. It is not evidence of a malformed code
grid or a scientific failure. Earlier per-batch and assembled-code checks had
already required every code tensor to have width 512.

## Blinding and recoverability

The runner called `verify_raw_grids` before writing `latent_codes.pt`,
`aggregate_features.npz`, `code_manifest.csv`, or
`aggregate_feature_manifest.csv`. Those decision inputs existed only in the
memory of the terminated process and are absent from quarantine. Therefore:

- no latent-derived metric, decision table, bootstrap result, or plot was
  available when this erratum was written;
- the frozen analyzer was never invoked;
- the failed in-memory codes cannot be recovered safely;
- a fresh encoder pass is required.

The quarantined metadata and raw-baseline files remain failure evidence only
and must not be copied into the formal recovery output or used as decision
features.

## Permitted recovery change

The recovery may change only:

1. the raw-grid width observation and its analyzer consumer so both record and
   compare the actual uniform integer width against the frozen expected width
   512 (including fail-closed rejection of the legacy Boolean value);
2. focused no-forward regression coverage for correct, wrong, and mixed code
   widths;
3. the default contract path in runner and analyzer so the original contract
   remains immutable and a separately audited recovery contract is used;
4. immutable provenance wiring that makes this erratum a declared recovery
   resource.

The frozen design, checkpoint, source and target weights, activation
templates, tilings, seeds, representations, model forward path, aggregate
features, statistical methods, success thresholds, and plot definitions must
not change. The formal recovery output must be built fresh; quarantine reuse
is prohibited.
