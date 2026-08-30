# Research workspace

This repository is a compatibility-preserving research monorepo. Existing
top-level paths are public interfaces for configs, scripts, notebooks, and old
artifacts, so they intentionally remain in place.

Use [`projects/`](projects/) as the organized entry point:

| Area | Contents |
| --- | --- |
| [`projects/weight-vae/`](projects/weight-vae/) | BigVAE, CELO, weight-latent optimization, and reparameterization research |
| [`projects/cfm/`](projects/cfm/) | CFM, Semicat/BD3LMS, TinyStories evaluation, and paper |
| [`projects/lsdl/`](projects/lsdl/) | LSDL/LFQ notebook work and regression evidence |
| [`projects/unclassified/`](projects/unclassified/) | Independent or unclear work without an established owner |
| [`projects/shared/`](projects/shared/) | Shared configs, scripts, tests, and docs |
| [`projects/storage/`](projects/storage/) | Data, artifacts, logs, and outputs |

Independent projects physically live below their research owner. Compatibility
links preserve former root and `external/` paths. Tightly coupled Weight VAE
components remain at the root and are linked from `projects/weight-vae/`
because their relative paths are runtime interfaces.

Start with:

- [`docs/repository_layout.md`](docs/repository_layout.md) for ownership and
  placement rules;
- [`projects/manifest.json`](projects/manifest.json) for the machine-readable
  map;
- [`docs/agent_operating_manual.md`](docs/agent_operating_manual.md) for the
  research workflow and evidence standard.

The root intentionally contains only repository-level documentation and
dependency metadata. Project exports, loose experiments, logs, and datasets
are owned by their corresponding directories.

## Development workflow

Work in this repository should resemble pragmatic human software development:

- implement the smallest coherent version that can answer the current question;
- run it early on real inputs and fix failures that are actually observed;
- avoid speculative frameworks, defensive layers, provenance machinery, and
  exhaustive validation unless the task has demonstrated a need for them;
- prefer a short working iteration over trying to prove the implementation
  perfect before its first useful run;
- keep test coverage reasonably broad at the level of major behaviors and
  workflows, not one test for every helper or every hypothetical edge case;
- after a change, run focused block-level tests and one appropriate integration
  path rather than repeatedly running the entire repository suite;
- ordinary implementation may use reviewer agents selectively and in parallel,
  without blocking routine progress. Before launching a large, expensive, or
  production-scale experiment, however, an independent reviewer must inspect
  the final setup and give an explicit GO. Preparation may continue while the
  review runs, but the experiment launch waits for that decision.

The default environment is a trusted single-user workspace. Optimize for useful
research progress and readable code. Add heavier safeguards only when there is
a real destructive action, a costly production boundary, or evidence of an
actual failure mode.
