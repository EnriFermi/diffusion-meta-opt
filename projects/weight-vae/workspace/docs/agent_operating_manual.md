# Agent Operating Manual

Date created: 2026-07-06.

Purpose: persistent handoff for future agents after context resets. This file
records how the user expects agents to work in this repository. It is not a
result report. For Variant A experiment results, start from
`docs/reparam_preconditioning_experiments/variant_A_causal_debug/knowledge_base/README.md`.

## Read Order For A New Agent

1. `AGENTS.md`
2. This file.
3. For Variant A / Li `c3` work:
   `docs/reparam_preconditioning_experiments/variant_A_causal_debug/knowledge_base/README.md`
4. For the cross-domain Weight-AE / global-context experiment:
   `docs/notes/crossmodal_united_structure_decision_log_20260816.md`. Read it
   through the final "Latest canonical status" section; earlier sealed/NO-GO
   entries are historical and were later violated by an invalid geometry run.
5. For the Weight-AE conditional-generation workshop submission:
   `docs/weight_operator_diffusion_workshop_bible_20260817.md`. This is the
   canonical paper direction, fairness contract, risk register, and staged
   go/no-go plan. Its explicit open decisions must not be silently guessed.
6. The specific report or script relevant to the current request.

## Interaction Contract

- Respond to the user in Russian unless there is a concrete reason not to.
- Be direct, technical, and evidence-oriented. Avoid fluff, reassurance, and
  generic summaries.
- The user may use profanity. Do not moralize or derail; extract the technical
  requirement and execute it.
- If the user asks for status, give the current state, blockers, artifacts, and
  next action. Do not pretend a weak result is conclusive.
- If the user says a prior answer was handwavy, treat that as a signal to return
  to artifacts and discriminating experiments, not to rephrase the same claim.

## What "Find The Cause" Means Here

The user does not accept symptoms, operational knobs, or localization as a
cause. A valid cause must be a causal mechanism.

Use these distinctions:

- Symptom: the observed failure, for example downstream worse, non-monotonic
  loss, bad decoded accuracy, divergence, or unreadable plots.
- Operational factor: a knob or condition that changes the symptom, for example
  LR, alpha, estimator scope, clipping, batch size, optimizer, or architecture
  submodule.
- Localization: identifying where the symptom appears, for example a specific
  layer, head, block, or latent component.
- Validity check: excluding stale cache, wrong config, wrong checkpoint, wrong
  split, mismatched starts, logging bug, missing rows, NaNs, duplicated
  artifacts, or broken plots.
- Causal mechanism: the internal process explaining why an operational factor
  produces the symptom, and why alternative mechanisms are less consistent with
  the evidence.

Examples of mechanism-level explanations in this project:

- surrogate/proxy optimizes a loose upper bound while exact downstream-causal
  `c3` remains high;
- HVP estimator heavy tails produce trust-region violations that damage decoded
  starts;
- decoder chart tangent space fails to cover raw optimizer directions, so latent
  updates cannot realize the useful part of the weight-space step;
- latent Adam is the wrong optimizer for the learned metric, while a projected
  or natural step would behave differently;
- objective terms fight over reconstruction vs curvature proxy in a way that
  changes logits/margins rather than downstream-useful curvature.

## Causal Debugging Standard

For any request like "why does this fail", "find the cause", "debug the
reason", or "understand why downstream is worse":

1. State the failure precisely.
2. Run pre-flight validity checks, but do not call them causal hypotheses.
3. List multiple competing mechanistic hypotheses.
4. For each hypothesis, write unique predictions that distinguish it from the
   others.
5. Design measurements or interventions that discriminate mechanisms. Avoid
   testing one knob at a time when several discriminators can run in parallel.
6. Back every claim with stored artifacts: CSV/JSON, plots, logs, exact configs,
   scripts, and paths.
7. If evidence does not distinguish mechanisms, say so. Use "leading
   hypothesis", not "found cause".
8. If a fix is possible, implement the fix and rerun the fair comparison. If
   not fixable by the agent, explain why and what external capability is needed.

Do not return to the user with unsupported causal prose. Every nontrivial claim
should have a concrete artifact path or should be explicitly labeled as
unproven.

## Subagent Policy

The user strongly prefers subagents for causal mechanism work and explicitly
expects them when the task asks for broad exploration or independent review.

- For causal mechanism debugging, spawn multiple subagents with different
  initial framings so they explore different hypothesis spaces.
- Give subagents self-contained tasks, clear artifact paths, and permission to
  inspect code/results. If they run experiments, require logs and artifact
  paths.
- Use at least one reviewer/critic subagent for important conclusions. The
  reviewer should try to falsify the main agent's claims and check for stale
  caches, unfair comparisons, and unsupported trends.
- If the user explicitly says "do not do this yourself, use a subagent", obey.
  The main agent may coordinate and verify the result, but should not secretly
  redo the delegated work as the primary author.
- Close completed subagents when no longer needed.

## Experiment Standard

When the user asks for an experiment, the request includes analysis after the
run. A completed process is not a completed experiment.

Required behavior:

- Print/log resolved config, device, dtype, seed, cache mode, and output
  directory at startup.
- Log pipeline stages: data loading, cache hit/miss, model build, training,
  geometry evaluation, downstream evaluation, and output writing.
- Use progress bars or periodic logs for long loops.
- Include method label, seed, LR, step counts, current loss/metric, and elapsed
  time when practical.
- On cache hits, report exactly which files were reused.
- At the end, report artifact paths and key metrics.
- Inspect metrics and plots yourself.
- Flag suspicious values, unreadable plots, inconsistent trends, stale caches,
  wrong configs, NaNs/infs, duplicated rows, missing rows, and impossible values.

For baselines:

- A baseline is a meaningful final-quality run, not a smoke test.
- Small sanity runs only prove code execution. They do not prove that the
  baseline is good.
- The machine has strong GPU capacity available for this project; do not use
  toy runs as evidence when the user asked for a real baseline or a causal
  conclusion.

For claims about trends:

- Do not claim monotonicity or an alpha dependence from two points.
- If a plot is evidence, inspect the plot and check readability before citing
  it.
- If a result is surprising, investigate instead of smoothing it over in prose.

## Reporting Standard

Reports should let another model or agent reconstruct what happened.

Include:

- failure definition and success criterion;
- implementation details, exact loss formulas as implemented, scripts, configs,
  commands if available, seeds, splits, starts, checkpoints, and cache mode;
- tables of metrics with source paths;
- plots that were inspected for readability;
- causal hypotheses and discriminating predictions;
- experiments run, results, conclusions, and caveats;
- excluded mechanisms and why;
- remaining gaps;
- next experiments that would actually discriminate mechanisms.

Do not write a report that only says "ran X and got Y". It must explain what X
tested, why that test discriminates mechanisms, and what remains unproven.

## Variant A / Li c3 Working Framing

Current project-specific framing:

- Variant A / `li_a_hvp` should be treated as an attempted upper-bound/proxy for
  Li-style `c3`, not as the exact Li/Kron/PSGD criterion.
- If downstream does not improve, the supported interpretation is not "Li/Kron
  is bad". The more appropriate family of explanations is:
  - the implementation did not reduce the relevant exact/downstream-causal `c3`
    enough;
  - the surrogate or upper bound is too loose or not faithful enough;
  - the downstream optimizer/harness does not realize the Li/Kron mechanism;
  - the learned VAE chart cannot express the useful preconditioned directions.
- Current A proxy reductions are not reductions by orders of magnitude. Do not
  claim the Li criterion was optimized to the useful regime unless exact/CG
  evidence shows that.
- Old A harness evidence is known to be technically contaminated by global vs
  local estimator mismatch and uncapped relative A-gradient pressure.
- Local estimator plus relative grad-ratio cap stabilized the old regression,
  but did not establish a downstream win.
- Lower proxy/exact-A alone is known to be insufficient: an old local no-cap
  run reduced exact-A strongly while worsening downstream.

Important current gaps:

- paired current A/control exact-CG or full-space damped `c3` audit;
- tightness check: A surrogate vs exact `c3`, with forward and inverse/barrier
  terms separated;
- direct PSGD/Kron positive-control downstream run;
- full-curve metric-aware or projected downstream run, not just one-step probes.

Primary knowledge base for these facts:

- `docs/reparam_preconditioning_experiments/variant_A_causal_debug/knowledge_base/README.md`
- `docs/reparam_preconditioning_experiments/variant_A_causal_debug/knowledge_base/experiment_registry.md`
- `docs/reparam_preconditioning_experiments/variant_A_causal_debug/knowledge_base/causal_hypotheses.md`
- `docs/reparam_preconditioning_experiments/variant_A_causal_debug/knowledge_base/artifacts_index.md`
- `docs/reparam_preconditioning_experiments/variant_A_causal_debug/knowledge_base/agent_onboarding.md`

## File Organization

Documentation and navigation:

- `docs/reparam_preconditioning_experiments/`
- `docs/reparam_preconditioning_experiments/variant_A_causal_debug/`
- `docs/reparam_preconditioning_experiments/variant_A_causal_debug/knowledge_base/`

Scripts:

- `scripts/`
- relevant examples include Variant A downstream, metric-aware, PDF review, and
  CG audit scripts listed in the knowledge base.

Heavy artifacts:

- `artifacts/loss_landscape_analysis/`

Core code:

- `post_train_research/`

Do not move or delete old artifacts unless the user explicitly asks. Prefer
creating indexes and manifests over destructive cleanup.

## Common Failure Modes To Avoid

- Calling a symptom a cause.
- Calling a knob a cause without explaining the mechanism.
- Treating cache/config checks as causal hypotheses.
- Making a broad Li/Kron claim from the current VAE surrogate.
- Saying a surrogate "does not correlate with downstream" without checking
  whether the surrogate actually reached the required scale or whether the bound
  is tight.
- Running one hypothesis at a time when several cheap discriminators could run
  in parallel.
- Returning before inspecting generated plots.
- Reporting only medians when tails/outliers determine failure.
- Hiding uncertainty behind confident prose.
- Using small smoke runs as proof of a good baseline.

## Minimal Handoff Template For Future Work

When starting a new serious task, write down:

```text
Task:
Success criterion:
Current evidence:
Validity checks:
Competing mechanisms:
Discriminating experiments:
Artifacts to produce:
Subagents to spawn:
Reviewer criteria:
```

When finishing, report:

```text
What changed:
Experiments run:
Key metrics:
Plots reviewed:
Supported conclusion:
What is not established:
Artifact paths:
Next discriminating step:
```
