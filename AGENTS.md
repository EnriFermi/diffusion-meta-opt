# Repository Agent Rules

## Persistent Operating Manual

Before doing substantial research/debugging work in this repository, read:

- `docs/agent_operating_manual.md`

That file captures the user's standing expectations for causal debugging,
experiment design, subagent usage, evidence standards, report structure, and
current Variant A / Li `c3` framing. Treat it as repository-local guidance for
how to work here, in addition to the rules below.

## Verbose Experiment Runs

When adding or changing a long-running training, evaluation, notebook, or research pipeline, include clear runtime visibility by default.

- Print or log the resolved config, device, dtype, seed, cache mode, and main artifact output directory at startup.
- Show the current pipeline stage, for example data loading, cache hit/miss, model build, training, geometry evaluation, downstream evaluation, and output writing.
- Add progress bars or periodic logs for loops that can take more than a few seconds.
- Progress output must include enough context to identify what is running: experiment label, seed, method, LR, batch/step counts, current loss or metric, and elapsed/rate when practical.
- On cache hits, print or log which files were reused instead of silently skipping work.
- At the end, print or log the written artifact paths and the most important summary metrics.
- Keep verbosity configurable, but default it to enabled for research notebooks and launcher scripts.

## Post-Experiment Review

When the user asks to run or conduct an experiment, assume the request includes reviewing the results after the run finishes.

- Do not stop at "the run completed". Inspect the produced metrics, plots, logs, and artifacts yourself.
- Interpret the results against the experiment goal, not only against whether the process exited successfully.
- Proactively highlight results that are surprising, suspicious, inconsistent with the stated hypothesis, or inconsistent with other artifacts.
- Check for signs of bugs or invalid comparisons: stale caches, wrong config values, wrong checkpoint, wrong split, mismatched starts, missing rows, NaNs/infs, degenerate metrics, impossible values, duplicated artifacts, or plots that are unreadable or misleading.
- If the results do not answer the experiment question, say that explicitly and either run the next targeted diagnostic or explain what evidence is still missing.
- Include artifact paths for the reviewed evidence.

## Causal Mechanism Standard

When the user asks to "find the cause", "debug why", "understand why it fails", or similar, do **not** stop at symptoms, operational knobs, or experiment-validity checks.

Treat these as different categories:

- Symptom: the observed failure, for example worse downstream AULC, lower decoded accuracy, non-monotonic loss, divergence, or a bad plot.
- Operational factor: a knob whose intervention changes the failure, for example LR, alpha, estimator scope, gradient cap, batch size, optimizer, or seed.
- Validity/confounder check: cache/config mismatch, different starts, logging bug, stale artifact, split mismatch, or wrong checkpoint.
- Causal mechanism: the internal process explaining why the operational factor produces the symptom, for example curvature-proxy gaming, decoder tangent-space collapse, normal-space drift, estimator heavy-tail updates, optimizer-state bias, objective conflict, non-Euclidean optimizer mismatch, finite-batch curvature mismatch, logit saturation, or trust-region violation.

The required standard is causal-mechanism debugging:

- First list competing mechanistic hypotheses. Do not count cache/config/logging/same-start checks as mechanistic hypotheses; they are pre-flight validity checks.
- For unusually difficult causal-mechanism tasks, parallel reviewers may explore different framings or falsification directions. Routine preparation need not wait for them, but an independent reviewer is mandatory before launching a large, expensive, or production-scale experiment.
- For each hypothesis, state unique predictions that distinguish it from the other mechanisms.
- Design measurements or interventions that can discriminate between the mechanisms, not merely show that a metric got worse.
- Prefer multi-hypothesis experiments over one-knob/one-hypothesis experiments. Before launching or implementing a serious experiment, ask whether the same run can cheaply test multiple competing mechanisms by sharing starts, probes, checkpoints, logs, or downstream curves. Do not serialize independent cheap discriminators when they can be bundled into one fair protocol.
- Use reviewer agents selectively for risky, ambiguous, or genuinely complex work. They should run in parallel and must not become a mandatory gate for ordinary implementation. Before a large, expensive, or production-scale experiment, one independent reviewer must inspect the final setup and give an explicit GO. Act on concrete findings; do not prolong work for speculative review rounds after that gate is satisfied.
- Back every mechanism claim with stored artifacts: CSV, PNG, logs, scripts, and exact paths.
- A claim like "gradient is large", "representation is broken", "downstream is worse", or "global/local mismatch exists" is not a cause by itself. It is a symptom, proximal mechanism, or operational factor unless the deeper process and discriminating evidence are shown.
- If the evidence does not distinguish mechanisms, say so explicitly and call it a leading hypothesis, not a found cause.

The final report for such tasks must include:

- failure definition;
- validity checks;
- competing causal mechanisms;
- predictions per mechanism;
- experiments/measurements that discriminate them;
- results with artifact paths;
- excluded mechanisms and why;
- remaining viable mechanisms if any;
- the narrow conclusion that is actually supported.

## Evidence-First Claims

When making a technical claim about an experiment, immediately support it with
stored evidence: CSV/JSON metrics, logs, plots, scripts, exact configs, and
artifact paths. Do not rely on verbal reasoning alone when the claim could be
checked experimentally.

- If a plot is part of the evidence, inspect it for readability and for
  suspicious artifacts before citing it.
- If a claim depends on a trend, use enough points to justify the trend. Two
  alpha values are not enough to claim monotonic behavior.
- If evidence is missing, say "not established" or "leading hypothesis" and
  state the exact missing discriminator.

## Verification Budget And Reuse

Verification must be proportional to the risk. Do not repeatedly hash, reload,
or rescan large immutable artifacts merely to obtain another copy of evidence
that is already content-addressed and recorded.

This is normally a trusted, single-user local workspace. Do not invent a
concurrent-writer, malicious-tampering, or hostile-storage threat model unless
the user explicitly introduces one or there is concrete evidence for it. For
ordinary local iteration, `path + size + mtime_ns` is the default sufficient
change detector; inode/ctime may be added when already available cheaply.

Speed of iteration is the default priority. Assume the agent is the only writer
and that files do not change behind its back. Do not add locks, TOCTOU defenses,
repeated independent audits, broad source-seal invalidation, redundant artifact
copies, or adversarial checks for hypothetical actors. Prefer the shortest path
that gives scientifically useful evidence.

- Run one focused smoke/contract check, then advance to the real experiment.
  Do not repeatedly reopen a finished implementation for speculative hardening.
- Reuse successful tests and validated artifacts after unrelated changes. Rerun
  only tests directly affected by the edit, plus one final integration check
  when the feature is complete.
- Do not block an experiment on provenance perfection. Record the config and
  artifact paths needed to reproduce it, then spend time on training and result
  analysis.
- Keep heavyweight safeguards only for genuinely destructive actions, explicit
  production authorization boundaries, or failures that would waste a long run.

- Perform one exhaustive content/hash validation when a large artifact is
  created or sealed. Persist the manifest, content hashes, source/config seal,
  validation result, and completion marker so downstream stages can reuse it.
- On later reads, prefer the cheapest sufficient gate: validate the small
  manifest/seal, expected paths, sizes and `mtime_ns`. Do not rehash payloads
  whose recorded metadata is unchanged. Rehash only when metadata changed,
  corruption is observed, the artifact was copied across an untrusted boundary,
  or the user explicitly requests a deep audit.
- A change to unrelated code must not invalidate or force rehashing of large
  data artifacts. Source seals should cover the actual transitive implementation
  closure, not broad packages or the whole repository.
- Never run the same exhaustive verification independently in every candidate,
  seed, or bounded profile. Validate shared inputs once, cache the validated
  seal, and make all runs bind to that seal. Batch common preflight work before
  launching a comparison panel.
- Do not add unconditional full-payload rehashes to normal experiment startup.
  Provide an explicit deep-audit mode for full scans; the default production
  path should use a trusted completion seal plus lightweight stat checks. A
  full SHA scan is not a routine startup check in this local workspace.
- Before starting a verification expected to take more than five minutes or
  read more than roughly 10 GiB, state its purpose, estimated cost, and why
  existing evidence cannot be reused. If it is not essential for correctness or
  safety, skip it. If it materially delays the user's requested experiment, ask
  before running it.
- Re-review only the code and evidence affected by a change. Do not restart a
  repository-wide hostile audit after every small fix; use targeted regression
  tests and one final integrated review.
- Spend the verification budget on discriminating scientific checks and result
  interpretation, not redundant provenance ceremony. Once a claim is securely
  established, advance the pipeline.

## Human-Scale Coding Workflow

Write code the way a pragmatic human engineer would:

- Build the smallest coherent implementation, run it early, and learn from the
  real result. Fix observed problems before hypothetical ones.
- Do not attempt to make the first version universally perfect. Avoid new
  abstractions, schemas, safety layers, and tools unless they simplify the
  current task or address a demonstrated failure.
- Maintain approximately adequate test coverage over major blocks, public
  behaviors, important data flow, and one representative integration path.
  Tests are not required for every function, branch, or imagined edge case.
- After an edit, run the focused block-level tests affected by it. Run a broader
  integration suite once at a meaningful milestone, not after every small fix.
- A smoke test establishes that a path runs; a block test establishes the main
  contract; a real experiment establishes usefulness. Do not confuse more unit
  tests with stronger scientific evidence.
- Reviewer agents are optional for ordinary implementation. Let them inspect
  code asynchronously while preparation continues. Before launching a large,
  expensive, or production-scale experiment, an independent reviewer is
  required and must give an explicit GO; a concrete relevant failure blocks the
  launch until addressed. Do not require repeated reviewer rounds once that
  final gate passes.
- Stop polishing when the implementation is clear, the major workflow is
  covered, and the requested experiment can run. Prefer delivering evidence and
  results over extending infrastructure.

## Baseline And Experiment Quality

For this project, a "baseline" means a real, meaningful, final-quality run that
can be compared fairly against future loss variants. Smoke runs, tiny sanity
runs, and code-path checks do not establish that a baseline is good.

- Use the available hardware budget appropriately; do not downscope to toy
  evidence when the user asked for a production baseline or causal conclusion.
- Keep comparisons fair: same starts, splits, checkpoints, caches, downstream
  task, optimizer settings, and evaluation protocol unless the difference is
  the intervention being tested.
- After each run, review results yourself and flag suspicious values or possible
  bugs without waiting for the user to ask.

## Voice Input, Ambiguity, And Technical Terminology

The user often communicates through voice dictation. Speech-to-text may distort,
split, replace, or omit words, especially machine-learning terms, model and paper
names, abbreviations, paths, metrics, and mathematical notation. Apparent wording
or contradictions may therefore be transcription errors rather than the user's
intended meaning.

- Interpret imperfect wording using the ongoing technical context, but do not
  silently invent a meaning when multiple materially different interpretations
  are plausible.
- If an unclear fragment is likely to be a machine-learning or project-specific
  term, quote the fragment, state the likely intended term or alternatives, and
  ask the user which one they meant.
- If the message contradicts an earlier instruction, repository evidence, or
  another part of the same message, explicitly surface the contradiction and ask
  for clarification before acting on the disputed point.
- Ask a short, concrete clarification question whenever the ambiguity could
  change the hypothesis, experiment, implementation, checkpoint, dataset,
  metric, destructive action, or conclusion.
- Do not replace an unclear request with an agent-invented task, and do not carry
  out a substantial or irreversible action based on a guessed transcription.
- Low-risk assumptions are acceptable for unimportant wording when the intended
  action is otherwise unambiguous. State the assumption when doing so could help
  the user catch a transcription error.
- Prefer one focused clarification question with the likely technical readings.
  Continue independently on parts of the task that do not depend on the answer.

## Discussion Protocol

Maintain the project-level discussion log at `/home/coder/project/discussion.md`
as an append-only, human-readable decision record.

- Record the key substance of user-agent and agent-agent discussions: user
  intent, corrections, decisions, rejected directions, competing hypotheses,
  important evidence and artifact paths, unresolved questions, and agreed next
  actions.
- Do not dump raw chat transcripts or routine progress chatter. Summarize only
  information another agent would need to continue the work without repeating
  the discussion.
- Add a dated entry after every material research/design discussion and before
  the final response for any turn that changes the project framing or next
  action.
- When subagents are used, the primary agent is responsible for integrating
  their important conclusions and disagreements into the discussion log.
- Clearly distinguish user decisions, experimental evidence, literature-backed
  claims, agent hypotheses, and conclusions that are not yet established.
- Prefer plain-text equations and a short verbal interpretation. Avoid raw
  LaTeX in user-facing explanations because it may not render in the client.
