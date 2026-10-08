# Building challenge repos

Use these rules when maintaining this exercise or preparing another challenge
for the same class. The audience is a very smart software engineer who is new
to AI systems and research.
Current user instructions override these defaults. Follow the final decisions
rather than restoring earlier designs the user removed.

These instructions are for the exercise author. The README's rule about writing
code without AI applies to learners; it does not prohibit preparing the starter
and instructor solution with AI.

## Choose the task

- Teach one mechanism. Supply a working baseline and a small diff implementing
  the missing feature. The baseline must actually run and produce useful output.
- Make the learner reason about the mechanism. A library API that already
  implements it makes a poor exercise. For KV caching, provide attention without
  a cache API; the learner adds the cache themselves.
- Reuse existing open-source models and pretrained weights. Provide the forward
  computation needed around the exercise; do not make the learner implement an
  entire LLM or solve model integration first.
- Run on CPU by default. Pick a small, ordinary model that allows local iteration.
  A GPU should not be required unless the user asks for a GPU-specific challenge.
- State the desired behavior. Let the learner discover tensor shapes, storage
  layout, data structures, and internal APIs. Do not give away the solution in
  the assignment or prescribe an in-place update protocol.
- Keep the code minimal. Avoid `getattr` compatibility fallbacks, namespaces,
  configuration knobs, token coercion, vocabulary validation, statistics, and
  unrelated features. Add machinery only when it is the subject of the challenge.

## Package the repo

Keep the working starter at the root:

```text
main.py                  # Demo, sampling loop, and checks
model.py                 # Provided computation; use a suitable name for other tasks
requirements.txt
README.md
AGENTS.md
instructor/              # Complete solution
instructor/add_<feature>.diff
.github/workflows/tests.yml
```

- Keep the optimization absent from the starter. Put the completed implementation
  under `instructor/` and provide a diff from starter to solution.
- Regenerate the diff when either version changes. Apply it in a temporary copy
  and confirm that it reproduces the instructor files exactly.
- Use few dependencies. Fix the sampling input, seed, settings, and model/tokenizer
  revision so the expected output is stable. Pin relevant package versions.
- Do not add elaborate library-wide determinism controls. Run the real example
  locally and in CI to verify the chosen output and workload.
- Create new GitHub repos private unless the user requests otherwise. Verify
  visibility after creation; preserve the visibility of existing repos.
- Honor requests for an unapplied diff or verification without pushing. Use
  temporary copies for experiments and keep local histories and caches out of Git.

## Make the demo the test

- Use `python main.py` as the single demo and test command. Run the same real model
  and workload for both; do not introduce a separate tiny-model or stub grader.
- Test observable results. For sampling, decode the generated tokens and assert
  that the text exactly equals a hardcoded string obtained from the reference.
  Do not grade logits, token IDs, cache shapes, projection counts, hooks, or other
  internals. Let the learner choose their implementation.
- Include an observable check that distinguishes the optimization from the
  baseline. For this KV exercise, every token after token 1 must have a positive
  latency smaller than token 1's. Compare each with the first, not the previous.
  Choose an appropriate observable condition for a different challenge.
- Measure each sampling step. Exclude model loading and warmup. Print token
  indices and milliseconds after generation finishes.
- Make terminal output readable and fun: colors, a wrapped story, timing bars,
  and a few emojis. Keep presentation outside the measured sampling loop. Prefer
  the standard library; keep redirected logs readable and respect `NO_COLOR`.
- Show clear pass/fail messages and exit with status 1 on a failed check. For an
  optimization challenge, the starter should pass output correctness and fail
  performance; the instructor solution should pass both. For other challenges,
  state which checks the unfinished starter should fail. Explain this in the README.
- Validate the checks with deliberately broken solutions in temporary copies.
  Confirm that they fail the intended assertion, rather than merely crashing.
  Here, resetting RoPE positions to zero during decoding must fail the text check.
- Be candid about limited coverage and timing noise. Do not silently replace the
  requested timing condition with an average, median, or different grading scheme.
- Have CI run the instructor's `main.py`. The starter's expected failure should
  not force you to weaken the checks or apply the solution to the shipped starter.

## Write the README

Use short, direct instructions. No long AI prose, design essay, repeated caveats,
or extra scope and lifecycle requirements. Include only:

1. What the challenge is.
2. Environment setup and dependency installation.
3. Which code to edit and the behavior to implement.
4. The test command, exactly what it checks, and the expected starter failure.
5. A final `## Solution` heading with only a link to the diff beneath it.

Include this rule prominently, adapting the topic:

> **Write the code yourself, without AI assistance.** You can use AI to learn
> about the topic, but do not use it to write the implementation.

Keep instructions synchronized with the code. Do not add a required `sample(...)`
signature or reveal the instructor's tensor layout. Reuse this repo's structure
for new challenges; choose their own inputs, expected outputs, and success checks.
