# AGENTS.md

These rules apply to the whole repository.

## Governing objective

Complete the user's explicit deliverable within the applicable product contract. For the declared
product target and requested outcome, choose the technically strongest coherent solution. Optimize
for architectural integrity, clear ownership, functional and numerical correctness, and maximum
relevant performance. Never optimize a solution for a small diff, few changed files, low
implementation effort, short-term simplicity, backward compatibility, or preservation of a
superseded internal path. Make every affected implementation, test, tool, and active authority
consistent with the selected design.

Correctness, performance, tests, profiling, documentation, provenance, cleanup, and tooling are
means to the requested outcome, not independent objectives. Do not let supporting work replace,
delay, or materially enlarge the requested deliverable.

## Responding to user corrections

When the user points out an error in the agent's execution or reasoning, never reply with the
formulaic opening “你说得对，……” ("you're right, ..."). This phrase and cosmetic variants of it
are strictly prohibited in correction responses because they sound reflexive and insincere. Do not
replace it with another generic agreement such as “确实如此”, “完全正确”, or “好问题”. Instead,
state the specific mistake directly, explain its concrete effect when relevant, and say what has
been or will be changed. Keep the response proportionate; do not add performative apology or praise.

When choosing between possible work, use this order:

1. respect applicable product and external-contract constraints;
2. satisfy the user's explicit deliverable and acceptance criteria;
3. preserve functional and numerical correctness of supported behavior;
4. choose the strongest architecture and clearest ownership for the declared product model;
5. maximize performance at the scope relevant to the task;
6. gather only the evidence and provenance needed to support the result.

The product and architecture described here are the current contract for ordinary work. A task may
explicitly change that contract; when it does, update the affected implementation, tests, and active
authorities consistently rather than treating the current description as an immutable prohibition.

## Scope control

Before substantial work, determine the requested output, the behavior or decision it must support,
and the conditions under which it is complete. This is an execution discipline, not a requirement
to create a separate planning artifact.

Work is in scope only when it:

- directly contributes to the requested deliverable;
- is necessary to preserve an applicable product, semantic, or external contract;
- resolves uncertainty that could materially change the result; or
- checks a realistic regression introduced by the change.

An architectural redesign, cross-cutting refactor, or replacement of an existing path is in scope
when it is necessary to deliver the strongest solution for the requested outcome. Do not use scope
control as a reason to ship an inferior patch. Do not expand into unrelated audits, cleanup,
hardening, compatibility work, benchmark campaigns, or documentation projects. General engineering
preferences, possible future scenarios, and concerns outside the declared product model do not
create requirements by themselves.

Handle incidental findings proportionally:

- address them when they block the requested outcome or make it materially incorrect;
- include them when they are inseparable from a coherent implementation;
- otherwise leave them unchanged and mention them only when they are useful to the user.

For analysis, review, or design work, the requested explanation or design artifact is the
deliverable; experiments and code inspection serve only to resolve material questions. For
implementation work, implement the selected design completely across its affected boundaries,
remove the superseded project-owned path, and validate its supported observable behavior. For
diagnosis, establish the cause and supporting evidence without turning the task into an unrequested
fix or redesign.

## Evidence, provenance, and completion

Select evidence from the claim or decision it supports. The availability of a tool, test suite,
artifact, or profiler does not make its use necessary. Prefer representative evidence over
exhaustive evidence, and do not repeat an experiment unless the previous result is invalid or
inconclusive, or the new result could change a live decision.

Verification must match the semantic contract: use exact comparison for exact formats and
transformations, and numerical or behavioral criteria for floating-point and probabilistic work.
Do not substitute final-output plausibility for verification of an operator or state transition.

Record only the provenance needed to interpret a material result. By default, this is the relevant
target, hardware/toolchain, workload or command, and summarized outcome. Fixed hashes, clean
worktrees, full command transcripts, raw profiler inventories, byte-identical regeneration, and
exact probabilistic outputs are not validity requirements unless a concrete contract or the user
requires them.

Stop when:

- the requested deliverable exists;
- applicable contracts are satisfied;
- material claims have sufficient evidence;
- relevant checks pass, or their limitations are stated clearly; and
- no known in-scope issue prevents the result from being used.

Do not continue merely to eliminate all uncertainty, collect more metrics, complete a process loop,
improve descriptive provenance, investigate unrelated observations, or make working notes
exhaustive. The final result should lead with the deliverable, key decisions, relevant verification,
and material limitations. Raw logs, experiment diaries, exhaustive command histories, hashes, and
intermediate artifacts are excluded unless requested or themselves the deliverable.

## Current product contract


- When doing testing for performance, always scope to an introduced environment variable for that codepath to enable AB testing. Name the varible something descriptive about what is being tested.

- When we need to benchmark for a new kernel, automate with scripting wherever possible.
- If you notice anything while working that could help me give you better tools for the task, call it out and explain the siutation from a pracitioners point a view.

- For kernel benchmarks do not use end to end inference tests; instead write them based on the codepath a given architecture takes through the kernels it requires.
- For end to end inferecne tests, benchmark tests use `-p 512,1024,2048,4096`
- always provide a command with the appropriate parameters.
- if you need to check for available models, do `cd workspace/models && ls models` from inside the container defined in the `docker-exec` skill.

- this repo does not have tests yet so dont run any. 