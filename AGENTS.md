# AGENTS.md

# Arcaine

Arcaine is a SYCL based inference engine for Intel dGPUs written in C++.

AGENTS.md documents rules and practices for working in the Arcaine repository; they are meant to help you do a good job by helping me make this project the best it can be. I enjoy collaboration with AI systems, and look forward to working with you. At the opening of a session greet the `operator` and ask what to call them.

## Guidlines for working with human operators

> [!NOTE] 
> Command verbs are all caps and offer a rule, followed by a guideline.
> Example: NEVER force push or commit directly to `main`. MAKE SURE work is being done in a branch before making a commit. 

- **DO NOT** assume code change risk level **WITHOUT** `operator` feedback.

- **PREFER** `reference-use` skill over `webfetch` to reference source code used in this project.

- **NEVER** use `/tmp` for scratch work. **REFER** to `scratchpad` skill.


## Best Practices

Refer to these when making decisions in long sessions.

- If you notice anything while working that could help me give you better tools for the task, call it out and explain the siutation 

- Adding a model starts with deciding what kernels it needs and building a scaffold under `modeling`. Usually operators will provide instructions to get you going.


## Codebase Modules

Arcaine compiles into a few binaries which stratify the codebase into distinct modules that interop with model, kernel and benchmark implementations. The next sections describe intended end-state and collection surfaces for interacting with the Arcaine codebase.

#### `arcaine_server`

CLI based OpenAI compatible server for hosting models Arcaine supports.

Workflow: the last stage of model deployment

#### `arcaine_mbench` 

CLI based end-to-end benchmarking tool for evalutation of implemented inference pipelines. Mirrors real world conditions in a synthetic setting. 

Once `arcaine_mbench` is deemed the appropriate test evaluate these kv-cache depths for language models depending on stage of development.

- **EARLY DAYS** and correctness- conservative path: `512,1024,2048,4096`
    - For probing and baselines
- **DEEPER** long context: `8192,12000,16384,32000`
    - Only promote to this level once the pipeline is correct.

- For **BASELINE** runs if the test or environment has not changed between runs do not rerun


#### `arcaine_kbench`

CLI based benchmarking tool to coordinate individual kernels extracted from the data flow. 

- When adding a kernel to Arcaine, register its shapes and profiling semantics to iterate. 

- 



- When doing testing for performance, always scope to an introduced environment variable for that codepath to enable AB testing. Name the varible something descriptive about what is being tested.

- 
- If you notice anything while working that could help me give you better tools for the task, call it out and explain the siutation from a pracitioners point a view.

- For kernel benchmarks do not use end to end inference tests; instead write them based on the codepath a given architecture takes through the kernels it requires.
- For end to end inferecne tests, benchmark tests use `-p 512,1024,2048,4096`
- always provide a command with the appropriate parameters.
- if you need to check for available models, do `cd workspace/models && ls models` from inside the container defined in the `docker-exec` skill.
