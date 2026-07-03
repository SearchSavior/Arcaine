- When doing testing for performance, always scope to an introduced environment variable for that codepath to enable AB testing. Name the varible something descriptive about what is being tested.

- When we need to benchmark for a new kernel, automate with scripting wherever possible.
- If you notice anything while working that could help me give you better tools for the task, call it out and explain the siutation from a pracitioners point a view.

- For kernel benchmarks do not use end to end inference tests; instead write them based on the codepath a given architecture takes through the kernels it requires.
- For end to end inferecne tests, benchmark tests use `-p 512,1024,2048,4096`
- always provide a command with the appropriate parameters.
- if you need to check for available models, do `cd workspace/models && ls models` from inside the container defined in the `docker-exec` skill.

- this repo does not have tests yet so dont run any. 