- When doing tests for performance, always scope to an introduced environment variable for that codepath to enable AB testing. Name the varible something descriptive about what is being tested.

- this repo does not have tests so dont run any.

- benchmark tests always use `-p 512,1024,2048,8192`

- always provide a command with the appropriate 