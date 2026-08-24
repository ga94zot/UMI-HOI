# UMI-HOI Development Guide

## General

- Before making any code changes, read the source code, propose a solution, and document it in a file.

- Only modify files relevant to the request.
- If the original code works, do not modify or replace it directly; instead, write a new function. If necessary, you can write a new script. 
- If the workload exceeds two hours, break the task down into three smaller tasks and tackle them one by one. Then write a document summarizing the results and any issues.


## Execution Policy

- Once a task is given, proceed autonomously through analysis, planning,
  implementation, and testing without asking for confirmation at each step.
- Do not pause after presenting a plan unless explicit approval is required by
  the rules below.
- Ask for confirmation only before Git operations or when the requested change
  is ambiguous and could cause significant unintended changes.
- Reading files, searching the codebase, running tests, debugging, and modifying
  relevant source files do not require confirmation.



## Rules
- Do not modify dataset files.
- Do not overwrite checkpoints.
- Keep training commands reproducible.
- Modify the source code; do not delete it directly. If you need to write new code, you can comment out the existing source code.
- Never remove existing functionality.
- Before performing any Git operations, ask for confirmation first.
- When you need to modify the code in a submodule, first comment out the source code, then write an explanation of why you made the changes in the new code.


