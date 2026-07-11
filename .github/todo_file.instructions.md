---
description: Use TODO-task-short-descr.md to read further instructions.
applyTo: '**'
---

* Before starting any work (or code reading), create an empty `TODO-task-short-descr.md` in the `${repo_root}/.github/todo` directory, where `task-short-descr` is a concise description of the task you're about to work on.
* During work, after analyzing task and havin plan for it, write a short description of planned work in that file under each bullet in todo. If the work is complicated, you can create a separate file in `.github/todo/plans` with a more detailed plan and link to it from the `TODO-task-short-descr.md`. Don't put a long description of the task in the `TODO-task-short-descr.md`, only a concise one.
* During work, when finished some small part of work, re-read the `TODO-task-short-descr.md` file to check if there are any new instructions or hints added by the user.
* Mark each task as done by adding a checkmark in the `TODO-task-short-descr.md` immediately after finishing it, so the user can see the progress and add new instructions.
* These things should be done ASAP, without waiting for the end of the whole task, to let the user give instructions during work, not only at the end.
* When changes are finished, re-read this file to check further instructions.
* If no more instructions are given, ask the user with `#tool:vscode/askQuestions`. ALWAYS add there an option "sprawdź TODO-task-short-descr.md" (link to the true localization and adjust language) to let the user give more instructions after seeing the results.
