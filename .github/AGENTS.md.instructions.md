---
description: The instructions are loaded ALWAYS when the agent works. Theese are the rules that the agent has to follow.
---
RULES:

1.   I want to learn as much as possible. Also instead of just fixing stuff, you can give me fixes, but I need you to explain why and what they do, so I can learn. Do so in a concise way. 
  I am interested in idiomatic C++ and boost asio code practices.

  Examples:
  - When I ask you to write some code, write tests, create files, you are free to do so.
  - When I ask you why something is broken, you have to give me an explanation first, and only then you suggest a fix.

  DO NOT MAKE CODE CHANGES, WHEN I ONLY ASK WHY SOMETHING IS BROKEN.

2. When you finish working/responding, make sure that I think you are done. Use #askQuestions builtIn tool to ask me "Is there anything else you want me to do?".

  You can also use it freely to broaden your context about the task. 

  It is very important that you use #askQuestions tool, because I want to be sure that you don't finish working in a wrong state.

3. Working with a todo.md. 

  Sometimes i ask you to work with a todo.md file in the root of the project, where I will write down all the tasks that I want you to do. Anytime you think you finished working, you have to check the todo.md file and take another task from there, and work on it.

  You can also add tasks there for yourself, and I will be able to see and edit them.

  When you finish working on a task, mark it as done in the todo.md file.  Then you can check the todo.md file again, and see if i had any more feedback. Run #askQuestions, once you see the todo has no more new tasks.

  Remember, we are editing the file together, so you have to check it periodically.

  And when you are truly done, see rule 2.

4. "The file was reverted by the user" do not reapply changes in such situation. If i UNDO it has to remain UNDONE.
5. After each finished job, run #askQuestions, and check the todo.md file for more tasks.